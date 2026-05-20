from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .assets import AssetLayout
from .attacks import ATTACKS, multimodal_asymmetric_pgd
from .compression import POSTPROCESSORS, PostprocessPlan, apply_postprocessor
from .manifests import ManifestRecord, load_manifest
from .metrics import classification_metrics, confidence_interval95
from .modeling import build_adapter, list_active_models
from .progress import log, progress_iter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(records: list[ManifestRecord], batch_size: int) -> Iterable[list[ManifestRecord]]:
    for index in range(0, len(records), batch_size):
        yield records[index:index + batch_size]


def _num_batches(record_count: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return (record_count + batch_size - 1) // batch_size


def _split_records(records: list[ManifestRecord], split: str) -> list[ManifestRecord]:
    return [record for record in records if record.split == split]


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                log(f"[resume] Ignoring truncated JSONL line {line_number} in {path.resolve()}.")
                break
    return rows


def _write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _canonical_kwargs(kwargs: dict) -> str:
    return json.dumps(kwargs, sort_keys=True, separators=(",", ":"))


def _fingerprint_payload(payload: dict) -> str:
    return hashlib.sha1(_canonical_kwargs(payload).encode("utf-8")).hexdigest()[:12]


def _sanitize_component(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(text))


def _cache_root(output_dir: Path) -> Path:
    return output_dir / "_resume"


def _clean_eval_cache_path(output_dir: Path, *, prefix: str, model_name: str, seed: int, batch_size: int) -> Path:
    filename = f"{_sanitize_component(model_name)}__seed{seed}__bs{batch_size}.jsonl"
    return _cache_root(output_dir) / prefix / filename


def _plan_eval_cache_path(
    output_dir: Path,
    *,
    prefix: str,
    model_name: str,
    seed: int,
    batch_size: int,
    attack_name: str,
    attack_kwargs: dict,
    postprocess_name: str | None = None,
    postprocess_kwargs: dict | None = None,
) -> Path:
    payload = {
        "attack": {"name": attack_name, "kwargs": attack_kwargs},
        "postprocess": {"name": postprocess_name, "kwargs": postprocess_kwargs},
    }
    digest = _fingerprint_payload(payload)
    parts = [
        _sanitize_component(model_name),
        f"seed{seed}",
        f"bs{batch_size}",
        _sanitize_component(attack_name),
    ]
    if postprocess_name is not None:
        parts.append(_sanitize_component(postprocess_name))
    filename = "__".join(parts) + f"__{digest}.jsonl"
    return _cache_root(output_dir) / prefix / filename


def _clean_run_key(model_name: str, seed: int) -> tuple[str, int]:
    return model_name, int(seed)


def _attack_run_key(model_name: str, seed: int, attack_name: str, attack_kwargs: dict) -> tuple[str, int, str, str]:
    return model_name, int(seed), attack_name, _canonical_kwargs(attack_kwargs)


def _postprocess_run_key(
    model_name: str,
    seed: int,
    attack_name: str,
    attack_kwargs: dict,
    postprocess_name: str,
    postprocess_kwargs: dict,
) -> tuple[str, int, str, str, str, str]:
    return (
        model_name,
        int(seed),
        attack_name,
        _canonical_kwargs(attack_kwargs),
        postprocess_name,
        _canonical_kwargs(postprocess_kwargs),
    )


def _load_completed_clean_runs(path: Path) -> dict[tuple[str, int], dict]:
    completed: dict[tuple[str, int], dict] = {}
    for row in _load_jsonl(path):
        model_name = row.get("model")
        seed = row.get("seed")
        if model_name is None or seed is None:
            continue
        completed[_clean_run_key(str(model_name), int(seed))] = row
    return completed


def _load_completed_attack_runs(path: Path) -> dict[tuple[str, int, str, str], dict]:
    completed: dict[tuple[str, int, str, str], dict] = {}
    for row in _load_jsonl(path):
        model_name = row.get("model")
        seed = row.get("seed")
        attack_name = row.get("attack")
        attack_kwargs = row.get("attack_kwargs")
        if model_name is None or seed is None or attack_name is None or not isinstance(attack_kwargs, dict):
            continue
        completed[_attack_run_key(str(model_name), int(seed), str(attack_name), attack_kwargs)] = row
    return completed


def _load_completed_postprocess_runs(path: Path) -> dict[tuple[str, int, str, str, str, str], dict]:
    completed: dict[tuple[str, int, str, str, str, str], dict] = {}
    for row in _load_jsonl(path):
        model_name = row.get("model")
        seed = row.get("seed")
        attack_name = row.get("attack")
        attack_kwargs = row.get("attack_kwargs")
        postprocess_name = row.get("postprocess")
        postprocess_kwargs = row.get("postprocess_kwargs")
        if (
            model_name is None
            or seed is None
            or attack_name is None
            or postprocess_name is None
            or not isinstance(attack_kwargs, dict)
            or not isinstance(postprocess_kwargs, dict)
        ):
            continue
        completed[
            _postprocess_run_key(
                str(model_name),
                int(seed),
                str(attack_name),
                attack_kwargs,
                str(postprocess_name),
                postprocess_kwargs,
            )
        ] = row
    return completed


def _flatten_batch_rows(rows: list[dict]) -> tuple[list[int], list[float], list[dict[str, float]]]:
    y_true: list[int] = []
    y_score: list[float] = []
    quality_rows: list[dict[str, float]] = []
    for row in rows:
        y_true.extend(int(value) for value in row.get("labels", []))
        y_score.extend(float(value) for value in row.get("scores", []))
        quality = row.get("quality")
        if isinstance(quality, dict) and quality:
            quality_rows.append({key: float(value) for key, value in quality.items()})
    return y_true, y_score, quality_rows


def _log_resume_state(label: str, cache_path: Path, completed_batches: int, total_batches: int) -> None:
    if completed_batches <= 0:
        return
    log(
        f"{label}: resuming from batch {completed_batches}/{total_batches} "
        f"using cache {cache_path.resolve()}."
    )


def _image_ssim_scores(clean: torch.Tensor, adv: torch.Tensor) -> np.ndarray:
    try:
        from pytorch_msssim import ssim
    except Exception:
        batch_size = clean.shape[0] if clean.ndim >= 1 else 1
        return np.full(batch_size, np.nan, dtype=np.float64)
    if clean.ndim == 4:
        return ssim(clean, adv, data_range=1.0, size_average=False).detach().cpu().numpy().astype(np.float64)
    if clean.ndim == 5:
        scores = []
        for frame_idx in range(clean.shape[2]):
            scores.append(ssim(clean[:, :, frame_idx], adv[:, :, frame_idx], data_range=1.0, size_average=False).detach().cpu().numpy())
        return np.mean(np.stack(scores, axis=0), axis=0).astype(np.float64)
    return np.full(clean.shape[0], np.nan, dtype=np.float64)


def _audio_pesq_scores(clean: torch.Tensor, adv: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
    try:
        from pesq import pesq
    except Exception:
        return np.full(clean.shape[0], np.nan, dtype=np.float64)
    scores = []
    for clean_wave, adv_wave in zip(clean.detach().cpu(), adv.detach().cpu()):
        try:
            scores.append(pesq(sample_rate, clean_wave.numpy(), adv_wave.numpy(), "wb"))
        except Exception:
            scores.append(float("nan"))
    return np.asarray(scores, dtype=np.float64)


def _release_memory(device: str | torch.device | None = None) -> None:
    gc.collect()
    if device is None:
        return
    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class AttackPlan:
    name: str
    kwargs: dict


@dataclass
class QualityGate:
    thresholds: dict[str, float]
    reject_on_nan: bool = True


def evaluate_model(
    adapter,
    records: list[ManifestRecord],
    data_root: str | Path,
    *,
    batch_size: int = 4,
    progress_label: str | None = None,
    batch_cache_path: Path | None = None,
) -> dict:
    total_batches = _num_batches(len(records), batch_size)
    cached_rows = _load_jsonl(batch_cache_path)[:total_batches] if batch_cache_path is not None else []
    y_true, y_score, _ = _flatten_batch_rows(cached_rows)
    completed_batches = min(len(cached_rows), total_batches)
    if batch_cache_path is not None:
        _log_resume_state(progress_label or "[clean]", batch_cache_path, completed_batches, total_batches)
    if completed_batches >= total_batches:
        _release_memory(adapter.device)
        return classification_metrics(y_true, y_score)

    batch_iter = _batches(records[completed_batches * batch_size:], batch_size)
    if progress_label:
        batch_iter = progress_iter(
            batch_iter,
            total=total_batches - completed_batches,
            desc=progress_label,
            unit="batch",
            leave=False,
        )
    with torch.inference_mode():
        for batch_records in batch_iter:
            inputs, labels = adapter.prepare_batch(batch_records, data_root)
            if labels.numel() == 0:
                if batch_cache_path is not None:
                    _append_jsonl(batch_cache_path, {"labels": [], "scores": []})
                continue
            scores = [float(value) for value in adapter.forward_scores(inputs).detach().cpu().tolist()]
            labels_list = [int(value) for value in labels.detach().cpu().tolist()]
            y_score.extend(scores)
            y_true.extend(labels_list)
            if batch_cache_path is not None:
                _append_jsonl(batch_cache_path, {"labels": labels_list, "scores": scores})
            del inputs, labels, scores
            _release_memory(adapter.device)
    _release_memory(adapter.device)
    return classification_metrics(y_true, y_score)


def run_clean_benchmark(
    *,
    assets: AssetLayout,
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    models: list[str] | None = None,
    split: str = "test",
    seeds: list[int] | None = None,
    batch_size: int = 4,
    device: str = "cpu",
) -> dict[str, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_runs_path = output_dir / "clean_runs.jsonl"
    completed_runs = _load_completed_clean_runs(clean_runs_path)
    records = _split_records(load_manifest(manifest_path), split)
    if not records:
        raise ValueError(f"No records found in split '{split}' for manifest {manifest_path}.")
    task = records[0].task
    if models is None:
        models = list_active_models(records[0].task if records else None)
    seeds = seeds or [0, 1, 2]
    dataset = records[0].dataset if records else ""

    log(
        f"[clean] Loaded {len(records)} {split} records for task={task}, dataset={dataset or 'unknown'} "
        f"from {Path(manifest_path).resolve()}."
    )
    log(f"[clean] Writing outputs to {output_dir.resolve()}.")

    aggregate: dict[str, dict] = {}
    for model_name in progress_iter(models, total=len(models), desc="[clean] models", unit="model"):
        log(f"[clean] Starting model '{model_name}' with batch_size={batch_size} on device={device}.")
        per_seed = []
        for seed in progress_iter(seeds, total=len(seeds), desc=f"[clean] {model_name} seeds", unit="seed", leave=False):
            existing = completed_runs.get(_clean_run_key(model_name, seed))
            if existing is not None:
                metrics = dict(existing)
                metrics.pop("model", None)
                per_seed.append(metrics)
                log(f"[clean] Reusing completed run for model='{model_name}', seed={seed} from {clean_runs_path.resolve()}.")
                continue

            set_seed(seed)
            log(f"[clean] Building adapter for model='{model_name}', seed={seed}.")
            adapter = build_adapter(model_name, assets, device=device)
            metrics = evaluate_model(
                adapter,
                records,
                data_root,
                batch_size=batch_size,
                progress_label=f"[clean] {model_name} seed={seed}",
                batch_cache_path=_clean_eval_cache_path(
                    output_dir,
                    prefix="clean",
                    model_name=model_name,
                    seed=seed,
                    batch_size=batch_size,
                ),
            )
            metrics["seed"] = seed
            metrics["dataset"] = dataset
            per_seed.append(metrics)
            payload = {"model": model_name, **metrics}
            _append_jsonl(clean_runs_path, payload)
            completed_runs[_clean_run_key(model_name, seed)] = payload
            log(
                f"[clean] Finished model='{model_name}', seed={seed}: "
                f"auc={metrics['auc']:.4f}, eer={metrics['eer']:.4f}, f1={metrics['f1']:.4f}."
            )
            del adapter
            _release_memory(device)
        auc_stats = confidence_interval95([run["auc"] for run in per_seed])
        eer_stats = confidence_interval95([run["eer"] for run in per_seed])
        f1_stats = confidence_interval95([run["f1"] for run in per_seed])
        aggregate[model_name] = {
            "task": task,
            "dataset": dataset,
            "per_seed": per_seed,
            "auc": auc_stats,
            "eer": eer_stats,
            "f1": f1_stats,
        }
        log(
            f"[clean] Summary for '{model_name}': "
            f"auc={auc_stats['mean']:.4f}+/-{auc_stats['ci95']:.4f}, "
            f"eer={eer_stats['mean']:.4f}+/-{eer_stats['ci95']:.4f}, "
            f"f1={f1_stats['mean']:.4f}+/-{f1_stats['ci95']:.4f}."
        )
        _write_clean_summary(output_dir, aggregate)
    _write_clean_summary(output_dir, aggregate)
    log(f"[clean] Completed benchmark. Summary written to {(output_dir / 'clean_summary.json').resolve()}.")
    return aggregate


def _run_attack(adapter, attack_name: str, inputs, labels, kwargs: dict):
    if attack_name == "asymmetric_audio":
        return multimodal_asymmetric_pgd(adapter, inputs, labels, mode="audio_only", **kwargs)
    if attack_name == "asymmetric_video":
        return multimodal_asymmetric_pgd(adapter, inputs, labels, mode="video_only", **kwargs)
    if attack_name == "asymmetric_both":
        return multimodal_asymmetric_pgd(adapter, inputs, labels, mode="both", **kwargs)
    if attack_name not in ATTACKS:
        raise KeyError(f"Unknown attack: {attack_name}")
    return ATTACKS[attack_name](adapter, inputs, labels, **kwargs)


def _batch_size(inputs) -> int:
    if isinstance(inputs, dict):
        return next(iter(inputs.values())).shape[0]
    return inputs.shape[0]


def _quality_scores(task: str, clean_inputs, adv_inputs) -> dict[str, np.ndarray]:
    if task == "audio":
        return {"pesq": _audio_pesq_scores(clean_inputs, adv_inputs)}
    if task in {"image", "video"}:
        return {"ssim": _image_ssim_scores(clean_inputs, adv_inputs)}
    if task == "multimodal":
        quality = {}
        if isinstance(clean_inputs, dict):
            quality["video_ssim"] = _image_ssim_scores(clean_inputs["video"], adv_inputs["video"])
            quality["audio_pesq"] = _audio_pesq_scores(clean_inputs["audio"], adv_inputs["audio"])
        return quality
    return {}


def _slice_inputs(inputs, mask: torch.Tensor):
    if isinstance(inputs, dict):
        return {key: value[mask] for key, value in inputs.items()}
    return inputs[mask]


def _merge_inputs(clean_inputs, adv_inputs, accepted_mask: torch.Tensor):
    if isinstance(clean_inputs, dict):
        merged = {key: value.detach().clone() for key, value in clean_inputs.items()}
        for key in merged:
            merged[key][accepted_mask] = adv_inputs[key][accepted_mask]
        return merged
    merged = clean_inputs.detach().clone()
    merged[accepted_mask] = adv_inputs[accepted_mask]
    return merged


def _quality_summary(task: str, clean_inputs, adv_inputs, quality_gate: QualityGate | None = None) -> tuple[dict[str, float], torch.Tensor]:
    arrays = _quality_scores(task, clean_inputs, adv_inputs)
    batch_size = _batch_size(clean_inputs)
    accepted = np.ones(batch_size, dtype=bool)
    summary: dict[str, float] = {}
    for metric_name, values in arrays.items():
        summary[metric_name] = float(np.nanmean(values)) if values.size else float("nan")
        summary[f"{metric_name}_min"] = float(np.nanmin(values)) if np.any(np.isfinite(values)) else float("nan")
        if quality_gate and metric_name in quality_gate.thresholds:
            if values.size and not np.any(np.isfinite(values)):
                raise RuntimeError(
                    f"Quality metric '{metric_name}' is unavailable for task='{task}'. "
                    "Install the required dependency or disable the corresponding quality gate."
                )
            threshold = quality_gate.thresholds[metric_name]
            if quality_gate.reject_on_nan:
                metric_mask = np.isfinite(values) & (values >= threshold)
            else:
                metric_mask = np.isnan(values) | (values >= threshold)
            accepted &= metric_mask
    summary["accepted_fraction"] = float(np.mean(accepted)) if batch_size else float("nan")
    summary["accepted_count"] = float(np.sum(accepted))
    summary["rejected_count"] = float(batch_size - np.sum(accepted))
    device = next(iter(clean_inputs.values())).device if isinstance(clean_inputs, dict) else clean_inputs.device
    return summary, torch.as_tensor(accepted, dtype=torch.bool, device=device)


def _aggregate_quality_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.nanmean([row.get(key, float("nan")) for row in rows])) for key in keys}


def _write_clean_summary(output_dir: Path, aggregate: dict[str, dict]) -> None:
    _write_json(output_dir / "clean_summary.json", aggregate)
    summary_rows = []
    for model_name, entry in aggregate.items():
        auc_stats = entry.get("auc", {})
        eer_stats = entry.get("eer", {})
        f1_stats = entry.get("f1", {})
        summary_rows.append(
            {
                "model": model_name,
                "task": entry.get("task", ""),
                "dataset": entry.get("dataset", ""),
                "auc_mean": auc_stats.get("mean", float("nan")),
                "auc_ci95": auc_stats.get("ci95", float("nan")),
                "eer_mean": eer_stats.get("mean", float("nan")),
                "eer_ci95": eer_stats.get("ci95", float("nan")),
                "f1_mean": f1_stats.get("mean", float("nan")),
                "f1_ci95": f1_stats.get("ci95", float("nan")),
            }
        )
    with (output_dir / "clean_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["model"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _write_result_summary(output_dir: Path, filename: str, payload: list[dict]) -> None:
    _write_json(output_dir / filename, payload)


def _reusable_clean_metrics(payloads: Iterable[dict], *, model_name: str, seed: int) -> dict | None:
    for payload in payloads:
        if payload.get("model") == model_name and int(payload.get("seed", -1)) == seed:
            clean_metrics = payload.get("clean_metrics")
            if isinstance(clean_metrics, dict):
                return clean_metrics
    return None


def _validate_optional_dependencies(*, attack_plans: list[AttackPlan], quality_gate: QualityGate | None = None) -> None:
    requirements: dict[str, str] = {}
    for plan in attack_plans:
        if plan.name == "cw":
            requirements.setdefault("torchattacks", "attack 'cw'")
        elif plan.name == "autoattack":
            requirements.setdefault("autoattack", "attack 'autoattack'")
    if quality_gate is not None:
        for metric_name in quality_gate.thresholds:
            if "ssim" in metric_name:
                requirements.setdefault("pytorch_msssim", f"quality metric '{metric_name}'")
            if "pesq" in metric_name:
                requirements.setdefault("pesq", f"quality metric '{metric_name}'")

    missing = []
    for module_name, reason in requirements.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(f"{module_name} ({reason})")
    if missing:
        raise RuntimeError(
            "Missing optional dependencies required for this run: "
            + ", ".join(missing)
            + ". Install them before launching the benchmark."
        )


def _evaluate_attack_plan(
    adapter,
    records: list[ManifestRecord],
    data_root: str | Path,
    *,
    task: str,
    attack_name: str,
    attack_kwargs: dict,
    batch_size: int,
    device: str,
    quality_gate: QualityGate | None,
    progress_label: str,
    batch_cache_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    total_batches = _num_batches(len(records), batch_size)
    cached_rows = _load_jsonl(batch_cache_path)[:total_batches] if batch_cache_path is not None else []
    y_true, y_score, quality_rows = _flatten_batch_rows(cached_rows)
    completed_batches = min(len(cached_rows), total_batches)
    if batch_cache_path is not None:
        _log_resume_state(progress_label, batch_cache_path, completed_batches, total_batches)
    if completed_batches < total_batches:
        batch_iter = progress_iter(
            _batches(records[completed_batches * batch_size:], batch_size),
            total=total_batches - completed_batches,
            desc=progress_label,
            unit="batch",
            leave=False,
        )
        for batch_records in batch_iter:
            inputs, labels = adapter.prepare_batch(batch_records, data_root)
            if labels.numel() == 0:
                if batch_cache_path is not None:
                    _append_jsonl(batch_cache_path, {"labels": [], "scores": []})
                continue
            clean_inputs = {key: value.detach().clone() for key, value in inputs.items()} if isinstance(inputs, dict) else inputs.detach().clone()
            adv_inputs = _run_attack(adapter, attack_name, inputs, labels, attack_kwargs)
            batch_quality, accepted_mask = _quality_summary(task, clean_inputs, adv_inputs, quality_gate)
            gated_inputs = _merge_inputs(clean_inputs, adv_inputs, accepted_mask)
            scores = [float(value) for value in adapter.forward_scores(gated_inputs).detach().cpu().tolist()]
            labels_list = [int(value) for value in labels.detach().cpu().tolist()]
            y_score.extend(scores)
            y_true.extend(labels_list)
            quality_rows.append(batch_quality)
            if batch_cache_path is not None:
                _append_jsonl(batch_cache_path, {"labels": labels_list, "scores": scores, "quality": batch_quality})
            del inputs, labels, clean_inputs, adv_inputs, accepted_mask, gated_inputs, scores
            _release_memory(device)
    return classification_metrics(y_true, y_score), (_aggregate_quality_rows(quality_rows) if quality_rows else {})


def _evaluate_postprocess_plan(
    adapter,
    records: list[ManifestRecord],
    data_root: str | Path,
    *,
    task: str,
    attack_plan: AttackPlan,
    postprocess_plan: PostprocessPlan,
    batch_size: int,
    device: str,
    quality_gate: QualityGate | None,
    progress_label: str,
    batch_cache_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    total_batches = _num_batches(len(records), batch_size)
    cached_rows = _load_jsonl(batch_cache_path)[:total_batches] if batch_cache_path is not None else []
    y_true, y_score, quality_rows = _flatten_batch_rows(cached_rows)
    completed_batches = min(len(cached_rows), total_batches)
    if batch_cache_path is not None:
        _log_resume_state(progress_label, batch_cache_path, completed_batches, total_batches)
    if completed_batches < total_batches:
        batch_iter = progress_iter(
            _batches(records[completed_batches * batch_size:], batch_size),
            total=total_batches - completed_batches,
            desc=progress_label,
            unit="batch",
            leave=False,
        )
        for batch_records in batch_iter:
            inputs, labels = adapter.prepare_batch(batch_records, data_root)
            if labels.numel() == 0:
                if batch_cache_path is not None:
                    _append_jsonl(batch_cache_path, {"labels": [], "scores": []})
                continue
            clean_inputs = {key: value.detach().clone() for key, value in inputs.items()} if isinstance(inputs, dict) else inputs.detach().clone()
            adv_inputs = _run_attack(adapter, attack_plan.name, inputs, labels, attack_plan.kwargs)
            batch_quality, accepted_mask = _quality_summary(task, clean_inputs, adv_inputs, quality_gate)
            gated_inputs = _merge_inputs(clean_inputs, adv_inputs, accepted_mask)
            postprocessed_inputs = apply_postprocessor(task, gated_inputs, postprocess_plan)
            scores = [float(value) for value in adapter.forward_scores(postprocessed_inputs).detach().cpu().tolist()]
            labels_list = [int(value) for value in labels.detach().cpu().tolist()]
            y_score.extend(scores)
            y_true.extend(labels_list)
            quality_rows.append(batch_quality)
            if batch_cache_path is not None:
                _append_jsonl(batch_cache_path, {"labels": labels_list, "scores": scores, "quality": batch_quality})
            del inputs, labels, clean_inputs, adv_inputs, accepted_mask, gated_inputs, postprocessed_inputs, scores
            _release_memory(device)
    return classification_metrics(y_true, y_score), (_aggregate_quality_rows(quality_rows) if quality_rows else {})


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _empty_clean_metrics(*, seed: int, dataset: str) -> dict[str, float | int | str]:
    return {
        "auc": float("nan"),
        "ap": float("nan"),
        "eer": float("nan"),
        "accuracy": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "count": 0.0,
        "seed": seed,
        "dataset": dataset,
    }


def _load_clean_summary(path: str | Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_clean_metrics(
    clean_summary: dict[str, dict] | None,
    *,
    model_name: str,
    seed: int,
    dataset: str,
) -> dict:
    default = _empty_clean_metrics(seed=seed, dataset=dataset)
    if clean_summary is None:
        return default
    entry = clean_summary.get(model_name)
    if entry is None:
        return default
    for per_seed in entry.get("per_seed", []):
        if int(per_seed.get("seed", -1)) == seed:
            resolved = default.copy()
            resolved.update(per_seed)
            return resolved
    resolved = default.copy()
    resolved["auc"] = float(entry.get("auc", {}).get("mean", float("nan")))
    resolved["eer"] = float(entry.get("eer", {}).get("mean", float("nan")))
    resolved["f1"] = float(entry.get("f1", {}).get("mean", float("nan")))
    return resolved


def run_attack_benchmark(
    *,
    assets: AssetLayout,
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    attack_plans: list[AttackPlan],
    models: list[str] | None = None,
    split: str = "test",
    seeds: list[int] | None = None,
    batch_size: int = 2,
    device: str = "cpu",
    quality_gate: QualityGate | None = None,
    skip_clean_baseline: bool = False,
    clean_summary_path: str | Path | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_runs_path = output_dir / "attack_runs.jsonl"
    completed_runs = _load_completed_attack_runs(attack_runs_path)
    records = _split_records(load_manifest(manifest_path), split)
    if not records:
        raise ValueError(f"No records found in split '{split}' for manifest {manifest_path}.")
    task = records[0].task if records else "unknown"
    dataset = records[0].dataset if records else ""
    if models is None:
        models = list_active_models(task)
    seeds = seeds or [0, 1, 2]
    clean_summary = _load_clean_summary(clean_summary_path)

    log(
        f"[attack] Loaded {len(records)} {split} records for task={task}, dataset={dataset or 'unknown'} "
        f"from {Path(manifest_path).resolve()}."
    )
    log(
        f"[attack] Running {len(models)} model(s), {len(seeds)} seed(s), {len(attack_plans)} attack plan(s) "
        f"with batch_size={batch_size} on device={device}. "
        f"skip_clean_baseline={skip_clean_baseline}."
    )
    _validate_optional_dependencies(attack_plans=attack_plans, quality_gate=quality_gate)

    results: list[dict] = []
    for model_name in progress_iter(models, total=len(models), desc="[attack] models", unit="model"):
        for seed in progress_iter(seeds, total=len(seeds), desc=f"[attack] {model_name} seeds", unit="seed", leave=False):
            existing_for_seed = [
                payload
                for payload in completed_runs.values()
                if payload.get("model") == model_name and int(payload.get("seed", -1)) == seed
            ]
            pending_plans = [
                plan
                for plan in attack_plans
                if _attack_run_key(model_name, seed, plan.name, plan.kwargs) not in completed_runs
            ]
            if not pending_plans:
                log(f"[attack] All plans already completed for model='{model_name}', seed={seed}; reusing saved results.")
                for plan in attack_plans:
                    results.append(completed_runs[_attack_run_key(model_name, seed, plan.name, plan.kwargs)])
                continue

            set_seed(seed)
            log(f"[attack] Building adapter for model='{model_name}', seed={seed}.")
            adapter = build_adapter(model_name, assets, device=device)
            if skip_clean_baseline:
                clean_metrics = _resolve_clean_metrics(
                    clean_summary,
                    model_name=model_name,
                    seed=seed,
                    dataset=dataset,
                )
                if clean_summary_path:
                    log(
                        f"[attack] Reusing clean reference for model='{model_name}', seed={seed}: "
                        f"auc={clean_metrics['auc']:.4f}, eer={clean_metrics['eer']:.4f}, f1={clean_metrics['f1']:.4f}."
                    )
                else:
                    log(
                        f"[attack] Skipping clean baseline for model='{model_name}', seed={seed} without a clean summary. "
                        "clean metrics and RAR will remain NaN until joined with external clean results."
                    )
            else:
                clean_metrics = _reusable_clean_metrics(existing_for_seed, model_name=model_name, seed=seed)
                if clean_metrics is None:
                    clean_metrics = evaluate_model(
                        adapter,
                        records,
                        data_root,
                        batch_size=batch_size,
                        progress_label=f"[attack] clean baseline {model_name} seed={seed}",
                        batch_cache_path=_clean_eval_cache_path(
                            output_dir,
                            prefix="attack_clean",
                            model_name=model_name,
                            seed=seed,
                            batch_size=batch_size,
                        ),
                    )
                log(
                    f"[attack] Clean baseline for model='{model_name}', seed={seed}: "
                    f"auc={clean_metrics['auc']:.4f}, eer={clean_metrics['eer']:.4f}, f1={clean_metrics['f1']:.4f}."
                )
            for plan in progress_iter(
                attack_plans,
                total=len(attack_plans),
                desc=f"[attack] {model_name} seed={seed} plans",
                unit="plan",
                leave=False,
            ):
                existing = completed_runs.get(_attack_run_key(model_name, seed, plan.name, plan.kwargs))
                if existing is not None:
                    log(
                        f"[attack] Reusing completed plan='{plan.name}' for model='{model_name}', "
                        f"seed={seed}, kwargs={plan.kwargs}."
                    )
                    results.append(existing)
                    continue
                log(f"[attack] Starting plan='{plan.name}' for model='{model_name}', seed={seed}, kwargs={plan.kwargs}.")
                attack_metrics, quality_summary = _evaluate_attack_plan(
                    adapter,
                    records,
                    data_root,
                    task=task,
                    attack_name=plan.name,
                    attack_kwargs=plan.kwargs,
                    batch_size=batch_size,
                    device=device,
                    quality_gate=quality_gate,
                    progress_label=f"[attack] {plan.name} batches",
                    batch_cache_path=_plan_eval_cache_path(
                        output_dir,
                        prefix="attack",
                        model_name=model_name,
                        seed=seed,
                        batch_size=batch_size,
                        attack_name=plan.name,
                        attack_kwargs=plan.kwargs,
                    ),
                )
                payload = {
                    "model": model_name,
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "attack": plan.name,
                    "attack_kwargs": plan.kwargs,
                    "clean_auc": clean_metrics["auc"],
                    "attack_auc": attack_metrics["auc"],
                    "rar": _safe_ratio(attack_metrics["auc"], clean_metrics["auc"]),
                    "clean_metrics": clean_metrics,
                    "attack_metrics": attack_metrics,
                }
                if quality_gate:
                    payload["quality_gate"] = quality_gate.thresholds
                if quality_summary:
                    payload["quality"] = quality_summary
                results.append(payload)
                _append_jsonl(attack_runs_path, payload)
                completed_runs[_attack_run_key(model_name, seed, plan.name, plan.kwargs)] = payload
                _write_result_summary(output_dir, "attack_summary.json", results)
                log(
                    f"[attack] Finished plan='{plan.name}' for model='{model_name}', seed={seed}: "
                    f"attack_auc={attack_metrics['auc']:.4f}, rar={payload['rar']:.4f}."
                )
            del adapter
            _release_memory(device)
    _write_result_summary(output_dir, "attack_summary.json", results)
    log(f"[attack] Completed benchmark. Summary written to {(output_dir / 'attack_summary.json').resolve()}.")
    return results


def run_postprocess_benchmark(
    *,
    assets: AssetLayout,
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    attack_plans: list[AttackPlan],
    postprocess_plans: list[PostprocessPlan],
    models: list[str] | None = None,
    split: str = "test",
    seeds: list[int] | None = None,
    batch_size: int = 2,
    device: str = "cpu",
    quality_gate: QualityGate | None = None,
    skip_clean_baseline: bool = False,
    clean_summary_path: str | Path | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postprocess_runs_path = output_dir / "postprocess_runs.jsonl"
    completed_runs = _load_completed_postprocess_runs(postprocess_runs_path)
    records = _split_records(load_manifest(manifest_path), split)
    if not records:
        raise ValueError(f"No records found in split '{split}' for manifest {manifest_path}.")
    task = records[0].task if records else "unknown"
    dataset = records[0].dataset if records else ""
    if models is None:
        models = list_active_models(task)
    seeds = seeds or [0, 1, 2]
    clean_summary = _load_clean_summary(clean_summary_path)

    if task not in POSTPROCESSORS:
        raise ValueError(f"No postprocessors registered for task '{task}'.")

    log(
        f"[postprocess] Loaded {len(records)} {split} records for task={task}, dataset={dataset or 'unknown'} "
        f"from {Path(manifest_path).resolve()}."
    )
    log(
        f"[postprocess] Running {len(models)} model(s), {len(seeds)} seed(s), "
        f"{len(attack_plans)} attack plan(s), and {len(postprocess_plans)} postprocess plan(s). "
        f"skip_clean_baseline={skip_clean_baseline}."
    )
    _validate_optional_dependencies(attack_plans=attack_plans, quality_gate=quality_gate)

    results: list[dict] = []
    for model_name in progress_iter(models, total=len(models), desc="[postprocess] models", unit="model"):
        for seed in progress_iter(seeds, total=len(seeds), desc=f"[postprocess] {model_name} seeds", unit="seed", leave=False):
            existing_for_seed = [
                payload
                for payload in completed_runs.values()
                if payload.get("model") == model_name and int(payload.get("seed", -1)) == seed
            ]
            pending_pairs = [
                (attack_plan, postprocess_plan)
                for attack_plan in attack_plans
                for postprocess_plan in postprocess_plans
                if _postprocess_run_key(
                    model_name,
                    seed,
                    attack_plan.name,
                    attack_plan.kwargs,
                    postprocess_plan.name,
                    postprocess_plan.kwargs,
                )
                not in completed_runs
            ]
            if not pending_pairs:
                log(f"[postprocess] All attack/postprocess pairs already completed for model='{model_name}', seed={seed}.")
                for attack_plan in attack_plans:
                    for postprocess_plan in postprocess_plans:
                        results.append(
                            completed_runs[
                                _postprocess_run_key(
                                    model_name,
                                    seed,
                                    attack_plan.name,
                                    attack_plan.kwargs,
                                    postprocess_plan.name,
                                    postprocess_plan.kwargs,
                                )
                            ]
                        )
                continue

            set_seed(seed)
            log(f"[postprocess] Building adapter for model='{model_name}', seed={seed}.")
            adapter = build_adapter(model_name, assets, device=device)
            if skip_clean_baseline:
                clean_metrics = _resolve_clean_metrics(
                    clean_summary,
                    model_name=model_name,
                    seed=seed,
                    dataset=dataset,
                )
                if clean_summary_path:
                    log(
                        f"[postprocess] Reusing clean reference for model='{model_name}', seed={seed}: "
                        f"auc={clean_metrics['auc']:.4f}, eer={clean_metrics['eer']:.4f}, f1={clean_metrics['f1']:.4f}."
                    )
                else:
                    log(
                        f"[postprocess] Skipping clean baseline for model='{model_name}', seed={seed} without a clean summary. "
                        "clean metrics and RAR will remain NaN until joined with external clean results."
                    )
            else:
                clean_metrics = _reusable_clean_metrics(existing_for_seed, model_name=model_name, seed=seed)
                if clean_metrics is None:
                    clean_metrics = evaluate_model(
                        adapter,
                        records,
                        data_root,
                        batch_size=batch_size,
                        progress_label=f"[postprocess] clean baseline {model_name} seed={seed}",
                        batch_cache_path=_clean_eval_cache_path(
                            output_dir,
                            prefix="postprocess_clean",
                            model_name=model_name,
                            seed=seed,
                            batch_size=batch_size,
                        ),
                    )
                log(
                    f"[postprocess] Clean baseline for model='{model_name}', seed={seed}: "
                    f"auc={clean_metrics['auc']:.4f}, eer={clean_metrics['eer']:.4f}, f1={clean_metrics['f1']:.4f}."
                )
            for attack_plan in progress_iter(
                attack_plans,
                total=len(attack_plans),
                desc=f"[postprocess] {model_name} seed={seed} attacks",
                unit="attack",
                leave=False,
            ):
                for postprocess_plan in progress_iter(
                    postprocess_plans,
                    total=len(postprocess_plans),
                    desc=f"[postprocess] {attack_plan.name} transforms",
                    unit="transform",
                    leave=False,
                ):
                    existing = completed_runs.get(
                        _postprocess_run_key(
                            model_name,
                            seed,
                            attack_plan.name,
                            attack_plan.kwargs,
                            postprocess_plan.name,
                            postprocess_plan.kwargs,
                        )
                    )
                    if existing is not None:
                        log(
                            f"[postprocess] Reusing completed attack='{attack_plan.name}', "
                            f"postprocess='{postprocess_plan.name}' for model='{model_name}', seed={seed}."
                        )
                        results.append(existing)
                        continue
                    log(
                        f"[postprocess] Starting attack='{attack_plan.name}', postprocess='{postprocess_plan.name}' "
                        f"for model='{model_name}', seed={seed}."
                    )
                    post_metrics, quality_summary = _evaluate_postprocess_plan(
                        adapter,
                        records,
                        data_root,
                        task=task,
                        attack_plan=attack_plan,
                        postprocess_plan=postprocess_plan,
                        batch_size=batch_size,
                        device=device,
                        quality_gate=quality_gate,
                        progress_label=f"[postprocess] {attack_plan.name}/{postprocess_plan.name} batches",
                        batch_cache_path=_plan_eval_cache_path(
                            output_dir,
                            prefix="postprocess",
                            model_name=model_name,
                            seed=seed,
                            batch_size=batch_size,
                            attack_name=attack_plan.name,
                            attack_kwargs=attack_plan.kwargs,
                            postprocess_name=postprocess_plan.name,
                            postprocess_kwargs=postprocess_plan.kwargs,
                        ),
                    )
                    payload = {
                        "model": model_name,
                        "task": task,
                        "dataset": dataset,
                        "seed": seed,
                        "attack": attack_plan.name,
                        "attack_kwargs": attack_plan.kwargs,
                        "postprocess": postprocess_plan.name,
                        "postprocess_kwargs": postprocess_plan.kwargs,
                        "clean_auc": clean_metrics["auc"],
                        "postprocess_auc": post_metrics["auc"],
                        "rar": _safe_ratio(post_metrics["auc"], clean_metrics["auc"]),
                        "clean_metrics": clean_metrics,
                        "postprocess_metrics": post_metrics,
                    }
                    if quality_gate:
                        payload["quality_gate"] = quality_gate.thresholds
                    if quality_summary:
                        payload["quality"] = quality_summary
                    results.append(payload)
                    _append_jsonl(postprocess_runs_path, payload)
                    completed_runs[
                        _postprocess_run_key(
                            model_name,
                            seed,
                            attack_plan.name,
                            attack_plan.kwargs,
                            postprocess_plan.name,
                            postprocess_plan.kwargs,
                        )
                    ] = payload
                    _write_result_summary(output_dir, "postprocess_summary.json", results)
                    log(
                        f"[postprocess] Finished attack='{attack_plan.name}', postprocess='{postprocess_plan.name}' "
                        f"for model='{model_name}', seed={seed}: "
                        f"postprocess_auc={post_metrics['auc']:.4f}, rar={payload['rar']:.4f}."
                    )
            del adapter
            _release_memory(device)
    _write_result_summary(output_dir, "postprocess_summary.json", results)
    log(f"[postprocess] Completed benchmark. Summary written to {(output_dir / 'postprocess_summary.json').resolve()}.")
    return results
