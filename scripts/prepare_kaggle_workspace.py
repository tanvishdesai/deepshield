from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.manifests import (
    generate_asvspoof2019_la_manifest,
    generate_asvspoof2021_df_manifest,
    generate_celebdf_image_manifest,
    generate_fakeavceleb_manifest,
    generate_ffpp_manifests,
    generate_lavdf_manifests,
)
from deepshield.progress import log, log_config, progress_iter


def _resolve_metadata_path(lavdf_root: Path) -> Path:
    for candidate in ("metadata.min.json", "metadata.json", "lavdf-metadata.json"):
        path = lavdf_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find metadata.min.json, metadata.json, or lavdf-metadata.json under {lavdf_root}.")


def _register_manifest(
    catalog: dict[str, dict],
    *,
    key: str,
    task: str,
    dataset: str,
    manifest_path: Path,
    data_root: Path,
    role: str,
) -> None:
    catalog[key] = {
        "task": task,
        "dataset": dataset,
        "manifest_path": str(manifest_path.resolve()),
        "data_root": str(data_root.resolve()),
        "role": role,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle DeepShield workspace for the intended multi-dataset benchmark.")
    parser.add_argument("--assets-root", required=True, help="Kaggle input path containing the packaged model_repos/ and artifacts/checkpoints/.")
    parser.add_argument("--output-dir", required=True, help="Writable working directory for generated manifests and validation reports.")
    parser.add_argument("--ffpp-root", help="Kaggle input path for FaceForensics++ c23 videos.")
    parser.add_argument("--ffpp-splits-root", help="Optional override for the FaceForensics++ official split JSON directory.")
    parser.add_argument("--celebdf-root", help="Kaggle input path for Celeb-DF v2.")
    parser.add_argument("--asvspoof2019-la-root", help="Kaggle input path for ASVspoof 2019 LA.")
    parser.add_argument("--asvspoof2021-df-root", help="Kaggle input path for ASVspoof 2021 DF audio files.")
    parser.add_argument("--asvspoof2021-keys-root", help="Optional Kaggle input path for the ASVspoof 2021 keys package.")
    parser.add_argument("--fakeavceleb-root", help="Kaggle input path for FakeAVCeleb.")
    parser.add_argument("--lavdf-root", help="Kaggle input path for the mounted LAV-DF dataset root.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model-validation", action="store_true")
    parser.add_argument("--image-frames-per-video", type=int, default=10)
    parser.add_argument("--audio-window-sec", type=float, default=4.0)
    parser.add_argument("--audio-stride-sec", type=float, default=2.0)
    parser.add_argument("--clip-window-sec", type=float, default=30.0)
    parser.add_argument("--clip-stride-sec", type=float, default=15.0)
    args = parser.parse_args()

    log("Starting Kaggle workspace preparation.")
    log_config(
        "Workspace configuration",
        {
            "assets_root": args.assets_root,
            "output_dir": args.output_dir,
            "ffpp_root": args.ffpp_root or "not set",
            "ffpp_splits_root": args.ffpp_splits_root or "auto from assets",
            "celebdf_root": args.celebdf_root or "not set",
            "asvspoof2019_la_root": args.asvspoof2019_la_root or "not set",
            "asvspoof2021_df_root": args.asvspoof2021_df_root or "not set",
            "asvspoof2021_keys_root": args.asvspoof2021_keys_root or "not set",
            "fakeavceleb_root": args.fakeavceleb_root or "not set",
            "lavdf_root": args.lavdf_root or "not set",
            "device": args.device,
            "skip_model_validation": args.skip_model_validation,
            "image_frames_per_video": args.image_frames_per_video,
            "audio_window_sec": args.audio_window_sec,
            "audio_stride_sec": args.audio_stride_sec,
            "clip_window_sec": args.clip_window_sec,
            "clip_stride_sec": args.clip_stride_sec,
        },
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    log("Resolving asset layout.")
    assets = resolve_asset_layout(args.assets_root)
    ffpp_splits_root = Path(args.ffpp_splits_root).resolve() if args.ffpp_splits_root else assets.model_repos / "FaceForensics" / "dataset" / "splits"
    log(f"Resolved assets under {assets.root}.")
    log(f"Using FFPP splits root {ffpp_splits_root}.")

    summary: dict[str, object] = {
        "assets_root": str(assets.root),
        "model_repos": str(assets.model_repos),
        "checkpoints": str(assets.checkpoints),
        "dataset_roots": {},
    }
    catalog: dict[str, dict] = {}

    if args.ffpp_root:
        ffpp_root = Path(args.ffpp_root).resolve()
        log(f"Generating FFPP manifests from {ffpp_root}.")
        ffpp_manifests = generate_ffpp_manifests(
            dataset_root=ffpp_root,
            output_dir=manifests_dir,
            splits_root=ffpp_splits_root,
            image_frames_per_video=args.image_frames_per_video,
            clip_window_sec=args.clip_window_sec,
            clip_stride_sec=args.clip_stride_sec,
            image_manifest_name="image_primary_ffpp_c23.csv",
            video_manifest_name="video_primary_ffpp_c23.csv",
        )
        summary["dataset_roots"]["ffpp_root"] = str(ffpp_root)
        summary["dataset_roots"]["ffpp_splits_root"] = str(ffpp_splits_root)
        _register_manifest(catalog, key="image_primary_ffpp_c23", task="image", dataset="ffpp_c23", manifest_path=ffpp_manifests["image"], data_root=ffpp_root, role="primary")
        _register_manifest(catalog, key="video_primary_ffpp_c23", task="video", dataset="ffpp_c23", manifest_path=ffpp_manifests["video"], data_root=ffpp_root, role="primary")
        log("Registered FFPP image and video manifests.")

    if args.celebdf_root:
        celebdf_root = Path(args.celebdf_root).resolve()
        log(f"Generating Celeb-DF manifest from {celebdf_root}.")
        celebdf_manifest = generate_celebdf_image_manifest(
            dataset_root=celebdf_root,
            output_dir=manifests_dir,
            image_frames_per_video=args.image_frames_per_video,
            manifest_name="image_cross_celebdf_v2.csv",
        )
        summary["dataset_roots"]["celebdf_root"] = str(celebdf_root)
        _register_manifest(catalog, key="image_cross_celebdf_v2", task="image", dataset="celebdf_v2", manifest_path=celebdf_manifest, data_root=celebdf_root, role="cross_dataset")
        log("Registered Celeb-DF image manifest.")

    if args.asvspoof2019_la_root:
        asvspoof2019_root = Path(args.asvspoof2019_la_root).resolve()
        log(f"Generating ASVspoof 2019 LA manifest from {asvspoof2019_root}.")
        asvspoof2019_manifest = generate_asvspoof2019_la_manifest(
            dataset_root=asvspoof2019_root,
            output_dir=manifests_dir,
            manifest_name="audio_primary_asvspoof2019_la.csv",
        )
        summary["dataset_roots"]["asvspoof2019_la_root"] = str(asvspoof2019_root)
        _register_manifest(catalog, key="audio_primary_asvspoof2019_la", task="audio", dataset="asvspoof2019_la", manifest_path=asvspoof2019_manifest, data_root=asvspoof2019_root, role="primary")
        log("Registered ASVspoof 2019 LA audio manifest.")

    if args.asvspoof2021_df_root:
        asvspoof2021_root = Path(args.asvspoof2021_df_root).resolve()
        asvspoof2021_keys_root = Path(args.asvspoof2021_keys_root).resolve() if args.asvspoof2021_keys_root else None
        log(f"Generating ASVspoof 2021 DF manifest from {asvspoof2021_root}.")
        asvspoof2021_manifest = generate_asvspoof2021_df_manifest(
            dataset_root=asvspoof2021_root,
            output_dir=manifests_dir,
            keys_root=asvspoof2021_keys_root,
            manifest_name="audio_cross_asvspoof2021_df.csv",
        )
        summary["dataset_roots"]["asvspoof2021_df_root"] = str(asvspoof2021_root)
        if asvspoof2021_keys_root:
            summary["dataset_roots"]["asvspoof2021_keys_root"] = str(asvspoof2021_keys_root)
        _register_manifest(catalog, key="audio_cross_asvspoof2021_df", task="audio", dataset="asvspoof2021_df", manifest_path=asvspoof2021_manifest, data_root=asvspoof2021_root, role="cross_dataset")
        log("Registered ASVspoof 2021 DF audio manifest.")

    if args.fakeavceleb_root:
        fakeavceleb_root = Path(args.fakeavceleb_root).resolve()
        log(f"Generating FakeAVCeleb manifest from {fakeavceleb_root}.")
        fakeavceleb_manifest = generate_fakeavceleb_manifest(
            dataset_root=fakeavceleb_root,
            output_dir=manifests_dir,
            clip_window_sec=args.clip_window_sec,
            clip_stride_sec=args.clip_stride_sec,
            manifest_name="multimodal_primary_fakeavceleb.csv",
        )
        summary["dataset_roots"]["fakeavceleb_root"] = str(fakeavceleb_root)
        _register_manifest(catalog, key="multimodal_primary_fakeavceleb", task="multimodal", dataset="fakeavceleb", manifest_path=fakeavceleb_manifest, data_root=fakeavceleb_root, role="primary")
        log("Registered FakeAVCeleb multimodal manifest.")

    if args.lavdf_root:
        lavdf_root = Path(args.lavdf_root).resolve()
        metadata_path = _resolve_metadata_path(lavdf_root)
        log(f"Generating LAV-DF manifests from {lavdf_root} using metadata {metadata_path}.")
        lavdf_manifests = generate_lavdf_manifests(
            metadata_path=metadata_path,
            output_dir=manifests_dir,
            image_frames_per_video=args.image_frames_per_video,
            audio_window_sec=args.audio_window_sec,
            audio_stride_sec=args.audio_stride_sec,
            clip_window_sec=args.clip_window_sec,
            clip_stride_sec=args.clip_stride_sec,
            manifest_names={
                "video": "video_cross_lavdf.csv",
                "multimodal": "multimodal_cross_lavdf.csv",
            },
            tasks={"video", "multimodal"},
        )
        summary["dataset_roots"]["lavdf_root"] = str(lavdf_root)
        summary["dataset_roots"]["lavdf_metadata_path"] = str(metadata_path)
        if "video" in lavdf_manifests:
            _register_manifest(catalog, key="video_cross_lavdf", task="video", dataset="lavdf", manifest_path=lavdf_manifests["video"], data_root=lavdf_root, role="cross_dataset")
        if "multimodal" in lavdf_manifests:
            _register_manifest(catalog, key="multimodal_cross_lavdf", task="multimodal", dataset="lavdf", manifest_path=lavdf_manifests["multimodal"], data_root=lavdf_root, role="cross_dataset")
        log("Registered LAV-DF video and multimodal manifests.")

    summary["manifests"] = catalog
    summary["phase0_clean_order"] = [
        key
        for key in (
            "image_primary_ffpp_c23",
            "image_cross_celebdf_v2",
            "audio_primary_asvspoof2019_la",
            "audio_cross_asvspoof2021_df",
            "video_primary_ffpp_c23",
            "video_cross_lavdf",
            "multimodal_primary_fakeavceleb",
            "multimodal_cross_lavdf",
        )
        if key in catalog
    ]
    summary["phase1_primary"] = {
        task: catalog[key]
        for task, key in (
            ("image", "image_primary_ffpp_c23"),
            ("audio", "audio_primary_asvspoof2019_la"),
            ("video", "video_primary_ffpp_c23"),
            ("multimodal", "multimodal_primary_fakeavceleb"),
        )
        if key in catalog
    }

    if not args.skip_model_validation:
        from deepshield.modeling import build_adapter, list_active_models

        validation_rows = []
        active_models = list_active_models()
        log(f"Starting model validation for {len(active_models)} active model(s).")
        for model_name in progress_iter(active_models, total=len(active_models), desc="[workspace] validate models", unit="model"):
            try:
                log(f"Validating model '{model_name}'.")
                adapter = build_adapter(model_name, assets, device=args.device)
                validation_rows.append(
                    {
                        "model": model_name,
                        "status": "ready",
                        "adapter": type(adapter).__name__,
                        "model_class": type(adapter.model).__name__,
                    }
                )
            except Exception as exc:
                validation_rows.append(
                    {
                        "model": model_name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        summary["model_validation"] = validation_rows
        ready_count = sum(1 for row in validation_rows if row["status"] == "ready")
        log(f"Model validation complete: {ready_count}/{len(validation_rows)} ready.")

    summary_path = output_dir / "kaggle_workspace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote workspace summary to {summary_path}.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
