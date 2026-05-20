from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .progress import log, progress_iter


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


@dataclass(frozen=True)
class LavdfMetadata:
    file: str
    n_fakes: int
    fake_periods: list[list[float]]
    duration: float
    original: str | None
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: str
    task: str
    file: str
    label: int
    start_sec: float
    end_sec: float
    timestamp_sec: float
    frame_index: int
    duration_sec: float
    source_type: str
    modify_video: bool
    modify_audio: bool
    original: str | None
    dataset: str = ""


def _normalize_relpath(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def _source_type(meta: LavdfMetadata) -> str:
    if meta.modify_video and meta.modify_audio:
        return "both"
    if meta.modify_video:
        return "video_only"
    if meta.modify_audio:
        return "audio_only"
    return "real"


def _metadata_from_path(path: Path) -> list[LavdfMetadata]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LavdfMetadata(**item) for item in raw]


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            records.append(
                ManifestRecord(
                    sample_id=row["sample_id"],
                    split=row["split"],
                    task=row["task"],
                    file=row["file"],
                    label=int(row["label"]),
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    frame_index=int(row["frame_index"]),
                    duration_sec=float(row["duration_sec"]),
                    source_type=row["source_type"],
                    modify_video=row["modify_video"].lower() == "true",
                    modify_audio=row["modify_audio"].lower() == "true",
                    original=row["original"] or None,
                    dataset=row.get("dataset", ""),
                )
            )
        return records


def _point_in_fake(timestamp_sec: float, periods: list[list[float]]) -> bool:
    return any(begin <= timestamp_sec <= end for begin, end in periods)


def _segment_overlaps_fake(start_sec: float, end_sec: float, periods: list[list[float]]) -> bool:
    return any(max(start_sec, begin) < min(end_sec, end) for begin, end in periods)


def _segment_windows(duration: float, window: float, stride: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return [(0.0, max(duration, 0.0))]
    if duration <= window:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window)
        windows.append((start, end))
        if end >= duration:
            break
        start += stride
    return windows


def _write_manifest(path: Path, rows: Iterable[ManifestRecord]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[manifest] Writing {len(rows)} rows to {path.resolve()}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRecord.__annotations__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _iter_media_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _probe_video(path: Path) -> tuple[float, int, float]:
    try:
        import cv2
    except Exception:
        return 0.0, 1, 25.0
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = max(frames, 1)
        duration = frames / fps if fps > 0 else 0.0
        return duration, frames, fps
    finally:
        capture.release()


def _probe_audio(path: Path) -> tuple[float, int, int]:
    try:
        import torchaudio
    except Exception:
        return 0.0, 1, 16000
    info = torchaudio.info(str(path))
    sample_rate = int(info.sample_rate or 16000)
    frames = int(info.num_frames or sample_rate)
    duration = frames / sample_rate if sample_rate > 0 else 0.0
    return duration, frames, sample_rate


def _uniform_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    frame_count = max(frame_count, 1)
    if num_frames <= 1:
        return [max(frame_count // 2, 0)]
    if frame_count == 1:
        return [0] * num_frames
    return [round((frame_count - 1) * idx / max(num_frames - 1, 1)) for idx in range(num_frames)]


def _video_source_type(modify_video: bool, modify_audio: bool) -> str:
    if modify_video and modify_audio:
        return "both"
    if modify_video:
        return "video_only"
    if modify_audio:
        return "audio_only"
    return "real"


def _hashed_split(value: str) -> str:
    bucket = int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def _image_rows_from_video(
    *,
    dataset: str,
    split: str,
    relative_path: str,
    label: int,
    frame_count: int,
    fps: float,
    duration: float,
    source_type: str,
    modify_video: bool,
    modify_audio: bool,
    original: str | None = None,
    frames_per_video: int = 10,
) -> list[ManifestRecord]:
    rows: list[ManifestRecord] = []
    for local_index, frame_index in enumerate(_uniform_frame_indices(frame_count, frames_per_video)):
        timestamp = min(frame_index / max(fps, 1e-6), duration) if duration > 0 else 0.0
        rows.append(
            ManifestRecord(
                sample_id=f"image::{dataset}::{relative_path}::{local_index}",
                split=split,
                task="image",
                file=relative_path,
                label=label,
                start_sec=timestamp,
                end_sec=timestamp,
                timestamp_sec=timestamp,
                frame_index=frame_index,
                duration_sec=duration,
                source_type=source_type,
                modify_video=modify_video,
                modify_audio=modify_audio,
                original=original,
                dataset=dataset,
            )
        )
    return rows


def _video_rows_from_video(
    *,
    task: str,
    dataset: str,
    split: str,
    relative_path: str,
    label: int,
    duration: float,
    source_type: str,
    modify_video: bool,
    modify_audio: bool,
    original: str | None = None,
    clip_window_sec: float = 30.0,
    clip_stride_sec: float = 15.0,
) -> list[ManifestRecord]:
    rows: list[ManifestRecord] = []
    for segment_idx, (start_sec, end_sec) in enumerate(_segment_windows(duration, clip_window_sec, clip_stride_sec)):
        rows.append(
            ManifestRecord(
                sample_id=f"{task}::{dataset}::{relative_path}::{segment_idx}",
                split=split,
                task=task,
                file=relative_path,
                label=label,
                start_sec=start_sec,
                end_sec=end_sec,
                timestamp_sec=start_sec,
                frame_index=-1,
                duration_sec=duration,
                source_type=source_type,
                modify_video=modify_video,
                modify_audio=modify_audio,
                original=original,
                dataset=dataset,
            )
        )
    return rows


def _audio_rows_from_file(
    *,
    dataset: str,
    split: str,
    relative_path: str,
    label: int,
    duration: float,
    source_type: str,
    modify_audio: bool,
    original: str | None = None,
) -> list[ManifestRecord]:
    return [
        ManifestRecord(
            sample_id=f"audio::{dataset}::{Path(relative_path).stem}",
            split=split,
            task="audio",
            file=relative_path,
            label=label,
            start_sec=0.0,
            end_sec=max(duration, 0.0),
            timestamp_sec=0.0,
            frame_index=-1,
            duration_sec=max(duration, 0.0),
            source_type=source_type,
            modify_video=False,
            modify_audio=modify_audio,
            original=original,
            dataset=dataset,
        )
    ]


def generate_lavdf_manifests(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    image_frames_per_video: int = 10,
    audio_window_sec: float = 4.0,
    audio_stride_sec: float = 2.0,
    clip_window_sec: float = 30.0,
    clip_stride_sec: float = 15.0,
    manifest_names: dict[str, str] | None = None,
    dataset_name: str = "lavdf",
    tasks: set[str] | None = None,
) -> dict[str, Path]:
    metadata = _metadata_from_path(Path(metadata_path))
    output_dir = Path(output_dir)
    log(f"[manifest] LAV-DF: loaded {len(metadata)} metadata entries from {Path(metadata_path).resolve()}.")

    image_rows: list[ManifestRecord] = []
    audio_rows: list[ManifestRecord] = []
    video_rows: list[ManifestRecord] = []
    multimodal_rows: list[ManifestRecord] = []

    for meta in progress_iter(metadata, total=len(metadata), desc="[manifest] LAV-DF entries", unit="entry", leave=False):
        source_type = _source_type(meta)
        frame_indices = _uniform_frame_indices(meta.video_frames or 1, image_frames_per_video)
        fps = (meta.video_frames / meta.duration) if meta.duration > 0 else 25.0
        for frame_number, frame_index in enumerate(frame_indices):
            timestamp = min(frame_index / max(fps, 1e-6), meta.duration) if meta.duration > 0 else 0.0
            label = int(meta.modify_video and _point_in_fake(timestamp, meta.fake_periods))
            image_rows.append(
                ManifestRecord(
                    sample_id=f"image::{dataset_name}::{meta.file}::{frame_number}",
                    split=meta.split,
                    task="image",
                    file=meta.file,
                    label=label,
                    start_sec=timestamp,
                    end_sec=timestamp,
                    timestamp_sec=timestamp,
                    frame_index=min(frame_index, max(meta.video_frames - 1, 0)),
                    duration_sec=meta.duration,
                    source_type=source_type,
                    modify_video=meta.modify_video,
                    modify_audio=meta.modify_audio,
                    original=meta.original,
                    dataset=dataset_name,
                )
            )

        for segment_idx, (start_sec, end_sec) in enumerate(_segment_windows(meta.duration, audio_window_sec, audio_stride_sec)):
            audio_label = int(meta.modify_audio and _segment_overlaps_fake(start_sec, end_sec, meta.fake_periods))
            audio_rows.append(
                ManifestRecord(
                    sample_id=f"audio::{dataset_name}::{meta.file}::{segment_idx}",
                    split=meta.split,
                    task="audio",
                    file=meta.file,
                    label=audio_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    timestamp_sec=start_sec,
                    frame_index=-1,
                    duration_sec=meta.duration,
                    source_type=source_type,
                    modify_video=meta.modify_video,
                    modify_audio=meta.modify_audio,
                    original=meta.original,
                    dataset=dataset_name,
                )
            )

        for segment_idx, (start_sec, end_sec) in enumerate(_segment_windows(meta.duration, clip_window_sec, clip_stride_sec)):
            video_label = int(meta.modify_video and _segment_overlaps_fake(start_sec, end_sec, meta.fake_periods))
            multimodal_label = int((meta.modify_video or meta.modify_audio) and _segment_overlaps_fake(start_sec, end_sec, meta.fake_periods))
            video_rows.append(
                ManifestRecord(
                    sample_id=f"video::{dataset_name}::{meta.file}::{segment_idx}",
                    split=meta.split,
                    task="video",
                    file=meta.file,
                    label=video_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    timestamp_sec=start_sec,
                    frame_index=-1,
                    duration_sec=meta.duration,
                    source_type=source_type,
                    modify_video=meta.modify_video,
                    modify_audio=meta.modify_audio,
                    original=meta.original,
                    dataset=dataset_name,
                )
            )
            multimodal_rows.append(
                ManifestRecord(
                    sample_id=f"multimodal::{dataset_name}::{meta.file}::{segment_idx}",
                    split=meta.split,
                    task="multimodal",
                    file=meta.file,
                    label=multimodal_label,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    timestamp_sec=start_sec,
                    frame_index=-1,
                    duration_sec=meta.duration,
                    source_type=source_type,
                    modify_video=meta.modify_video,
                    modify_audio=meta.modify_audio,
                    original=meta.original,
                    dataset=dataset_name,
                )
            )

    manifest_names = manifest_names or {}
    selected_tasks = tasks or {"image", "audio", "video", "multimodal"}
    manifests: dict[str, Path] = {}
    if "image" in selected_tasks:
        manifests["image"] = output_dir / manifest_names.get("image", "image_manifest.csv")
        _write_manifest(manifests["image"], image_rows)
    if "audio" in selected_tasks:
        manifests["audio"] = output_dir / manifest_names.get("audio", "audio_manifest.csv")
        _write_manifest(manifests["audio"], audio_rows)
    if "video" in selected_tasks:
        manifests["video"] = output_dir / manifest_names.get("video", "video_manifest.csv")
        _write_manifest(manifests["video"], video_rows)
    if "multimodal" in selected_tasks:
        manifests["multimodal"] = output_dir / manifest_names.get("multimodal", "multimodal_manifest.csv")
        _write_manifest(manifests["multimodal"], multimodal_rows)

    summary = {
        "dataset": dataset_name,
        "metadata_entries": len(metadata),
        "image_rows": len(image_rows),
        "audio_rows": len(audio_rows),
        "video_rows": len(video_rows),
        "multimodal_rows": len(multimodal_rows),
    }
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"[manifest] LAV-DF summary: image={len(image_rows)}, audio={len(audio_rows)}, "
        f"video={len(video_rows)}, multimodal={len(multimodal_rows)}."
    )
    return manifests


def _load_ffpp_split_maps(splits_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    pair_splits: dict[str, str] = {}
    id_splits: dict[str, str] = {}
    for source_name, split_name in (("train", "train"), ("val", "dev"), ("test", "test")):
        pairs = json.loads((splits_root / f"{source_name}.json").read_text(encoding="utf-8"))
        for left, right in pairs:
            pair_splits[f"{left}_{right}"] = split_name
            pair_splits[f"{right}_{left}"] = split_name
            id_splits[left] = split_name
            id_splits[right] = split_name
    return pair_splits, id_splits


def generate_ffpp_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    splits_root: str | Path,
    image_frames_per_video: int = 10,
    clip_window_sec: float = 30.0,
    clip_stride_sec: float = 15.0,
    image_manifest_name: str = "image_manifest.csv",
    video_manifest_name: str = "video_manifest.csv",
    dataset_name: str = "ffpp_c23",
) -> dict[str, Path]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    pair_splits, id_splits = _load_ffpp_split_maps(Path(splits_root).resolve())
    fake_tokens = {
        "deepfakes",
        "deepfakedetection",
        "face2face",
        "faceswap",
        "faceshifter",
        "neuraltextures",
        "manipulated_sequences",
    }

    image_rows: list[ManifestRecord] = []
    video_rows: list[ManifestRecord] = []
    media_files = _iter_media_files(dataset_root, VIDEO_EXTENSIONS)
    log(f"[manifest] FFPP: found {len(media_files)} candidate videos under {dataset_root}.")

    for path in progress_iter(media_files, total=len(media_files), desc="[manifest] FFPP videos", unit="video", leave=False):
        relative_path = _normalize_relpath(path.relative_to(dataset_root))
        parts_lower = [part.lower() for part in Path(relative_path).parts]
        stem = path.stem
        is_real = any(part in {"original", "original_sequences"} for part in parts_lower)
        is_fake = any(part in fake_tokens for part in parts_lower)
        if not is_real and not is_fake:
            continue
        split = pair_splits.get(stem) if "_" in stem else id_splits.get(stem)
        if split is None:
            continue
        label = int(is_fake)
        duration, frame_count, fps = _probe_video(path)
        source_type = _video_source_type(modify_video=bool(label), modify_audio=False)
        image_rows.extend(
            _image_rows_from_video(
                dataset=dataset_name,
                split=split,
                relative_path=relative_path,
                label=label,
                frame_count=frame_count,
                fps=fps,
                duration=duration,
                source_type=source_type,
                modify_video=bool(label),
                modify_audio=False,
                frames_per_video=image_frames_per_video,
            )
        )
        video_rows.extend(
            _video_rows_from_video(
                task="video",
                dataset=dataset_name,
                split=split,
                relative_path=relative_path,
                label=label,
                duration=duration,
                source_type=source_type,
                modify_video=bool(label),
                modify_audio=False,
                clip_window_sec=clip_window_sec,
                clip_stride_sec=clip_stride_sec,
            )
        )

    manifests = {
        "image": output_dir / image_manifest_name,
        "video": output_dir / video_manifest_name,
    }
    _write_manifest(manifests["image"], image_rows)
    _write_manifest(manifests["video"], video_rows)
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(
        json.dumps({"dataset": dataset_name, "image_rows": len(image_rows), "video_rows": len(video_rows)}, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] FFPP summary: image={len(image_rows)}, video={len(video_rows)}.")
    return manifests


def generate_celebdf_image_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    image_frames_per_video: int = 10,
    manifest_name: str = "image_manifest.csv",
    dataset_name: str = "celebdf_v2",
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    testing_list = next(iter(sorted(dataset_root.rglob("List_of_testing_videos.txt"))), None)
    if testing_list is None:
        candidates = [path.relative_to(dataset_root) for path in _iter_media_files(dataset_root, VIDEO_EXTENSIONS)]
        log("[manifest] Celeb-DF: no testing list found, falling back to scanning all video files.")
    else:
        log(f"[manifest] Celeb-DF: using testing list from {testing_list}.")
        candidates = []
        for line in testing_list.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[0] in {"0", "1"}:
                text = parts[1]
            candidates.append(Path(_normalize_relpath(text)))
    log(f"[manifest] Celeb-DF: processing {len(candidates)} candidate videos from {dataset_root}.")

    image_rows: list[ManifestRecord] = []
    missing_count = 0
    for relative in progress_iter(candidates, total=len(candidates), desc="[manifest] Celeb-DF videos", unit="video", leave=False):
        path = dataset_root / relative
        if not path.exists():
            missing_count += 1
            continue
        relative_path = _normalize_relpath(relative)
        parts_lower = [part.lower() for part in Path(relative_path).parts]
        label = 1 if any("synthesis" in part for part in parts_lower) else 0
        duration, frame_count, fps = _probe_video(path)
        image_rows.extend(
            _image_rows_from_video(
                dataset=dataset_name,
                split="test",
                relative_path=relative_path,
                label=label,
                frame_count=frame_count,
                fps=fps,
                duration=duration,
                source_type=_video_source_type(modify_video=bool(label), modify_audio=False),
                modify_video=bool(label),
                modify_audio=False,
                frames_per_video=image_frames_per_video,
            )
        )

    manifest_path = output_dir / manifest_name
    _write_manifest(manifest_path, image_rows)
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(
        json.dumps({"dataset": dataset_name, "image_rows": len(image_rows)}, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] Celeb-DF summary: image={len(image_rows)}, missing_candidates={missing_count}.")
    return manifest_path


def _discover_asvspoof2019_protocols(dataset_root: Path) -> dict[str, Path]:
    mapping = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    discovered: dict[str, Path] = {}
    for split, filename in mapping.items():
        matches = sorted(dataset_root.rglob(filename))
        if matches:
            discovered[split] = matches[0]
            continue
        fallback = sorted(dataset_root.rglob(f"*{split}*.txt"))
        if fallback:
            discovered[split] = fallback[0]
    return discovered


def _discover_asvspoof2019_audio_dirs(dataset_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    split_tokens = (("train", "ASVspoof2019_LA_train"), ("dev", "ASVspoof2019_LA_dev"), ("eval", "ASVspoof2019_LA_eval"))
    flac_dirs = sorted(path for path in dataset_root.rglob("flac") if path.is_dir())
    log(f"[manifest] ASVspoof2019-LA: scanning {len(flac_dirs)} flac directories to locate train/dev/eval roots.")
    for path in progress_iter(flac_dirs, total=len(flac_dirs), desc="[manifest] ASVspoof2019 locate flac dirs", unit="dir", leave=False):
        parent_lower = str(path.parent).lower()
        for split, token in split_tokens:
            if split not in discovered and token.lower() in parent_lower:
                discovered[split] = path
        if len(discovered) == len(split_tokens):
            break
    return discovered


def generate_asvspoof2019_la_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    manifest_name: str = "audio_manifest.csv",
    dataset_name: str = "asvspoof2019_la",
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    protocols = _discover_asvspoof2019_protocols(dataset_root)
    audio_dirs = _discover_asvspoof2019_audio_dirs(dataset_root)
    log(
        f"[manifest] ASVspoof2019-LA: discovered protocol files for {sorted(protocols)} "
        f"and audio roots for {sorted(audio_dirs)}."
    )
    required = {"train", "dev", "eval"}
    if required - protocols.keys():
        missing = ", ".join(sorted(required - protocols.keys()))
        raise FileNotFoundError(f"Missing ASVspoof 2019 LA protocol files for: {missing}")
    if required - audio_dirs.keys():
        missing = ", ".join(sorted(required - audio_dirs.keys()))
        raise FileNotFoundError(f"Missing ASVspoof 2019 LA audio directories for: {missing}")

    log("[manifest] ASVspoof2019-LA: building utterance-level manifest without per-file duration probing.")
    split_map = {"train": "train", "dev": "dev", "eval": "test"}
    rows: list[ManifestRecord] = []
    for raw_split in ("train", "dev", "eval"):
        lines = protocols[raw_split].read_text(encoding="utf-8").splitlines()
        log(f"[manifest] ASVspoof2019-LA: processing {len(lines)} protocol rows for split '{raw_split}'.")
        for line in progress_iter(
            lines,
            total=len(lines),
            desc=f"[manifest] ASVspoof2019 {raw_split}",
            unit="row",
            leave=False,
        ):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            key = parts[1]
            label_token = parts[-1].lower()
            label = 0 if label_token == "bonafide" else 1
            audio_path = audio_dirs[raw_split] / f"{key}.flac"
            if not audio_path.exists():
                continue
            rows.extend(
                _audio_rows_from_file(
                    dataset=dataset_name,
                    split=split_map[raw_split],
                    relative_path=_normalize_relpath(audio_path.relative_to(dataset_root)),
                    label=label,
                    duration=0.0,
                    source_type="real" if label == 0 else "audio_only",
                    modify_audio=bool(label),
                )
            )

    manifest_path = output_dir / manifest_name
    _write_manifest(manifest_path, rows)
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(
        json.dumps({"dataset": dataset_name, "audio_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] ASVspoof2019-LA summary: audio={len(rows)}.")
    return manifest_path


def _discover_asvspoof2021_df_protocol(dataset_root: Path, keys_root: Path | None) -> Path:
    search_roots = [root for root in (keys_root, dataset_root) if root is not None]
    for root in search_roots:
        matches = sorted(root.rglob("trial_metadata.txt"))
        for match in matches:
            normalized = _normalize_relpath(match.relative_to(root))
            if normalized.endswith("DF/CM/trial_metadata.txt"):
                return match
    raise FileNotFoundError("Could not find ASVspoof 2021 DF trial_metadata.txt under the provided roots.")


def _index_audio_by_stem(dataset_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _iter_media_files(dataset_root, AUDIO_EXTENSIONS):
        index.setdefault(path.stem, path)
    return index


def _subset_to_split(subset: str) -> str:
    value = subset.strip().lower()
    if value == "train":
        return "train"
    if value in {"dev", "progress"}:
        return "dev"
    return "test"


def generate_asvspoof2021_df_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    keys_root: str | Path | None = None,
    manifest_name: str = "audio_manifest.csv",
    dataset_name: str = "asvspoof2021_df",
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    keys_root_path = Path(keys_root).resolve() if keys_root else None
    protocol_path = _discover_asvspoof2021_df_protocol(dataset_root, keys_root_path)
    audio_index = _index_audio_by_stem(dataset_root)
    protocol_lines = protocol_path.read_text(encoding="utf-8").splitlines()
    log(
        f"[manifest] ASVspoof2021-DF: loaded {len(protocol_lines)} protocol rows and "
        f"indexed {len(audio_index)} audio files."
    )
    log("[manifest] ASVspoof2021-DF: building utterance-level manifest without per-file duration probing.")

    rows: list[ManifestRecord] = []
    for line in progress_iter(
        protocol_lines,
        total=len(protocol_lines),
        desc="[manifest] ASVspoof2021 rows",
        unit="row",
        leave=False,
    ):
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        key = parts[1]
        label_token = parts[5].lower()
        subset = parts[7]
        audio_path = audio_index.get(key)
        if audio_path is None:
            continue
        label = 0 if label_token == "bonafide" else 1
        rows.extend(
            _audio_rows_from_file(
                dataset=dataset_name,
                split=_subset_to_split(subset),
                relative_path=_normalize_relpath(audio_path.relative_to(dataset_root)),
                label=label,
                duration=0.0,
                source_type="real" if label == 0 else "audio_only",
                modify_audio=bool(label),
            )
        )

    manifest_path = output_dir / manifest_name
    _write_manifest(manifest_path, rows)
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(
        json.dumps({"dataset": dataset_name, "audio_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] ASVspoof2021-DF summary: audio={len(rows)}.")
    return manifest_path


def _fakeavceleb_meta_index(dataset_root: Path) -> tuple[dict[str, dict[str, str]], str | None]:
    metadata_path = next(iter(sorted(dataset_root.rglob("meta_data.csv"))), None)
    if metadata_path is None:
        return {}, None
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}, None
    path_field = next(
        (
            field
            for field in reader.fieldnames or []
            if field and any(token in field.lower() for token in ("path", "file", "video", "name"))
        ),
        None,
    )
    if path_field is None:
        return {}, None
    index = {_normalize_relpath(row[path_field]): row for row in rows if row.get(path_field)}
    return index, path_field


def _fakeavceleb_split(relative_path: str, metadata_row: dict[str, str] | None) -> str:
    if metadata_row:
        for key, value in metadata_row.items():
            if value and key and any(token in key.lower() for token in ("split", "partition", "set")):
                lowered = value.strip().lower()
                if lowered in {"train", "training"}:
                    return "train"
                if lowered in {"val", "valid", "validation", "dev"}:
                    return "dev"
                if lowered in {"test", "eval", "evaluation"}:
                    return "test"
    return _hashed_split(relative_path)


def _fakeavceleb_source_type(relative_path: str) -> tuple[str, bool, bool]:
    normalized = relative_path.lower()
    if "fakevideo-fakeaudio" in normalized:
        return "both", True, True
    if "fakevideo-realaudio" in normalized:
        return "video_only", True, False
    if "realvideo-fakeaudio" in normalized:
        return "audio_only", False, True
    if "realvideo-realaudio" in normalized:
        return "real", False, False
    return ("both" if "fake" in normalized else "real"), ("fake" in normalized), ("fake" in normalized)


def generate_fakeavceleb_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    clip_window_sec: float = 30.0,
    clip_stride_sec: float = 15.0,
    manifest_name: str = "multimodal_manifest.csv",
    dataset_name: str = "fakeavceleb",
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir)
    metadata_index, _ = _fakeavceleb_meta_index(dataset_root)
    media_files = _iter_media_files(dataset_root, VIDEO_EXTENSIONS)
    log(
        f"[manifest] FakeAVCeleb: found {len(media_files)} candidate videos and "
        f"{len(metadata_index)} metadata entries under {dataset_root}."
    )

    rows: list[ManifestRecord] = []
    for path in progress_iter(media_files, total=len(media_files), desc="[manifest] FakeAVCeleb videos", unit="video", leave=False):
        relative_path = _normalize_relpath(path.relative_to(dataset_root))
        metadata_row = metadata_index.get(relative_path)
        source_type, modify_video, modify_audio = _fakeavceleb_source_type(relative_path)
        duration, _, _ = _probe_video(path)
        rows.extend(
            _video_rows_from_video(
                task="multimodal",
                dataset=dataset_name,
                split=_fakeavceleb_split(relative_path, metadata_row),
                relative_path=relative_path,
                label=int(modify_video or modify_audio),
                duration=duration,
                source_type=source_type,
                modify_video=modify_video,
                modify_audio=modify_audio,
                clip_window_sec=clip_window_sec,
                clip_stride_sec=clip_stride_sec,
            )
        )

    manifest_path = output_dir / manifest_name
    _write_manifest(manifest_path, rows)
    (output_dir / f"{dataset_name}_manifest_summary.json").write_text(
        json.dumps({"dataset": dataset_name, "multimodal_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] FakeAVCeleb summary: multimodal={len(rows)}.")
    return manifest_path
