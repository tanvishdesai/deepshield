from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.manifests import generate_lavdf_manifests
from deepshield.progress import log, log_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate modality-specific manifests from LAV-DF metadata.")
    parser.add_argument("--metadata-path", required=True, help="Path to metadata.min.json, metadata.json, or lavdf-metadata.json.")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV manifests into.")
    parser.add_argument("--image-frames-per-video", type=int, default=10)
    parser.add_argument("--audio-window-sec", type=float, default=4.0)
    parser.add_argument("--audio-stride-sec", type=float, default=2.0)
    parser.add_argument("--clip-window-sec", type=float, default=30.0)
    parser.add_argument("--clip-stride-sec", type=float, default=15.0)
    args = parser.parse_args()

    log("Starting LAV-DF manifest generation.")
    log_config(
        "Run configuration",
        {
            "metadata_path": args.metadata_path,
            "output_dir": args.output_dir,
            "image_frames_per_video": args.image_frames_per_video,
            "audio_window_sec": args.audio_window_sec,
            "audio_stride_sec": args.audio_stride_sec,
            "clip_window_sec": args.clip_window_sec,
            "clip_stride_sec": args.clip_stride_sec,
        },
    )
    manifests = generate_lavdf_manifests(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        image_frames_per_video=args.image_frames_per_video,
        audio_window_sec=args.audio_window_sec,
        audio_stride_sec=args.audio_stride_sec,
        clip_window_sec=args.clip_window_sec,
        clip_stride_sec=args.clip_stride_sec,
    )
    log(f"Generated {len(manifests)} manifest file(s).")
    for name, path in manifests.items():
        print(f"{name}: {Path(path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
