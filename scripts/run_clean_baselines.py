from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.progress import log, log_config
from deepshield.runners import run_clean_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0 clean baselines for a single manifest/dataset pairing.")
    parser.add_argument("--assets-root", required=True, help="Root directory containing model_repos/ and artifacts/checkpoints/.")
    parser.add_argument("--manifest-path", required=True, help="Manifest CSV for the target task.")
    parser.add_argument("--data-root", required=True, help="Root directory for the media files referenced by the manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to store clean baseline outputs.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Safe default for Kaggle. Increase manually after a smoke test succeeds.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of models. Defaults to all active models for the manifest task.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    log("Starting Phase 0 clean baseline runner.")
    log_config(
        "Run configuration",
        {
            "assets_root": args.assets_root,
            "manifest_path": args.manifest_path,
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "split": args.split,
            "device": args.device,
            "batch_size": args.batch_size,
            "models": args.models or "all active models for manifest task",
            "seeds": args.seeds,
        },
    )
    log("Resolving asset layout.")
    assets = resolve_asset_layout(args.assets_root)
    log(f"Resolved assets under {assets.root}.")
    models = args.models if args.models else None
    run_clean_benchmark(
        assets=assets,
        manifest_path=args.manifest_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        models=models,
        split=args.split,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
    )
    log("Completed clean baseline run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
