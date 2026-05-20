from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.progress import log, log_config
from deepshield.runners import run_clean_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the intended Phase 0 clean benchmark matrix across primary and cross-dataset manifests.")
    parser.add_argument("--assets-root", required=True, help="Root containing model_repos/ and artifacts/checkpoints/.")
    parser.add_argument("--workspace-summary", required=True, help="Path to kaggle_workspace_summary.json from prepare_kaggle_workspace.py.")
    parser.add_argument("--output-dir", required=True, help="Directory to store the clean benchmark outputs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=1, help="Safe default for image/audio on Kaggle. Increase manually after a smoke test succeeds.")
    parser.add_argument("--sequence-batch-size", type=int, default=1, help="Batch size for video and multimodal runs.")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Optional subset of manifest keys from phase0_clean_order.")
    args = parser.parse_args()

    log("Starting Phase 0 clean matrix runner.")
    log_config(
        "Run configuration",
        {
            "assets_root": args.assets_root,
            "workspace_summary": args.workspace_summary,
            "output_dir": args.output_dir,
            "split": args.split,
            "device": args.device,
            "batch_size": args.batch_size,
            "sequence_batch_size": args.sequence_batch_size,
            "benchmarks": args.benchmarks or "workspace phase0_clean_order",
            "seeds": args.seeds,
        },
    )
    log("Resolving asset layout.")
    assets = resolve_asset_layout(args.assets_root)
    log(f"Resolved assets under {assets.root}.")
    workspace = json.loads(Path(args.workspace_summary).read_text(encoding="utf-8"))
    manifest_catalog = workspace.get("manifests", {})
    selected_keys = args.benchmarks or workspace.get("phase0_clean_order", [])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Selected {len(selected_keys)} benchmark(s).")

    for key in selected_keys:
        if key not in manifest_catalog:
            raise KeyError(f"Unknown benchmark key '{key}' in workspace summary.")
        entry = manifest_catalog[key]
        task = entry["task"]
        batch_size = args.sequence_batch_size if task in {"video", "multimodal"} else args.batch_size
        log(
            f"Starting benchmark '{key}' for task={task} with data_root={entry['data_root']} "
            f"and batch_size={batch_size}."
        )
        run_clean_benchmark(
            assets=assets,
            manifest_path=entry["manifest_path"],
            data_root=entry["data_root"],
            output_dir=output_dir / key,
            split=args.split,
            seeds=args.seeds,
            batch_size=batch_size,
            device=args.device,
        )
        log(f"Completed benchmark '{key}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
