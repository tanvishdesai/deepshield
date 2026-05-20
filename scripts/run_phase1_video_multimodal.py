from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.progress import log, log_config
from deepshield.runners import AttackPlan, QualityGate, run_attack_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 video and multimodal attacks for the active DeepShield scope.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--video-manifest-path", required=True)
    parser.add_argument("--multimodal-manifest-path", required=True)
    parser.add_argument("--data-root", help="Optional shared root when both manifests point to the same dataset.")
    parser.add_argument("--video-data-root", help="Root directory for the video manifest dataset.")
    parser.add_argument("--multimodal-data-root", help="Root directory for the multimodal manifest dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-models", nargs="*", default=["lipforensics", "ftcn", "realforensics"])
    parser.add_argument("--multimodal-models", nargs="*", default=["batfd", "late_fusion_baseline"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-clean-baseline", action="store_true", help="Skip the per-model clean pass at the start of each attack run.")
    parser.add_argument("--video-clean-summary-path", help="Optional clean_summary.json for the video primary benchmark.")
    parser.add_argument("--multimodal-clean-summary-path", help="Optional clean_summary.json for the multimodal primary benchmark.")
    args = parser.parse_args()

    log("Starting Phase 1 video + multimodal attack runner.")
    log_config(
        "Run configuration",
        {
            "assets_root": args.assets_root,
            "video_manifest_path": args.video_manifest_path,
            "multimodal_manifest_path": args.multimodal_manifest_path,
            "data_root": args.data_root or "not set",
            "video_data_root": args.video_data_root or "not set",
            "multimodal_data_root": args.multimodal_data_root or "not set",
            "output_dir": args.output_dir,
            "split": args.split,
            "device": args.device,
            "batch_size": args.batch_size,
            "video_models": args.video_models,
            "multimodal_models": args.multimodal_models,
            "seeds": args.seeds,
            "skip_clean_baseline": args.skip_clean_baseline,
            "video_clean_summary_path": args.video_clean_summary_path or "not set",
            "multimodal_clean_summary_path": args.multimodal_clean_summary_path or "not set",
        },
    )
    output_dir = Path(args.output_dir)
    eps_values = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
    video_plans = []
    for eps in eps_values:
        video_plans.extend(
            [
                AttackPlan("pgd", {"eps": eps, "steps": 20}),
                AttackPlan("sparse_frame_pgd", {"eps": eps, "steps": 20, "frame_fraction": 0.2}),
                AttackPlan("temporally_consistent_pgd", {"eps": eps, "steps": 20, "smoothness_weight": 0.2}),
            ]
        )
    multimodal_plans = [
        AttackPlan("asymmetric_audio", {"eps": 0.002, "steps": 20}),
        AttackPlan("asymmetric_video", {"eps": 4 / 255, "steps": 20}),
        AttackPlan("asymmetric_both", {"eps": 4 / 255, "steps": 20}),
    ]

    video_data_root = args.video_data_root or args.data_root
    multimodal_data_root = args.multimodal_data_root or args.data_root
    if not video_data_root or not multimodal_data_root:
        raise ValueError("Provide --data-root for a shared dataset root or pass both --video-data-root and --multimodal-data-root.")

    log(f"Prepared {len(video_plans)} video plan(s) and {len(multimodal_plans)} multimodal plan(s).")
    log("Resolving asset layout.")
    run_attack_benchmark(
        assets=resolve_asset_layout(args.assets_root),
        manifest_path=args.video_manifest_path,
        data_root=video_data_root,
        output_dir=output_dir / "video",
        attack_plans=video_plans,
        models=args.video_models,
        split=args.split,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        quality_gate=QualityGate({"ssim": 0.95}),
        skip_clean_baseline=args.skip_clean_baseline,
        clean_summary_path=args.video_clean_summary_path,
    )
    log("Completed video attack suite. Starting multimodal attack suite.")
    run_attack_benchmark(
        assets=resolve_asset_layout(args.assets_root),
        manifest_path=args.multimodal_manifest_path,
        data_root=multimodal_data_root,
        output_dir=output_dir / "multimodal",
        attack_plans=multimodal_plans,
        models=args.multimodal_models,
        split=args.split,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        quality_gate=QualityGate({"video_ssim": 0.95, "audio_pesq": 3.5}),
        skip_clean_baseline=args.skip_clean_baseline,
        clean_summary_path=args.multimodal_clean_summary_path,
    )
    log("Completed video and multimodal attack suites.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
