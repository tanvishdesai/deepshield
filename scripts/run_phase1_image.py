from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.progress import log, log_config
from deepshield.runners import AttackPlan, QualityGate, run_attack_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 image attacks for the active DeepShield scope.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Safe default for Kaggle image attacks.")
    parser.add_argument("--models", nargs="*", default=["xceptionnet", "f3net", "universalfakedetect", "mat", "sbi"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-clean-baseline", action="store_true", help="Skip the per-model clean pass at the start of the attack run.")
    parser.add_argument("--clean-summary-path", help="Optional clean_summary.json from Phase 0 to reuse when skipping clean baselines.")
    args = parser.parse_args()

    log("Starting Phase 1 image attack runner.")
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
            "models": args.models,
            "seeds": args.seeds,
            "skip_clean_baseline": args.skip_clean_baseline,
            "clean_summary_path": args.clean_summary_path or "not set",
        },
    )
    eps_values = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]
    c_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    attack_plans = [AttackPlan("cw", {"c": c_value, "steps": 100}) for c_value in c_values]
    for eps in eps_values:
        attack_plans.extend(
            [
                AttackPlan("fgsm", {"eps": eps}),
                AttackPlan("bim", {"eps": eps, "steps": 10}),
                AttackPlan("pgd", {"eps": eps, "steps": 100}),
                AttackPlan("mifgsm", {"eps": eps, "steps": 40}),
                AttackPlan("patch_pgd", {"eps": eps, "steps": 40, "patch_ratio": 0.2}),
                AttackPlan("frequency_targeted_pgd", {"eps": eps, "steps": 40, "high_freq_ratio": 0.5}),
                AttackPlan("autoattack", {"eps": eps}),
            ]
        )

    log(f"Prepared {len(attack_plans)} image attack plan(s).")
    log("Resolving asset layout.")
    run_attack_benchmark(
        assets=resolve_asset_layout(args.assets_root),
        manifest_path=args.manifest_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        attack_plans=attack_plans,
        models=args.models,
        split=args.split,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        quality_gate=QualityGate({"ssim": 0.95}),
        skip_clean_baseline=args.skip_clean_baseline,
        clean_summary_path=args.clean_summary_path,
    )
    log("Completed image attack suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
