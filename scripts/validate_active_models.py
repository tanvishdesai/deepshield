from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.assets import resolve_asset_layout
from deepshield.modeling import build_adapter, list_active_models
from deepshield.progress import log, log_config, progress_iter


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test construction of all active DeepShield model adapters.")
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    log("Starting active-model validation.")
    log_config(
        "Run configuration",
        {
            "assets_root": args.assets_root,
            "output_path": args.output_path,
            "device": args.device,
            "models": args.models or "all active models",
        },
    )
    assets = resolve_asset_layout(args.assets_root)
    output_path = Path(args.output_path)
    results = []

    selected_models = args.models or list_active_models()
    log(f"Resolved assets under {assets.root}. Validating {len(selected_models)} model(s).")
    for model_name in progress_iter(selected_models, total=len(selected_models), desc="[validate-active] models", unit="model"):
        try:
            log(f"Building adapter for '{model_name}'.")
            adapter = build_adapter(model_name, assets, device=args.device)
            results.append(
                {
                    "model": model_name,
                    "status": "ready",
                    "adapter": type(adapter).__name__,
                    "model_class": type(adapter.model).__name__,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ready = sum(1 for item in results if item["status"] == "ready")
    log(f"Validated {ready}/{len(results)} active models. Results written to {output_path.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
