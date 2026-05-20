from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepshield.progress import log, log_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a DeepShield asset bundle inside a Kaggle working directory.")
    parser.add_argument("--bundle-path", required=True, help="Path to a directory or zip file containing model_repos/ and artifacts/checkpoints/.")
    parser.add_argument("--output-dir", required=True, help="Destination directory where the asset bundle should be extracted or copied.")
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("Starting Kaggle asset staging.")
    log_config(
        "Asset staging configuration",
        {
            "bundle_path": bundle_path,
            "output_dir": output_dir,
            "bundle_type": "directory" if bundle_path.is_dir() else bundle_path.suffix.lower() or "unknown",
        },
    )

    if bundle_path.is_dir():
        target = output_dir / bundle_path.name
        if target.exists():
            log(f"Removing existing directory at {target}.")
            shutil.rmtree(target)
        log(f"Copying asset directory into {target}.")
        shutil.copytree(bundle_path, target)
        log(f"Completed asset staging into {target}.")
        print(target)
        return 0

    if bundle_path.suffix.lower() == ".zip":
        log(f"Extracting zip bundle into {output_dir}.")
        with zipfile.ZipFile(bundle_path, "r") as archive:
            archive.extractall(output_dir)
        log(f"Completed zip extraction into {output_dir}.")
        print(output_dir)
        return 0

    raise ValueError("bundle-path must point to either a directory or a .zip file.")


if __name__ == "__main__":
    raise SystemExit(main())
