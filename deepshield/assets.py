from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetLayout:
    root: Path
    model_repos: Path
    checkpoints: Path


def _candidate_roots(root: Path) -> list[Path]:
    candidates = [root]
    candidates.extend(root.iterdir() if root.exists() and root.is_dir() else [])
    return [candidate for candidate in candidates if candidate.is_dir()]


def resolve_asset_layout(root: str | Path) -> AssetLayout:
    root = Path(root).resolve()
    for candidate in _candidate_roots(root):
        model_repos = candidate / "model_repos"
        checkpoints = candidate / "artifacts" / "checkpoints"
        if model_repos.exists() and checkpoints.exists():
            return AssetLayout(root=candidate, model_repos=model_repos, checkpoints=checkpoints)
    raise FileNotFoundError(
        f"Could not locate a DeepShield asset bundle under {root}. "
        "Expected to find model_repos/ and artifacts/checkpoints/."
    )
