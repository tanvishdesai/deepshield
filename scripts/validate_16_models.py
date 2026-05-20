from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
import types
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOS = ROOT / "model_repos"
CHECKPOINTS = ROOT / "artifacts" / "checkpoints"
REPORT_DIR = ROOT / "artifacts" / "model_validation"
SCRIPT_PATH = Path(__file__).resolve()
JSON_SENTINEL = "JSON_RESULT::"

sys.path.insert(0, str(ROOT))

from deepshield.progress import log, log_config, progress_iter


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    repo_paths: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...] = ()


MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="xceptionnet",
        label="XceptionNet",
        repo_paths=(MODEL_REPOS / "FaceForensics", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "xception_best.pth",),
    ),
    ModelEntry(
        key="f3net",
        label="F3-Net",
        repo_paths=(MODEL_REPOS / "F3Net", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "f3net_best.pth",),
    ),
    ModelEntry(
        key="universalfakedetect",
        label="UniversalFakeDetect",
        repo_paths=(MODEL_REPOS / "UniversalFakeDetect",),
        checkpoint_paths=(MODEL_REPOS / "UniversalFakeDetect" / "pretrained_weights" / "fc_weights.pth",),
    ),
    ModelEntry(
        key="mat",
        label="MAT",
        repo_paths=(MODEL_REPOS / "multiple-attention", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "mat_bundle" / "multi-attention" / "pretrained" / "ff_c23.pth",),
    ),
    ModelEntry(
        key="sbi",
        label="SBI",
        repo_paths=(MODEL_REPOS / "SelfBlendedImages", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "sbi_c23.tar",),
    ),
    ModelEntry(
        key="aasist",
        label="AASIST",
        repo_paths=(MODEL_REPOS / "aasist",),
        checkpoint_paths=(MODEL_REPOS / "aasist" / "models" / "weights" / "AASIST.pth",),
    ),
    ModelEntry(
        key="rawnet2",
        label="RawNet2",
        repo_paths=(MODEL_REPOS / "RawNet", MODEL_REPOS / "rawnet2-antispoofing", MODEL_REPOS / "2021"),
        checkpoint_paths=(CHECKPOINTS / "rawnet2_pretrained" / "pre_trained_DF_RawNet2.pth",),
    ),
    ModelEntry(
        key="lcnn",
        label="LCNN",
        repo_paths=(MODEL_REPOS / "aasist", MODEL_REPOS / "2021"),
        checkpoint_paths=(CHECKPOINTS / "lcnn_pretrained" / "la_trained_network.pt",),
    ),
    ModelEntry(
        key="wav2vec2_classifier",
        label="Wav2Vec2 + classifier",
        repo_paths=(MODEL_REPOS / "SSL_Anti-spoofing",),
        checkpoint_paths=(
            CHECKPOINTS / "ssl_antispoofing" / "LA_model.pth",
            CHECKPOINTS / "ssl_antispoofing" / "Best_LA_model_for_DF.pth",
            CHECKPOINTS / "xlsr2_300m.pt",
        ),
    ),
    ModelEntry(
        key="lipforensics",
        label="LipForensics",
        repo_paths=(MODEL_REPOS / "LipForensics",),
        checkpoint_paths=(CHECKPOINTS / "lipforensics_ff.pth",),
    ),
    ModelEntry(
        key="ftcn",
        label="FTCN",
        repo_paths=(MODEL_REPOS / "FTCN", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "ftcn_tt.pth", CHECKPOINTS / "I3D_8x8_R50.pth"),
    ),
    ModelEntry(
        key="altfreezing",
        label="AltFreezing",
        repo_paths=(MODEL_REPOS / "AltFreezing", MODEL_REPOS / "DeepfakeBench"),
        checkpoint_paths=(CHECKPOINTS / "I3D_8x8_R50.pth",),
    ),
    ModelEntry(
        key="realforensics",
        label="RealForensics",
        repo_paths=(MODEL_REPOS / "RealForensics",),
        checkpoint_paths=(CHECKPOINTS / "realforensics_ff.pth",),
    ),
    ModelEntry(
        key="avoid_df",
        label="AVoiD-DF",
        repo_paths=(MODEL_REPOS / "AVoiD-DF",),
    ),
    ModelEntry(
        key="batfd",
        label="BA-TFD",
        repo_paths=(MODEL_REPOS / "LAV-DF",),
        checkpoint_paths=(CHECKPOINTS / "batfd_default.ckpt",),
    ),
    ModelEntry(
        key="late_fusion_baseline",
        label="Late-fusion baseline",
        repo_paths=(),
    ),
)

MODEL_INDEX = {entry.key: entry for entry in MODELS}


def repo_strings(paths: tuple[Path, ...]) -> list[str]:
    return [str(path.resolve()) for path in paths]


def file_strings(paths: tuple[Path, ...]) -> list[str]:
    return [str(path.resolve()) for path in paths]


def entry_base(entry: ModelEntry) -> dict:
    return {
        "key": entry.key,
        "label": entry.label,
        "repo_paths": repo_strings(entry.repo_paths),
        "checkpoint_paths": file_strings(entry.checkpoint_paths),
        "repo_paths_present": [path.exists() for path in entry.repo_paths],
        "checkpoint_paths_present": [path.exists() for path in entry.checkpoint_paths],
    }


def make_result(
    entry: ModelEntry,
    *,
    status: str,
    ready: bool,
    notes: str,
    used_repo: Path | None = None,
    used_checkpoint: Path | None = None,
    load_details: dict | None = None,
    error: str | None = None,
) -> dict:
    result = entry_base(entry)
    result.update(
        {
            "status": status,
            "ready": ready,
            "notes": notes,
            "used_repo": str(used_repo.resolve()) if used_repo else None,
            "used_checkpoint": str(used_checkpoint.resolve()) if used_checkpoint else None,
            "load_details": load_details or {},
            "error": error,
        }
    )
    return result


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


@contextlib.contextmanager
def prepend_sys_path(*paths: Path):
    originals = list(sys.path)
    for path in reversed(paths):
        sys.path.insert(0, str(path.resolve()))
    try:
        yield
    finally:
        sys.path[:] = originals


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def unpack_load_result(result) -> tuple[list[str], list[str]]:
    if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
        return list(result.missing_keys), list(result.unexpected_keys)
    missing, unexpected = result
    return list(missing), list(unexpected)


def ensure_clean_load(entry: ModelEntry, missing: list[str], unexpected: list[str], *, used_repo: Path, used_checkpoint: Path, notes: str) -> dict:
    if missing or unexpected:
        return make_result(
            entry,
            status="load_error",
            ready=False,
            notes=notes,
            used_repo=used_repo,
            used_checkpoint=used_checkpoint,
            load_details={"missing_keys": missing, "unexpected_keys": unexpected},
            error="state_dict mismatch",
        )
    return make_result(
        entry,
        status="ready",
        ready=True,
        notes=notes,
        used_repo=used_repo,
        used_checkpoint=used_checkpoint,
        load_details={"missing_keys": [], "unexpected_keys": []},
    )


def resolve_yaml_reference(path: Path) -> Path:
    text = path.read_text(encoding="utf-8").strip()
    if "\n" not in text and text.endswith(".yaml") and text != path.name:
        return (path.parent / text).resolve()
    return path


def install_legacy_fft_shim() -> None:
    import torch

    if hasattr(torch, "rfft") and hasattr(torch, "irfft"):
        return

    def rfft(input_tensor, signal_ndim, normalized=False, onesided=True):
        if signal_ndim != 1:
            raise NotImplementedError("Only 1D legacy torch.rfft is needed here.")
        norm = "ortho" if normalized else "backward"
        if onesided:
            output = torch.fft.rfft(input_tensor, dim=-1, norm=norm)
        else:
            output = torch.fft.fft(input_tensor, dim=-1, norm=norm)
        return torch.view_as_real(output)

    def irfft(input_tensor, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        if signal_ndim != 1:
            raise NotImplementedError("Only 1D legacy torch.irfft is needed here.")
        norm = "ortho" if normalized else "backward"
        complex_input = torch.view_as_complex(input_tensor.contiguous())
        n_value = signal_sizes[0] if signal_sizes else None
        if onesided:
            return torch.fft.irfft(complex_input, n=n_value, dim=-1, norm=norm)
        return torch.fft.ifft(complex_input, n=n_value, dim=-1, norm=norm).real

    torch.rfft = rfft
    torch.irfft = irfft


def validate_xceptionnet() -> dict:
    import torch

    entry = MODEL_INDEX["xceptionnet"]
    repo = MODEL_REPOS / "DeepfakeBench" / "training"
    ckpt = CHECKPOINTS / "xception_best.pth"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from networks.xception import Xception

        model = Xception({"mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
        state_dict = torch.load(ckpt, map_location="cpu")
        backbone_state = {key[len("backbone."):]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        missing, unexpected = unpack_load_result(model.load_state_dict(backbone_state, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the DeepfakeBench Xception backbone checkpoint on CPU.",
    )


def validate_f3net() -> dict:
    import numpy as np
    import torch
    import torch.nn as nn

    entry = MODEL_INDEX["f3net"]
    repo = MODEL_REPOS / "DeepfakeBench" / "training"
    ckpt = CHECKPOINTS / "f3net_best.pth"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from networks.xception import Xception

        def dct_mat(size: int) -> list[list[float]]:
            return [
                [
                    (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
                    * np.cos((j + 0.5) * np.pi * i / size)
                    for j in range(size)
                ]
                for i in range(size)
            ]

        def generate_filter(start: int, end: int, size: int) -> list[list[float]]:
            return [[0.0 if i + j > end or i + j < start else 1.0 for j in range(size)] for i in range(size)]

        def norm_sigma(x):
            return 2.0 * torch.sigmoid(x) - 1.0

        class Filter(nn.Module):
            def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
                super().__init__()
                self.use_learnable = use_learnable
                self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
                if self.use_learnable:
                    self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
                    self.learnable.data.normal_(0.0, 0.1)
                self.norm = norm
                if norm:
                    self.ft_num = nn.Parameter(
                        torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False
                    )

            def forward(self, x):
                filt = self.base + norm_sigma(self.learnable) if self.use_learnable else self.base
                return x * filt / self.ft_num if self.norm else x * filt

        class FADHead(nn.Module):
            def __init__(self, size):
                super().__init__()
                self._DCT_all = nn.Parameter(torch.tensor(dct_mat(size)).float(), requires_grad=False)
                self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(dct_mat(size)).float(), 0, 1), requires_grad=False)
                self.filters = nn.ModuleList(
                    [
                        Filter(size, 0, size // 2.82),
                        Filter(size, size // 2.82, size // 2),
                        Filter(size, size // 2, size * 2),
                        Filter(size, 0, size * 2),
                    ]
                )

            def forward(self, x):
                x_freq = self._DCT_all @ x @ self._DCT_all_T
                outputs = []
                for current_filter in self.filters:
                    x_pass = current_filter(x_freq)
                    outputs.append(self._DCT_all_T @ x_pass @ self._DCT_all)
                return torch.cat(outputs, dim=1)

        class F3NetWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Xception({"mode": "original", "num_classes": 2, "inc": 12, "dropout": 0.5})
                self.FAD_head = FADHead(256)

        model = F3NetWrapper()
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the public DeepfakeBench F3-Net checkpoint with a minimal FAD branch wrapper on CPU.",
    )


def validate_universalfakedetect() -> dict:
    import torch

    entry = MODEL_INDEX["universalfakedetect"]
    repo = MODEL_REPOS / "UniversalFakeDetect"
    fc_ckpt = repo / "pretrained_weights" / "fc_weights.pth"
    require_exists(repo)
    require_exists(fc_ckpt)
    with prepend_sys_path(repo):
        from models.clip_models import CLIPModel

        model = CLIPModel("ViT-L/14", num_classes=1)
        state_dict = torch.load(fc_ckpt, map_location="cpu")
        missing, unexpected = unpack_load_result(model.fc.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=fc_ckpt,
        notes="Loaded the CLIP ViT-L/14 backbone plus the public fc head weights on CPU.",
    )


def validate_mat() -> dict:
    import torch

    entry = MODEL_INDEX["mat"]
    repo = MODEL_REPOS / "multiple-attention"
    ckpt = CHECKPOINTS / "mat_bundle" / "multi-attention" / "pretrained" / "ff_c23.pth"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from models.MAT import MAT

        model = MAT(net="efficientnet-b4", attention_layer="b5", feature_layer="b2", size=(380, 380), M=4)
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the official Multi-Attention FF++ c23 checkpoint with the matching EfficientNet-B4 config.",
    )


def validate_sbi() -> dict:
    import torch

    entry = MODEL_INDEX["sbi"]
    repo = MODEL_REPOS / "SelfBlendedImages"
    ckpt = CHECKPOINTS / "sbi_c23.tar"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo / "src"):
        from model import Detector

        model = Detector()
        state_dict = torch.load(ckpt, map_location="cpu")["model"]
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the SelfBlendedImages detector checkpoint on CPU.",
    )


def validate_aasist() -> dict:
    import json as json_lib
    import torch

    entry = MODEL_INDEX["aasist"]
    repo = MODEL_REPOS / "aasist"
    ckpt = repo / "models" / "weights" / "AASIST.pth"
    config_path = repo / "config" / "AASIST.conf"
    require_exists(repo)
    require_exists(ckpt)
    require_exists(config_path)
    with prepend_sys_path(repo):
        from models.AASIST import Model

        config = json_lib.loads(config_path.read_text(encoding="utf-8"))
        model = Model(config["model_config"])
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the public AASIST checkpoint on CPU.",
    )


def validate_rawnet2() -> dict:
    import torch
    import yaml

    entry = MODEL_INDEX["rawnet2"]
    repo = MODEL_REPOS / "2021" / "LA" / "Baseline-RawNet2"
    config_path = resolve_yaml_reference(repo / "model_config_RawNet.yaml")
    ckpt = CHECKPOINTS / "rawnet2_pretrained" / "pre_trained_DF_RawNet2.pth"
    require_exists(repo)
    require_exists(config_path)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from model import RawNet

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = RawNet(config["model"], device="cpu")
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the ASVspoof RawNet2 checkpoint on CPU.",
    )


def validate_lcnn() -> dict:
    import torch
    from types import SimpleNamespace

    entry = MODEL_INDEX["lcnn"]
    repo = MODEL_REPOS / "2021" / "LA" / "Baseline-LFCC-LCNN"
    project_dir = repo / "project" / "baseline_LA"
    ckpt = CHECKPOINTS / "lcnn_pretrained" / "la_trained_network.pt"
    require_exists(repo)
    require_exists(project_dir)
    require_exists(ckpt)
    install_legacy_fft_shim()
    with prepend_sys_path(repo, project_dir):
        from model import Model

        prj_conf = SimpleNamespace(optional_argument=["protocol_missing.txt"], wav_samp_rate=16000)
        model = Model(1, 1, None, prj_conf, mean_std=None)
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the LCNN checkpoint after adding a compatibility shim for legacy torch.rfft/irfft APIs.",
    )


def validate_wav2vec2_classifier() -> dict:
    import torch

    entry = MODEL_INDEX["wav2vec2_classifier"]
    repo = MODEL_REPOS / "SSL_Anti-spoofing"
    fairseq_dir = next(repo.glob("fairseq-*"), None)
    ckpt = CHECKPOINTS / "ssl_antispoofing" / "LA_model.pth"
    xlsr_ckpt = CHECKPOINTS / "xlsr2_300m.pt"
    require_exists(repo)
    require_exists(ckpt)
    require_exists(xlsr_ckpt)
    if fairseq_dir is None:
        return make_result(
            entry,
            status="environment_incompatible",
            ready=False,
            notes="The local fairseq source tree is missing, so the SSL model cannot be imported.",
            used_repo=repo,
            used_checkpoint=ckpt,
            error="missing fairseq source tree",
        )

    with prepend_sys_path(fairseq_dir, repo), pushd(CHECKPOINTS):
        try:
            from model import Model

            model = Model(None, device="cpu")
            state_dict = torch.load(ckpt, map_location="cpu")
            missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
            return ensure_clean_load(
                entry,
                missing,
                unexpected,
                used_repo=repo,
                used_checkpoint=ckpt,
                notes="Loaded the SSL_Anti-spoofing Wav2Vec2-style model on CPU.",
            )
        except Exception as exc:
            return make_result(
                entry,
                status="environment_incompatible",
                ready=False,
                notes="The public checkpoint is present, but the bundled fairseq code is incompatible with Python 3.12 in this environment.",
                used_repo=repo,
                used_checkpoint=ckpt,
                error=f"{type(exc).__name__}: {exc}",
            )


def validate_lipforensics() -> dict:
    import json as json_lib
    import torch

    entry = MODEL_INDEX["lipforensics"]
    repo = MODEL_REPOS / "LipForensics"
    ckpt = CHECKPOINTS / "lipforensics_ff.pth"
    config_path = repo / "models" / "configs" / "lrw_resnet18_mstcn.json"
    require_exists(repo)
    require_exists(ckpt)
    require_exists(config_path)
    with prepend_sys_path(repo):
        from models.spatiotemporal_net import Lipreading

        config = json_lib.loads(config_path.read_text(encoding="utf-8"))
        tcn_options = {
            "num_layers": config["tcn_num_layers"],
            "kernel_size": config["tcn_kernel_size"],
            "dropout": config["tcn_dropout"],
            "dwpw": config["tcn_dwpw"],
            "width_mult": config["tcn_width_mult"],
        }
        model = Lipreading(num_classes=1, tcn_options=tcn_options, relu_type=config["relu_type"])
        state_dict = torch.load(ckpt, map_location="cpu")["model"]
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the LipForensics FF++ checkpoint on CPU.",
    )


def validate_ftcn() -> dict:
    import importlib

    entry = MODEL_INDEX["ftcn"]
    repo = MODEL_REPOS / "FTCN"
    ckpt = CHECKPOINTS / "ftcn_tt.pth"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from config import config as cfg
        from config import finalize_configs

        cfg.init_with_yaml()
        cfg.update_with_yaml("ftcn_tt.yaml")
        finalize_configs(cfg, freeze=True, verbose=False)

        module = importlib.import_module("model.classifier.i3d_temporal_var_fix_dropout_tt_cfg")
        module.parameters = [parameter for parameter in module.parameters if parameter not in ("device", "dtype")]
        classifier = module.Classifier()
        loaded_ok, loaded_epoch = classifier.load(fullpath=str(ckpt))
        if not loaded_ok:
            return make_result(
                entry,
                status="load_error",
                ready=False,
                notes="The FTCN classifier class built successfully, but the checkpoint loader returned False.",
                used_repo=repo,
                used_checkpoint=ckpt,
                error="classifier.load returned False",
            )
        return make_result(
            entry,
            status="ready",
            ready=True,
            notes="Loaded the official FTCN+TT checkpoint on CPU after filtering the deprecated Conv3d device/dtype signature args.",
            used_repo=repo,
            used_checkpoint=ckpt,
            load_details={"loaded_epoch": loaded_epoch},
        )


def validate_altfreezing() -> dict:
    entry = MODEL_INDEX["altfreezing"]
    repo = MODEL_REPOS / "AltFreezing"
    shared_backbone = CHECKPOINTS / "I3D_8x8_R50.pth"
    require_exists(repo)
    notes = "The repo is cloned and the shared public I3D backbone is present, but no public AltFreezing detector checkpoint was available locally."
    if shared_backbone.exists():
        notes += " DeepfakeBench only exposes the shared 3D backbone, not the final AltFreezing weights."
    return make_result(
        entry,
        status="missing_checkpoint",
        ready=False,
        notes=notes,
        used_repo=repo,
        used_checkpoint=shared_backbone if shared_backbone.exists() else None,
        error="final AltFreezing checkpoint not publicly retrieved",
    )


def validate_realforensics() -> dict:
    import torch
    from hydra import compose, initialize_config_dir

    entry = MODEL_INDEX["realforensics"]
    repo = MODEL_REPOS / "RealForensics"
    ckpt = CHECKPOINTS / "realforensics_ff.pth"
    conf_dir = repo / "stage2" / "conf"
    require_exists(repo)
    require_exists(conf_dir)
    require_exists(ckpt)
    with prepend_sys_path(repo, repo / "stage2"):
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name="config_combined")

        from stage2.models.model_combined import ModelCombined

        model = ModelCombined(cfg)
        state_dict = torch.load(ckpt, map_location="cpu")
        backbone_state = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith("backbone")}
        df_head_state = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith("df_head")}
        backbone_missing, backbone_unexpected = unpack_load_result(model.backbone.load_state_dict(backbone_state, strict=False))
        head_missing, head_unexpected = unpack_load_result(model.df_head.load_state_dict(df_head_state, strict=False))
    missing = backbone_missing + head_missing
    unexpected = backbone_unexpected + head_unexpected
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the RealForensics stage-2 visual backbone and deepfake head on CPU.",
    )


def validate_avoid_df() -> dict:
    entry = MODEL_INDEX["avoid_df"]
    repo = MODEL_REPOS / "AVoiD-DF"
    require_exists(repo)
    return make_result(
        entry,
        status="restricted",
        ready=False,
        notes="The AVoiD-DF repo itself says the datasets and weights are not publicly available because of protocol restrictions.",
        used_repo=repo,
        error="weights not publicly released",
    )


def validate_batfd() -> dict:
    import torch

    entry = MODEL_INDEX["batfd"]
    repo = MODEL_REPOS / "LAV-DF"
    ckpt = CHECKPOINTS / "batfd_default.ckpt"
    require_exists(repo)
    require_exists(ckpt)
    with prepend_sys_path(repo):
        from model.batfd import Batfd

        model = Batfd()
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = unpack_load_result(model.load_state_dict(state_dict, strict=False))
    return ensure_clean_load(
        entry,
        missing,
        unexpected,
        used_repo=repo,
        used_checkpoint=ckpt,
        notes="Loaded the public BA-TFD checkpoint on CPU.",
    )


def validate_late_fusion_baseline() -> dict:
    import torch
    import torch.nn as nn

    entry = MODEL_INDEX["late_fusion_baseline"]

    class LateFusionBaseline(nn.Module):
        def forward(self, audio_score, video_score):
            return (audio_score + video_score) / 2.0

    model = LateFusionBaseline()
    _ = model(torch.tensor([0.25]), torch.tensor([0.75]))
    return make_result(
        entry,
        status="ready",
        ready=True,
        notes="Constructed and executed a minimal score-level averaging baseline.",
        load_details={"implementation": "average(audio_score, video_score)"},
    )


VALIDATORS: dict[str, Callable[[], dict]] = {
    "xceptionnet": validate_xceptionnet,
    "f3net": validate_f3net,
    "universalfakedetect": validate_universalfakedetect,
    "mat": validate_mat,
    "sbi": validate_sbi,
    "aasist": validate_aasist,
    "rawnet2": validate_rawnet2,
    "lcnn": validate_lcnn,
    "wav2vec2_classifier": validate_wav2vec2_classifier,
    "lipforensics": validate_lipforensics,
    "ftcn": validate_ftcn,
    "altfreezing": validate_altfreezing,
    "realforensics": validate_realforensics,
    "avoid_df": validate_avoid_df,
    "batfd": validate_batfd,
    "late_fusion_baseline": validate_late_fusion_baseline,
}


def run_single(key: str) -> dict:
    entry = MODEL_INDEX[key]
    validator = VALIDATORS[key]
    try:
        result = validator()
    except Exception as exc:
        result = make_result(
            entry,
            status="load_error",
            ready=False,
            notes="The validator itself raised an exception before a clean load decision could be made.",
            error=f"{type(exc).__name__}: {exc}",
            load_details={"traceback": traceback.format_exc()},
        )
    result["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def run_all() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    log(f"Starting full 16-model validation sweep into {REPORT_DIR}.")
    for entry in progress_iter(MODELS, total=len(MODELS), desc="[validate-16] models", unit="model"):
        log(f"Launching isolated validator for '{entry.label}' ({entry.key}).")
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--single", entry.key],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        parsed = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(JSON_SENTINEL):
                parsed = json.loads(line[len(JSON_SENTINEL):])
                break
        if parsed is None:
            parsed = make_result(
                entry,
                status="load_error",
                ready=False,
                notes="The child validator did not emit a parseable JSON result.",
                error=f"exit_code={completed.returncode}",
                load_details={
                    "stdout_tail": completed.stdout.splitlines()[-20:],
                    "stderr_tail": completed.stderr.splitlines()[-20:],
                },
            )
        parsed["subprocess_returncode"] = completed.returncode
        results.append(parsed)
        log(
            f"Finished '{entry.label}' with status={parsed['status']} "
            f"and ready={parsed['ready']}."
        )

    ready = [result for result in results if result["ready"]]
    not_ready = [result for result in results if not result["ready"]]
    summary_lines = [
        f"Validated {len(results)} models",
        f"Ready: {len(ready)}",
        f"Not ready: {len(not_ready)}",
        "",
        "Ready models:",
    ]
    summary_lines.extend(f"- {result['label']}" for result in ready)
    summary_lines.append("")
    summary_lines.append("Not ready models:")
    summary_lines.extend(f"- {result['label']}: {result['status']} ({result['notes']})" for result in not_ready)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "results": results,
        "summary": {
            "total": len(results),
            "ready": len(ready),
            "not_ready": len(not_ready),
            "ready_labels": [result["label"] for result in ready],
            "not_ready_labels": [result["label"] for result in not_ready],
        },
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the 16 models referenced in Project-Manager-planning.html.")
    parser.add_argument("--single", choices=sorted(MODEL_INDEX), help="Run exactly one validator in isolation.")
    args = parser.parse_args()

    if args.single:
        log(f"Running isolated validator for '{args.single}'.")
        result = run_single(args.single)
        print(f"{JSON_SENTINEL}{json.dumps(result, sort_keys=True)}")
        return 0

    log_config(
        "Validation environment",
        {
            "workspace_root": ROOT,
            "report_dir": REPORT_DIR,
            "python_executable": sys.executable,
        },
    )
    report = run_all()
    summary = report["summary"]
    print(f"Validated {summary['total']} models")
    print(f"Ready: {summary['ready']}")
    print(f"Not ready: {summary['not_ready']}")
    print(f"Report: {REPORT_DIR / 'report.json'}")
    print(f"Summary: {REPORT_DIR / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
