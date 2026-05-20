from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .assets import AssetLayout
from .manifests import ManifestRecord
from .media import (
    FaceCropper,
    crop_face_sequence,
    grayscale_clip,
    pad_or_trim_1d,
    read_audio_segment,
    read_frame,
    read_video_clip,
    resize_frame,
)
from .progress import log


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextlib.contextmanager
def prepend_sys_path(*paths: Path):
    original = list(sys.path)
    for path in reversed(paths):
        sys.path.insert(0, str(path.resolve()))
    try:
        yield
    finally:
        sys.path[:] = original


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _module_is_from_roots(module, roots: tuple[Path, ...]) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file:
        try:
            file_path = Path(module_file).resolve()
            if any(file_path.is_relative_to(root) for root in roots):
                return True
        except Exception:
            pass
    try:
        module_paths = getattr(module, "__path__", None)
    except Exception:
        module_paths = None
    if module_paths is not None:
        if isinstance(module_paths, (str, os.PathLike)):
            module_paths_iter = [module_paths]
        else:
            try:
                module_paths_iter = list(module_paths)
            except Exception:
                module_paths_iter = []
        for module_path in module_paths_iter:
            if module_path is None:
                continue
            try:
                path = Path(module_path).resolve()
                if any(path.is_relative_to(root) for root in roots):
                    return True
            except Exception:
                continue
    return False


@contextlib.contextmanager
def isolated_repo_import(*paths: Path, prefixes: Iterable[str] = ()) -> Iterator[None]:
    roots = tuple(path.resolve() for path in paths if path.exists())
    prefixes = tuple(prefixes)
    original_modules = dict(sys.modules)
    for name, module in list(sys.modules.items()):
        if _matches_prefix(name, prefixes) or _module_is_from_roots(module, roots):
            sys.modules.pop(name, None)
    with prepend_sys_path(*paths):
        try:
            yield
        finally:
            for name, module in list(sys.modules.items()):
                if _matches_prefix(name, prefixes) or _module_is_from_roots(module, roots):
                    sys.modules.pop(name, None)
            for name, module in original_modules.items():
                if name not in sys.modules:
                    sys.modules[name] = module


def _resolve_yaml_reference(path: Path) -> Path:
    text = path.read_text(encoding="utf-8").strip()
    if "\n" not in text and text.endswith(".yaml") and text != path.name:
        return (path.parent / text).resolve()
    return path


def _install_legacy_fft_shim() -> None:
    if hasattr(torch, "rfft") and hasattr(torch, "irfft"):
        return

    def rfft(input_tensor, signal_ndim, normalized=False, onesided=True):
        if signal_ndim != 1:
            raise NotImplementedError("Only 1D legacy torch.rfft is supported here.")
        norm = "ortho" if normalized else "backward"
        output = torch.fft.rfft(input_tensor, dim=-1, norm=norm) if onesided else torch.fft.fft(input_tensor, dim=-1, norm=norm)
        return torch.view_as_real(output)

    def irfft(input_tensor, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        if signal_ndim != 1:
            raise NotImplementedError("Only 1D legacy torch.irfft is supported here.")
        norm = "ortho" if normalized else "backward"
        complex_input = torch.view_as_complex(input_tensor.contiguous())
        n_value = signal_sizes[0] if signal_sizes else None
        if onesided:
            return torch.fft.irfft(complex_input, n=n_value, dim=-1, norm=norm)
        return torch.fft.ifft(complex_input, n=n_value, dim=-1, norm=norm).real

    torch.rfft = rfft  # type: ignore[attr-defined]
    torch.irfft = irfft  # type: ignore[attr-defined]


def _install_legacy_stft_shim() -> None:
    current_stft = getattr(torch, "stft", None)
    if current_stft is None or getattr(current_stft, "_deepshield_legacy_wrapper", False):
        return

    def legacy_stft(input_tensor, *args, **kwargs):
        # Older upstream audio repos expect the pre-2.x default real-valued STFT
        # output shape when they omit return_complex for real inputs.
        if "return_complex" not in kwargs and not torch.is_complex(input_tensor):
            kwargs["return_complex"] = False
        return current_stft(input_tensor, *args, **kwargs)

    legacy_stft._deepshield_legacy_wrapper = True  # type: ignore[attr-defined]
    torch.stft = legacy_stft  # type: ignore[assignment]


def _normalize(image: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor


def _logit_from_prob(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1 - 1e-6)
    return torch.logit(prob)


def _fake_logit_from_two_class(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[1] == 1:
        return logits.squeeze(1)
    return logits[:, 1] - logits[:, 0]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str


ACTIVE_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("xceptionnet", "image"),
    ModelSpec("f3net", "image"),
    ModelSpec("universalfakedetect", "image"),
    ModelSpec("mat", "image"),
    ModelSpec("sbi", "image"),
    ModelSpec("aasist", "audio"),
    ModelSpec("rawnet2", "audio"),
    ModelSpec("lcnn", "audio"),
    ModelSpec("lipforensics", "video"),
    ModelSpec("ftcn", "video"),
    ModelSpec("realforensics", "video"),
    ModelSpec("batfd", "multimodal"),
    ModelSpec("late_fusion_baseline", "multimodal"),
)


def list_active_models(task: str | None = None) -> list[str]:
    return [spec.name for spec in ACTIVE_MODEL_SPECS if task is None or spec.task == task]


class BaseModelAdapter:
    name: str = ""
    task: str = ""
    disable_cudnn_for_input_grads: bool = False

    def __init__(self, assets: AssetLayout, *, device: str = "cpu") -> None:
        self.assets = assets
        self.device = torch.device(device)
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self) -> nn.Module:
        raise NotImplementedError

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        raise NotImplementedError

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        raise NotImplementedError

    def forward_scores(self, inputs) -> torch.Tensor:
        return torch.sigmoid(self.forward_attack_logits(inputs))

    def attack_autograd_context(self):
        if self.disable_cudnn_for_input_grads and self.device.type == "cuda":
            return torch.backends.cudnn.flags(enabled=False)
        return contextlib.nullcontext()


class XceptionInfer(nn.Module):
    def __init__(self, repo_root: Path, checkpoint: Path):
        super().__init__()
        with isolated_repo_import(repo_root / "training", prefixes=("networks",)):
            from networks.xception import Xception

            self.backbone = Xception({"mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
        state_dict = torch.load(checkpoint, map_location="cpu")
        backbone_state = {key[len("backbone."):]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(image)
        return self.backbone.classifier(features)


class FADFilter(nn.Module):
    def __init__(self, size: int, band_start: int, band_end: int, use_learnable: bool = True, norm: bool = False):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(self._generate_filter(band_start, band_end, size), dtype=torch.float32), requires_grad=False)
        self.use_learnable = use_learnable
        if use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size) * 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(self._generate_filter(band_start, band_end, size), dtype=torch.float32)), requires_grad=False)

    @staticmethod
    def _generate_filter(start: int, end: int, size: int) -> list[list[float]]:
        return [[0.0 if i + j > end or i + j < start else 1.0 for j in range(size)] for i in range(size)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = self.base
        if self.use_learnable:
            filt = filt + (2.0 * torch.sigmoid(self.learnable) - 1.0)
        return x * filt / self.ft_num if self.norm else x * filt


class FADHead(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        dct = self._dct_matrix(size)
        self._dct_all = nn.Parameter(torch.tensor(dct, dtype=torch.float32), requires_grad=False)
        self._dct_all_t = nn.Parameter(torch.tensor(dct, dtype=torch.float32).t(), requires_grad=False)
        self.filters = nn.ModuleList(
            [
                FADFilter(size, 0, size // 2.82),
                FADFilter(size, size // 2.82, size // 2),
                FADFilter(size, size // 2, size * 2),
                FADFilter(size, 0, size * 2),
            ]
        )

    @staticmethod
    def _dct_matrix(size: int) -> list[list[float]]:
        import math

        return [
            [
                (math.sqrt(1.0 / size) if i == 0 else math.sqrt(2.0 / size)) * math.cos((j + 0.5) * math.pi * i / size)
                for j in range(size)
            ]
            for i in range(size)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = self._dct_all @ x @ self._dct_all_t
        outputs = []
        for current_filter in self.filters:
            x_pass = current_filter(x_freq)
            outputs.append(self._dct_all_t @ x_pass @ self._dct_all)
        return torch.cat(outputs, dim=1)


class F3NetInfer(nn.Module):
    def __init__(self, repo_root: Path, checkpoint: Path, image_size: int = 256):
        super().__init__()
        with isolated_repo_import(repo_root / "training", prefixes=("networks",)):
            from networks.xception import Xception

            self.backbone = Xception({"mode": "original", "num_classes": 2, "inc": 12, "dropout": 0.5})
        self.fad_head = FADHead(image_size)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(self.fad_head(image))
        return self.backbone.classifier(features)


class ImageAdapter(BaseModelAdapter):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_mode = "full_face"
    size = 256

    def __init__(self, assets: AssetLayout, *, device: str = "cpu") -> None:
        self.cropper = FaceCropper(device=device)
        super().__init__(assets, device=device)

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        images = []
        labels = []
        for record in records:
            frame = read_frame(record, data_root)
            crop = self.cropper.crop_frame(frame, mode=self.crop_mode)
            crop = resize_frame(crop, self.size)
            images.append(crop)
            labels.append(record.label)
        return torch.stack(images, dim=0).to(self.device), torch.tensor(labels, dtype=torch.float32, device=self.device)

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        normalized = torch.stack([_normalize(image, self.mean, self.std) for image in inputs], dim=0)
        return _fake_logit_from_two_class(self.model(normalized))


class XceptionAdapter(ImageAdapter):
    name = "xceptionnet"
    task = "image"
    size = 256

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "DeepfakeBench"
        checkpoint = self.assets.checkpoints / "xception_best.pth"
        return XceptionInfer(repo, checkpoint)


class F3NetAdapter(ImageAdapter):
    name = "f3net"
    task = "image"
    size = 256

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "DeepfakeBench"
        checkpoint = self.assets.checkpoints / "f3net_best.pth"
        return F3NetInfer(repo, checkpoint, image_size=self.size)


class UniversalFakeDetectAdapter(ImageAdapter):
    name = "universalfakedetect"
    task = "image"
    size = 224
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "UniversalFakeDetect"
        with isolated_repo_import(repo, prefixes=("models",)):
            from models.clip_models import CLIPModel

            model = CLIPModel("ViT-L/14", num_classes=1)
        state_dict = torch.load(repo / "pretrained_weights" / "fc_weights.pth", map_location="cpu")
        model.fc.load_state_dict(state_dict, strict=False)
        return model


class MATAdapter(ImageAdapter):
    name = "mat"
    task = "image"
    size = 380

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "multiple-attention"
        checkpoint = self.assets.checkpoints / "mat_bundle" / "multi-attention" / "pretrained" / "ff_c23.pth"
        with isolated_repo_import(repo, prefixes=("models",)):
            from models.MAT import MAT

            model = MAT(net="efficientnet-b4", attention_layer="b5", feature_layer="b2", size=(380, 380), M=4)
        state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model


class SBIAdapter(ImageAdapter):
    name = "sbi"
    task = "image"
    size = 256

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "SelfBlendedImages"
        checkpoint = self.assets.checkpoints / "sbi_c23.tar"
        with isolated_repo_import(repo / "src", prefixes=("model",)):
            from model import Detector

            model = Detector()
        state_dict = torch.load(checkpoint, map_location="cpu")["model"]
        model.load_state_dict(state_dict, strict=False)
        return model


class AudioAdapter(BaseModelAdapter):
    sample_rate = 16000
    sample_length = 64600

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        waveforms = []
        labels = []
        for record in records:
            try:
                waveform = read_audio_segment(record, data_root, target_rate=self.sample_rate)
            except Exception as exc:
                log(
                    f"[audio] Skipping unreadable sample '{record.sample_id}' "
                    f"({record.file}): {exc}"
                )
                continue
            waveform = pad_or_trim_1d(waveform, self.sample_length)
            waveforms.append(waveform)
            labels.append(record.label)
        if not waveforms:
            empty_waveforms = torch.empty((0, self.sample_length), dtype=torch.float32, device=self.device)
            empty_labels = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_waveforms, empty_labels
        return torch.stack(waveforms, dim=0).to(self.device), torch.tensor(labels, dtype=torch.float32, device=self.device)


class AASISTAdapter(AudioAdapter):
    name = "aasist"
    task = "audio"

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "aasist"
        checkpoint = repo / "models" / "weights" / "AASIST.pth"
        config = json.loads((repo / "config" / "AASIST.conf").read_text(encoding="utf-8"))
        with isolated_repo_import(repo, prefixes=("models",)):
            from models.AASIST import Model

            model = Model(config["model_config"])
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        _, logits = self.model(inputs)
        return _fake_logit_from_two_class(logits)


class RawNet2Adapter(AudioAdapter):
    name = "rawnet2"
    task = "audio"
    disable_cudnn_for_input_grads = True

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "2021" / "LA" / "Baseline-RawNet2"
        config_path = _resolve_yaml_reference(repo / "model_config_RawNet.yaml")
        checkpoint = self.assets.checkpoints / "rawnet2_pretrained" / "pre_trained_DF_RawNet2.pth"
        with isolated_repo_import(repo, prefixes=("model",)):
            from model import RawNet

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model = RawNet(config["model"], device="cpu")
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        if hasattr(self.model, "Sinc_conv"):
            self.model.Sinc_conv.device = inputs.device
        if hasattr(self.model, "device"):
            self.model.device = str(inputs.device)
        logits = self.model(inputs)
        return _fake_logit_from_two_class(logits)


class LCNNAdapter(AudioAdapter):
    name = "lcnn"
    task = "audio"
    disable_cudnn_for_input_grads = True

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "2021" / "LA" / "Baseline-LFCC-LCNN"
        project_dir = repo / "project" / "baseline_LA"
        checkpoint = self.assets.checkpoints / "lcnn_pretrained" / "la_trained_network.pt"
        _install_legacy_fft_shim()
        _install_legacy_stft_shim()
        with isolated_repo_import(repo, project_dir, prefixes=("model",)):
            from model import Model

            prj_conf = SimpleNamespace(optional_argument=["protocol_missing.txt"], wav_samp_rate=16000)
            model = Model(1, 1, None, prj_conf, mean_std=None)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        waveforms, labels = super().prepare_batch(records, data_root)
        return waveforms.unsqueeze(-1), labels

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        datalength = [inputs.shape[1]] * inputs.shape[0]
        feature_vec = self.model._compute_embedding(inputs, datalength)
        return feature_vec.squeeze(1)


class VideoAdapter(BaseModelAdapter):
    target_fps = 8.0
    num_frames = 32
    size = 224
    crop_mode = "full_face"

    def __init__(self, assets: AssetLayout, *, device: str = "cpu") -> None:
        self.cropper = FaceCropper(device=device)
        super().__init__(assets, device=device)

    def _prepare_clip(self, record: ManifestRecord, data_root: str | Path) -> torch.Tensor:
        clip = read_video_clip(record, data_root, target_fps=self.target_fps, target_frames=self.num_frames)
        clip = crop_face_sequence(clip, cropper=self.cropper, size=self.size, mode=self.crop_mode, num_frames=self.num_frames)
        return clip.permute(1, 0, 2, 3)

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        clips = [self._prepare_clip(record, data_root) for record in records]
        labels = torch.tensor([record.label for record in records], dtype=torch.float32, device=self.device)
        return torch.stack(clips, dim=0).to(self.device), labels


class LipForensicsAdapter(VideoAdapter):
    name = "lipforensics"
    task = "video"
    target_fps = 25.0
    num_frames = 25
    size = 88
    crop_mode = "mouth"

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "LipForensics"
        checkpoint = self.assets.checkpoints / "lipforensics_ff.pth"
        with isolated_repo_import(repo, prefixes=("models", "utils")), pushd(repo):
            from models.spatiotemporal_net import Lipreading, load_json

            args_loaded = load_json("./models/configs/lrw_resnet18_mstcn.json")
            tcn_options = {
                "num_layers": args_loaded["tcn_num_layers"],
                "kernel_size": args_loaded["tcn_kernel_size"],
                "dropout": args_loaded["tcn_dropout"],
                "dwpw": args_loaded["tcn_dwpw"],
                "width_mult": args_loaded["tcn_width_mult"],
            }
            model = Lipreading(num_classes=1, tcn_options=tcn_options, relu_type=args_loaded["relu_type"])
        state_dict = torch.load(checkpoint, map_location="cpu")["model"]
        model.load_state_dict(state_dict, strict=False)
        return model

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        clips = []
        labels = []
        for record in records:
            clip = read_video_clip(record, data_root, target_fps=self.target_fps, target_frames=self.num_frames)
            clip = crop_face_sequence(clip, cropper=self.cropper, size=self.size, mode=self.crop_mode, num_frames=self.num_frames)
            clip = grayscale_clip(clip)
            clips.append(clip.permute(1, 0, 2, 3))
            labels.append(record.label)
        return torch.stack(clips, dim=0).to(self.device), torch.tensor(labels, dtype=torch.float32, device=self.device)

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        normalized = (inputs - 0.421) / 0.165
        logits = self.model(normalized, lengths=[self.num_frames] * inputs.shape[0])
        return logits.squeeze(-1)


class FTCNAdapter(VideoAdapter):
    name = "ftcn"
    task = "video"
    target_fps = 8.0
    num_frames = 32
    size = 224

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "FTCN"
        checkpoint = self.assets.checkpoints / "ftcn_tt.pth"
        with isolated_repo_import(repo, prefixes=("config", "model")), pushd(repo):
            from config import config as cfg
            from config import finalize_configs

            cfg.init_with_yaml()
            cfg.update_with_yaml("ftcn_tt.yaml")
            finalize_configs(cfg, freeze=True, verbose=False)
            module = importlib.import_module("model.classifier.i3d_temporal_var_fix_dropout_tt_cfg")
            module.parameters = [parameter for parameter in module.parameters if parameter not in ("device", "dtype")]
            classifier = module.Classifier()
            classifier.load(fullpath=str(checkpoint))
        return classifier

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        clips = []
        labels = []
        for record in records:
            clips.append(self._prepare_clip(record, data_root))
            labels.append(record.label)
        return torch.stack(clips, dim=0).to(self.device), torch.tensor(labels, dtype=torch.float32, device=self.device)

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        clip = inputs.permute(0, 2, 1, 3, 4) * 255.0
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=clip.dtype, device=clip.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=clip.dtype, device=clip.device).view(1, 1, 3, 1, 1)
        clip = ((clip - mean) / std).permute(0, 2, 1, 3, 4)
        probabilities = self.model(clip)["final_output"].flatten()
        return _logit_from_prob(probabilities)


class RealForensicsAdapter(VideoAdapter):
    name = "realforensics"
    task = "video"
    target_fps = 25.0
    num_frames = 25
    size = 112

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "RealForensics"
        checkpoint = self.assets.checkpoints / "realforensics_ff.pth"
        with isolated_repo_import(repo, repo / "stage2", prefixes=("stage2",)):
            from hydra import compose, initialize_config_dir
            from stage2.models.model_combined import ModelCombined

            with initialize_config_dir(version_base=None, config_dir=str(repo / "stage2" / "conf")):
                cfg = compose(config_name="config_combined")
            model = ModelCombined(cfg)
        state_dict = torch.load(checkpoint, map_location="cpu")
        backbone_state = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith("backbone")}
        df_head_state = {".".join(key.split(".")[1:]): value for key, value in state_dict.items() if key.startswith("df_head")}
        model.backbone.load_state_dict(backbone_state, strict=False)
        model.df_head.load_state_dict(df_head_state, strict=False)
        return model

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        normalized = torch.stack(
            [
                torch.stack([_normalize(frame, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) for frame in clip.permute(1, 0, 2, 3)], dim=0)
                for clip in inputs
            ],
            dim=0,
        ).permute(0, 2, 1, 3, 4)
        features = self.model.backbone(normalized)
        return self.model.df_head(features).squeeze(-1)


class BATFDAdapter(BaseModelAdapter):
    name = "batfd"
    task = "multimodal"
    frame_padding = 512
    audio_padding = int(frame_padding / 25 * 16000)
    size = 96

    def _load_model(self) -> nn.Module:
        repo = self.assets.model_repos / "LAV-DF"
        checkpoint = self.assets.checkpoints / "batfd_default.ckpt"
        with isolated_repo_import(repo, prefixes=("model", "dataset", "utils", "metrics", "loss")):
            from model.batfd import Batfd

            model = Batfd()
        state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model

    def _prepare_video(self, record: ManifestRecord, data_root: str | Path) -> torch.Tensor:
        clip = read_video_clip(record, data_root, target_fps=25.0, target_frames=self.frame_padding)
        if clip.shape[0] >= self.frame_padding:
            indices = torch.linspace(0, clip.shape[0] - 1, self.frame_padding).round().long()
            clip = clip.index_select(0, indices)
        else:
            pad = clip[-1:].repeat(self.frame_padding - clip.shape[0], 1, 1, 1)
            clip = torch.cat([clip, pad], dim=0)
        clip = F.interpolate(clip, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return clip.permute(1, 0, 2, 3)

    def _prepare_audio(self, record: ManifestRecord, data_root: str | Path) -> torch.Tensor:
        waveform = read_audio_segment(record, data_root, target_rate=16000)
        return pad_or_trim_1d(waveform, self.audio_padding)

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        videos = []
        audios = []
        labels = []
        for record in records:
            videos.append(self._prepare_video(record, data_root))
            audios.append(self._prepare_audio(record, data_root))
            labels.append(record.label)
        inputs = {
            "video": torch.stack(videos, dim=0).to(self.device),
            "audio": torch.stack(audios, dim=0).to(self.device),
        }
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        return inputs, labels

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        try:
            import torchaudio
        except Exception as exc:
            raise RuntimeError("torchaudio is required for BA-TFD preprocessing.") from exc
        mel = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64).to(inputs["audio"].device)(inputs["audio"])
        mel = torch.log(mel + 0.01)
        if mel.shape[-1] >= 2048:
            mel = mel[:, :, :2048]
        else:
            mel = F.pad(mel, (0, 2048 - mel.shape[-1]))
        fusion_bm_map, _, _, _, _, _, _ = self.model(inputs["video"], mel)
        pooled = fusion_bm_map.amax(dim=(1, 2))
        return pooled


class LateFusionAdapter(BaseModelAdapter):
    name = "late_fusion_baseline"
    task = "multimodal"

    def _load_model(self) -> nn.Module:
        return nn.Identity()

    def __init__(self, assets: AssetLayout, *, device: str = "cpu") -> None:
        self.audio_adapter = AASISTAdapter(assets, device=device)
        self.video_adapter = FTCNAdapter(assets, device=device)
        super().__init__(assets, device=device)

    def prepare_batch(self, records: list[ManifestRecord], data_root: str | Path):
        audio_inputs, labels = self.audio_adapter.prepare_batch(records, data_root)
        video_inputs, _ = self.video_adapter.prepare_batch(records, data_root)
        return {"audio": audio_inputs, "video": video_inputs}, labels

    def forward_attack_logits(self, inputs) -> torch.Tensor:
        audio_scores = torch.sigmoid(self.audio_adapter.forward_attack_logits(inputs["audio"]))
        video_scores = torch.sigmoid(self.video_adapter.forward_attack_logits(inputs["video"]))
        return _logit_from_prob((audio_scores + video_scores) / 2.0)


ADAPTERS: dict[str, type[BaseModelAdapter]] = {
    "xceptionnet": XceptionAdapter,
    "f3net": F3NetAdapter,
    "universalfakedetect": UniversalFakeDetectAdapter,
    "mat": MATAdapter,
    "sbi": SBIAdapter,
    "aasist": AASISTAdapter,
    "rawnet2": RawNet2Adapter,
    "lcnn": LCNNAdapter,
    "lipforensics": LipForensicsAdapter,
    "ftcn": FTCNAdapter,
    "realforensics": RealForensicsAdapter,
    "batfd": BATFDAdapter,
    "late_fusion_baseline": LateFusionAdapter,
}


def build_adapter(name: str, assets: AssetLayout, *, device: str = "cpu") -> BaseModelAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown model adapter: {name}")
    return ADAPTERS[name](assets, device=device)
