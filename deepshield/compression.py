from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .media import pad_or_trim_1d, pil_to_tensor, tensor_to_pil


def jpeg_compress(image: torch.Tensor, *, quality: int = 75) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        path = Path(handle.name)
    try:
        tensor_to_pil(image).save(path, format="JPEG", quality=quality)
        return pil_to_tensor(Image.open(path).convert("RGB"))
    finally:
        path.unlink(missing_ok=True)


def social_media_resize(image: torch.Tensor, *, side: int = 512, quality: int = 75) -> torch.Tensor:
    pil = tensor_to_pil(image)
    pil = pil.resize((side, side))
    return jpeg_compress(pil_to_tensor(pil), quality=quality)


@dataclass
class PostprocessPlan:
    name: str
    kwargs: dict


def _ffmpeg_output(stream, output_path: str | Path, **kwargs):
    try:
        import ffmpeg
    except Exception as exc:
        raise RuntimeError("ffmpeg-python is required for video or audio re-encoding.") from exc
    stream = ffmpeg.output(stream, str(output_path), **kwargs)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)


def ffmpeg_reencode(input_path: str | Path, output_path: str | Path, *, codec: str, extra_args: list[str]) -> None:
    import ffmpeg

    stream = ffmpeg.input(str(input_path))
    output_kwargs = {}
    if codec.startswith("libx264"):
        output_kwargs["vcodec"] = codec
    if codec.startswith("libmp3"):
        output_kwargs["acodec"] = codec
    stream = ffmpeg.output(stream, str(output_path), **output_kwargs)
    ffmpeg.run(stream.global_args(*extra_args), overwrite_output=True, quiet=True)


def jpeg_batch(images: torch.Tensor, *, quality: int = 75) -> torch.Tensor:
    return torch.stack([jpeg_compress(image, quality=quality) for image in images], dim=0).to(images.device)


def social_media_resize_batch(images: torch.Tensor, *, side: int = 512, quality: int = 75) -> torch.Tensor:
    return torch.stack([social_media_resize(image, side=side, quality=quality) for image in images], dim=0).to(images.device)


def mp3_roundtrip_batch(waveforms: torch.Tensor, *, sample_rate: int = 16000, bitrate: str = "128k") -> torch.Tensor:
    try:
        import ffmpeg
        import torchaudio
    except Exception as exc:
        raise RuntimeError("ffmpeg-python and torchaudio are required for MP3 round-trip compression.") from exc

    restored = []
    for waveform in waveforms.detach().cpu():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "input.wav"
            mp3_path = tmpdir / "compressed.mp3"
            roundtrip_path = tmpdir / "roundtrip.wav"
            torchaudio.save(str(wav_path), waveform.unsqueeze(0), sample_rate)
            ffmpeg.run(
                ffmpeg.output(ffmpeg.input(str(wav_path)), str(mp3_path), acodec="libmp3lame", audio_bitrate=bitrate),
                overwrite_output=True,
                quiet=True,
            )
            ffmpeg.run(
                ffmpeg.output(ffmpeg.input(str(mp3_path)), str(roundtrip_path), acodec="pcm_s16le", ar=sample_rate, ac=1),
                overwrite_output=True,
                quiet=True,
            )
            restored_waveform, restored_rate = torchaudio.load(str(roundtrip_path))
            restored_waveform = restored_waveform.mean(dim=0)
            if restored_rate != sample_rate:
                restored_waveform = torchaudio.functional.resample(restored_waveform.unsqueeze(0), restored_rate, sample_rate).squeeze(0)
            restored.append(pad_or_trim_1d(restored_waveform, waveform.numel()))
    return torch.stack(restored, dim=0).to(waveforms.device)


POSTPROCESSORS = {
    "image": {
        "jpeg": jpeg_batch,
        "social_resize": social_media_resize_batch,
    },
    "audio": {
        "mp3": mp3_roundtrip_batch,
    },
}


def apply_postprocessor(task: str, inputs, plan: PostprocessPlan):
    if task not in POSTPROCESSORS:
        raise KeyError(f"No postprocessors registered for task '{task}'.")
    if plan.name not in POSTPROCESSORS[task]:
        raise KeyError(f"Unknown postprocessor '{plan.name}' for task '{task}'.")
    return POSTPROCESSORS[task][plan.name](inputs, **plan.kwargs)
