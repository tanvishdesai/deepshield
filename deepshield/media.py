from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from .manifests import ManifestRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


def _to_tchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got shape {tuple(video.shape)}")
    if video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)
    return video.contiguous()


@lru_cache(maxsize=256)
def _video_metadata(path: str) -> dict[str, float]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video file: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0:
        fps = 25.0
    return {
        "video_fps": fps,
        "video_frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _read_video_frames(path: str, frame_indices: list[int]) -> torch.Tensor:
    if not frame_indices:
        raise ValueError("frame_indices must contain at least one index.")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video file: {path}")

    frames_by_index: dict[int, torch.Tensor] = {}
    sorted_unique = sorted(set(int(index) for index in frame_indices))
    try:
        for frame_index in sorted_unique:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_by_index[frame_index] = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
    finally:
        capture.release()

    if not frames_by_index:
        raise RuntimeError(f"Could not decode requested frames from video file: {path}")

    first_available = next(iter(frames_by_index.values()))
    output_frames = []
    last_frame = first_available
    for frame_index in frame_indices:
        frame = frames_by_index.get(int(frame_index), last_frame)
        output_frames.append(frame)
        last_frame = frame
    return torch.stack(output_frames, dim=0)


def _sample_frame_indices(
    *,
    start: int,
    end: int,
    fps: float,
    target_fps: float,
    target_frames: int | None,
) -> list[int]:
    end = max(end, start + 1)
    available = end - start
    if target_frames is not None:
        if target_frames <= 1 or available <= 1:
            return [start]
        return torch.linspace(start, end - 1, target_frames).round().long().tolist()
    stride = max(int(round(fps / max(target_fps, 1e-6))), 1)
    indices = list(range(start, end, stride))
    return indices or [start]


@lru_cache(maxsize=2)
def _read_video_cached(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = _to_tchw(video)
    audio = audio.float()
    if audio.ndim == 2:
        audio = audio.permute(1, 0)
    return video, audio, info


@lru_cache(maxsize=128)
def _read_audio_cached(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    errors: list[str] = []
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError("torchaudio is required to read standalone audio files.") from exc
    try:
        waveform, sample_rate = torchaudio.load(path)
    except Exception as exc:
        errors.append(f"torchaudio: {exc}")
        try:
            import numpy as np
            import soundfile as sf

            audio_np, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            waveform = torch.from_numpy(np.asarray(audio_np).T)
        except Exception as fallback_exc:
            errors.append(f"soundfile: {fallback_exc}")
            details = " | ".join(errors)
            raise RuntimeError(f"Failed to decode audio file '{path}'. {details}") from fallback_exc
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    video = torch.zeros((0, 3, 0, 0), dtype=torch.float32)
    info = {"audio_fps": sample_rate, "video_fps": 0.0}
    return video, waveform.float(), info


@lru_cache(maxsize=128)
def _read_image_cached(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    image = Image.open(path).convert("RGB")
    tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    audio = torch.zeros((1, 0), dtype=torch.float32)
    info = {"audio_fps": 16000, "video_fps": 1.0}
    return tensor, audio, info


def _square_crop(frame: torch.Tensor, box: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    _, height, width = frame.shape
    if box is None:
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return frame[:, top:top + side, left:left + side]
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    return frame[:, y1:y2, x1:x2]


@dataclass
class FaceCropper:
    device: str = "cpu"
    use_mtcnn: bool = True
    margin: float = 0.15

    def __post_init__(self) -> None:
        self.detector = None
        if self.use_mtcnn:
            try:
                from facenet_pytorch import MTCNN

                self.detector = MTCNN(keep_all=False, device=self.device)
            except Exception:
                self.detector = None

    def _detect(self, frame: torch.Tensor) -> tuple[int, int, int, int] | None:
        if self.detector is None:
            return None
        pil_image = torchvision.transforms.functional.to_pil_image(frame.cpu())
        with torch.inference_mode():
            boxes, _ = self.detector.detect(pil_image)
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = boxes[0]
        width = x2 - x1
        height = y2 - y1
        x1 -= width * self.margin
        x2 += width * self.margin
        y1 -= height * self.margin
        y2 += height * self.margin
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def crop_frame(self, frame: torch.Tensor, *, mode: str = "full_face") -> torch.Tensor:
        crop = _square_crop(frame, self._detect(frame))
        if mode == "mouth":
            _, height, _ = crop.shape
            mouth = crop[:, int(height * 0.35):, :]
            return _square_crop(mouth)
        return crop


def _resample_audio(audio: torch.Tensor, original_rate: int, target_rate: int = 16000) -> torch.Tensor:
    if original_rate == target_rate:
        return audio
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError("torchaudio is required to resample audio.") from exc
    return torchaudio.functional.resample(audio, original_rate, target_rate)


def read_media(data_root: str | Path, relative_path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    path = (Path(data_root) / relative_path).resolve()
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return _read_audio_cached(str(path))
    if suffix in IMAGE_EXTENSIONS:
        return _read_image_cached(str(path))
    return _read_video_cached(str(path))


def read_frame(record: ManifestRecord, data_root: str | Path) -> torch.Tensor:
    path = (Path(data_root) / record.file).resolve()
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image, _, _ = read_media(data_root, record.file)
        return image[0].float().clone()

    metadata = _video_metadata(str(path))
    frame_count = max(int(metadata["video_frame_count"]), 1)
    frame_index = min(max(record.frame_index, 0), frame_count - 1)
    frame = _read_video_frames(str(path), [frame_index])[0]
    return frame.float().div(255.0).clone()


def read_video_clip(
    record: ManifestRecord,
    data_root: str | Path,
    *,
    target_fps: float = 8.0,
    target_frames: int | None = None,
) -> torch.Tensor:
    path = (Path(data_root) / record.file).resolve()
    metadata = _video_metadata(str(path))
    fps = float(metadata.get("video_fps", 25.0) or 25.0)
    frame_count = max(int(metadata.get("video_frame_count", 0)), 1)
    start = min(max(int(record.start_sec * fps), 0), frame_count - 1)
    if record.end_sec > record.start_sec:
        end = min(max(int(record.end_sec * fps), start + 1), frame_count)
    else:
        end = frame_count
    frame_indices = _sample_frame_indices(
        start=start,
        end=end,
        fps=fps,
        target_fps=target_fps,
        target_frames=target_frames,
    )
    clip = _read_video_frames(str(path), frame_indices)
    return clip.float().div(255.0).clone()


def read_audio_segment(record: ManifestRecord, data_root: str | Path, *, target_rate: int = 16000) -> torch.Tensor:
    _, audio, info = read_media(data_root, record.file)
    rate = int(info.get("audio_fps", target_rate) or target_rate)
    if audio.numel() == 0:
        audio = torch.zeros(1, int((record.end_sec - record.start_sec) * target_rate) or target_rate)
        rate = target_rate
    audio = _resample_audio(audio, rate, target_rate)
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    start = max(int(record.start_sec * target_rate), 0)
    if record.end_sec > record.start_sec:
        end = max(int(record.end_sec * target_rate), start + 1)
    else:
        end = audio.shape[0]
    return audio[start:end].clone()


def pad_or_trim_1d(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    waveform = waveform.flatten()
    if waveform.shape[0] == target_length:
        return waveform
    if waveform.shape[0] > target_length:
        return waveform[:target_length]
    return F.pad(waveform, (0, target_length - waveform.shape[0]))


def resize_frame(frame: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    size = (size, size) if isinstance(size, int) else size
    return F.interpolate(frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def resize_clip(clip: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    size = (size, size) if isinstance(size, int) else size
    return F.interpolate(clip, size=size, mode="bilinear", align_corners=False)


def grayscale_clip(clip: torch.Tensor) -> torch.Tensor:
    if clip.shape[1] == 1:
        return clip
    r, g, b = clip[:, 0:1], clip[:, 1:2], clip[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def crop_face_sequence(
    clip: torch.Tensor,
    *,
    cropper: FaceCropper,
    size: int,
    mode: str = "full_face",
    num_frames: int | None = None,
) -> torch.Tensor:
    if clip.shape[0] == 0:
        raise ValueError("Cannot crop an empty video clip.")
    if num_frames is not None:
        if clip.shape[0] >= num_frames:
            indices = torch.linspace(0, clip.shape[0] - 1, num_frames).round().long()
            clip = clip.index_select(0, indices)
        else:
            pad = clip[-1:].repeat(num_frames - clip.shape[0], 1, 1, 1)
            clip = torch.cat([clip, pad], dim=0)
    crops = []
    box = cropper._detect(clip[len(clip) // 2]) if cropper.detector is not None else None
    for frame in clip:
        crop = _square_crop(frame, box)
        if mode == "mouth":
            _, height, _ = crop.shape
            crop = crop[:, int(height * 0.35):, :]
            crop = _square_crop(crop)
        crops.append(resize_frame(crop, size))
    return torch.stack(crops, dim=0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    return torchvision.transforms.functional.to_pil_image(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torchvision.transforms.functional.to_tensor(image)


def video_labels_from_record(record: ManifestRecord, num_frames: int, fps: float) -> torch.Tensor:
    labels = torch.zeros(num_frames, dtype=torch.float32)
    if record.label == 0 or record.end_sec <= record.start_sec:
        return labels
    start_frame = int(record.start_sec * fps)
    end_frame = max(int(record.end_sec * fps), start_frame + 1)
    labels[start_frame:min(end_frame, num_frames)] = 1.0
    return labels
