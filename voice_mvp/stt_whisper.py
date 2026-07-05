from __future__ import annotations

from pathlib import Path


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_audio(path: Path, model_name: str = "base", device: str = "auto") -> str:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper is not installed. Run ./scripts/setup_macos.sh") from exc

    resolved_device = _resolve_device(device)
    model = whisper.load_model(model_name, device=resolved_device)
    result = model.transcribe(str(path), fp16=resolved_device != "cpu")
    return str(result.get("text", "")).strip()

