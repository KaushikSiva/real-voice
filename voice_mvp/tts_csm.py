from __future__ import annotations

from pathlib import Path
from typing import Any


class CsmTts:
    sample_rate = 24000

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        speaker: str,
        reference_audio: Path | None,
        reference_text: str | None,
        allow_cpu: bool,
        max_new_tokens: int | None,
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device, allow_cpu=allow_cpu)
        self.dtype = dtype
        self.speaker = speaker
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.max_new_tokens = max_new_tokens
        self.processor: Any | None = None
        self.model: Any | None = None

        self._load()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
        speaker: str | None = None,
        max_new_tokens: int | None = None,
        context_segments: list[dict[str, Any]] | None = None,
    ) -> Path:
        if self.processor is None or self.model is None:
            raise RuntimeError("CSM model is not loaded.")

        conversation = self._conversation(
            text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            speaker=speaker,
            context_segments=context_segments,
        )
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)
        inputs = self._cast_floating_inputs(inputs)

        kwargs: dict[str, Any] = {"output_audio": True}
        token_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if token_limit is not None:
            kwargs["max_new_tokens"] = token_limit

        import torch

        with torch.inference_mode():
            audio = self.model.generate(**inputs, **kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.processor.save_audio(audio, output_path)
        return output_path

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CsmForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("CSM dependencies are missing. Run ./scripts/setup_macos.sh") from exc

        dtype = self._torch_dtype(torch, self.dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load CSM processor for {self.model_id}. If this is sesame/csm-1b, "
                "run `huggingface-cli login` and request/accept access at "
                "https://huggingface.co/sesame/csm-1b."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        try:
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map={"": self.device},
                **load_kwargs,
            )
        except Exception:
            try:
                self.model = CsmForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
                self.model.to(self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load CSM model weights for {self.model_id}. If this is sesame/csm-1b, "
                    "run `huggingface-cli login` and request/accept access at "
                    "https://huggingface.co/sesame/csm-1b."
                ) from exc

        self.model.eval()

    def _cast_floating_inputs(self, inputs: Any) -> Any:
        if self.model is None:
            return inputs

        import torch

        model_dtype = next(self.model.parameters()).dtype
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                inputs[key] = value.to(device=self.device, dtype=model_dtype)
        return inputs

    def _conversation(
        self,
        text: str,
        reference_audio: Path | None = None,
        reference_text: str | None = None,
        speaker: str | None = None,
        context_segments: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        active_reference_audio = reference_audio or self.reference_audio
        active_reference_text = reference_text or self.reference_text
        active_speaker = speaker or self.speaker
        conversation: list[dict[str, Any]] = []
        if active_reference_audio and active_reference_text:
            audio = _load_audio_24khz(active_reference_audio)
            conversation.append(
                {
                    "role": active_speaker,
                    "content": [
                        {"type": "text", "text": active_reference_text},
                        {"type": "audio", "path": audio},
                    ],
                }
            )

        for segment in context_segments or []:
            segment_text = str(segment.get("text", "")).strip()
            segment_audio = segment.get("audio_path")
            if not segment_text or not segment_audio:
                continue
            audio_path = Path(segment_audio)
            if not audio_path.exists():
                continue
            conversation.append(
                {
                    "role": str(segment.get("speaker", active_speaker)),
                    "content": [
                        {"type": "text", "text": segment_text},
                        {"type": "audio", "path": _load_audio_24khz(audio_path)},
                    ],
                }
            )

        conversation.append(
            {
                "role": active_speaker,
                "content": [{"type": "text", "text": text}],
            }
        )
        return conversation

    @staticmethod
    def _resolve_device(requested: str, allow_cpu: bool) -> str:
        import torch

        if requested == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            if allow_cpu:
                return "cpu"
            raise RuntimeError(
                "No GPU backend is available for CSM. On Apple Silicon, fix PyTorch MPS first "
                "or pass --allow-cpu-tts for a very slow CPU run."
            )

        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested --tts-device mps, but torch.backends.mps.is_available() is false.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --tts-device cuda, but torch.cuda.is_available() is false.")
        if requested == "cpu" and not allow_cpu:
            raise RuntimeError("CPU TTS requested. Add --allow-cpu-tts if you really want this slow path.")
        return requested

    @staticmethod
    def _torch_dtype(torch: Any, dtype: str) -> Any | None:
        if dtype == "auto":
            return None
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]


def _load_audio_24khz(path: Path) -> Any:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != CsmTts.sample_rate:
        audio = _resample_linear(audio, sample_rate, CsmTts.sample_rate)

    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio.astype(np.float32)


def _resample_linear(audio: Any, orig_rate: int, new_rate: int) -> Any:
    import numpy as np

    if orig_rate == new_rate:
        return audio.astype(np.float32)

    duration = len(audio) / float(orig_rate)
    new_length = max(1, int(round(duration * new_rate)))
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_x = np.linspace(0.0, duration, num=new_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)
