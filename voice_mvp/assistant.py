from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .audio_io import play_audio, record_audio
from .ollama_client import OllamaClient
from .stt_whisper import transcribe_audio
from .text_chunks import SentenceChunker
from .tts_csm import CsmTts


@dataclass(frozen=True)
class RunConfig:
    ollama_base_url: str
    ollama_model: str
    csm_model_id: str
    output_dir: Path
    play_audio: bool
    stream_tts: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Whisper -> Ollama -> CSM voice MVP")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Typed user input. Skips Whisper.")
    input_group.add_argument("--audio-input", type=Path, help="Audio file to transcribe with Whisper.")
    input_group.add_argument(
        "--record-seconds",
        type=float,
        help="Record this many seconds from the default macOS microphone.",
    )

    parser.add_argument("--audio-device", default=":0", help="ffmpeg avfoundation input, default ':0'.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name, e.g. tiny/base/small.")
    parser.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])

    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--system-prompt", default="You are a concise local voice assistant. Keep replies brief.")
    parser.add_argument("--temperature", type=float, default=0.4)

    parser.add_argument("--csm-model-id", default=os.getenv("CSM_MODEL_ID", "sesame/csm-1b"))
    parser.add_argument("--speaker", default="0", help="CSM speaker role/id.")
    parser.add_argument("--reference-audio", type=Path, help="Optional reference/context voice clip.")
    parser.add_argument("--reference-text", help="Transcript of --reference-audio.")
    parser.add_argument(
        "--tts-device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device for CSM. 'auto' prefers MPS on Mac.",
    )
    parser.add_argument(
        "--tts-dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. float16 keeps memory lower on Apple Silicon.",
    )
    parser.add_argument("--max-new-tokens", type=int, help="Optional CSM generation token limit.")
    parser.add_argument(
        "--allow-cpu-tts",
        action="store_true",
        help="Allow CSM on CPU. This is very slow and is off by default.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-play", action="store_true", help="Save WAV files without playing them.")
    parser.add_argument(
        "--no-stream-tts",
        action="store_true",
        help="Wait for full LLM response, then generate one TTS file.",
    )
    return parser.parse_args(argv)


def resolve_user_text(args: argparse.Namespace, output_dir: Path) -> str:
    if args.text:
        return args.text.strip()

    if args.audio_input:
        audio_path = args.audio_input
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = output_dir / "recordings" / f"mic_{stamp}.wav"
        print(f"Recording {args.record_seconds:.1f}s to {audio_path}...", file=sys.stderr)
        record_audio(audio_path, args.record_seconds, args.audio_device)

    print(f"Transcribing {audio_path} with Whisper {args.whisper_model}...", file=sys.stderr)
    return transcribe_audio(audio_path, model_name=args.whisper_model, device=args.whisper_device)


def build_tts(args: argparse.Namespace) -> CsmTts:
    if bool(args.reference_audio) != bool(args.reference_text):
        raise ValueError("--reference-audio and --reference-text must be provided together.")

    return CsmTts(
        model_id=args.csm_model_id,
        device=args.tts_device,
        dtype=args.tts_dtype,
        speaker=args.speaker,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        allow_cpu=args.allow_cpu_tts,
        max_new_tokens=args.max_new_tokens,
    )


def run_streaming_tts(
    client: OllamaClient,
    tts: CsmTts,
    cfg: RunConfig,
    args: argparse.Namespace,
    user_text: str,
) -> str:
    text_queue: queue.Queue[str | None] = queue.Queue()
    errors: queue.Queue[BaseException] = queue.Queue()
    chunker = SentenceChunker()
    response_parts: list[str] = []

    def tts_worker() -> None:
        idx = 1
        while True:
            text = text_queue.get()
            if text is None:
                return
            try:
                wav_path = cfg.output_dir / f"tts_chunk_{idx:03d}.wav"
                print(f"\n[TTS {idx}] {text}", file=sys.stderr)
                tts.synthesize(text, wav_path)
                print(f"[saved] {wav_path}", file=sys.stderr)
                if cfg.play_audio:
                    play_audio(wav_path)
                idx += 1
            except BaseException as exc:
                errors.put(exc)
                return

    worker = threading.Thread(target=tts_worker, daemon=True)
    worker.start()

    for token in client.stream_chat(
        model=args.ollama_model,
        user_text=user_text,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
    ):
        print(token, end="", flush=True)
        response_parts.append(token)
        for sentence in chunker.push(token):
            text_queue.put(sentence)
        if not errors.empty():
            raise errors.get()

    print()
    for sentence in chunker.flush():
        text_queue.put(sentence)
    text_queue.put(None)
    worker.join()

    if not errors.empty():
        raise errors.get()

    return "".join(response_parts).strip()


def run_single_tts(
    client: OllamaClient,
    tts: CsmTts,
    cfg: RunConfig,
    args: argparse.Namespace,
    user_text: str,
) -> str:
    response = client.chat(
        model=args.ollama_model,
        user_text=user_text,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
    )
    print(response)
    wav_path = cfg.output_dir / "tts_response.wav"
    tts.synthesize(response, wav_path)
    print(f"[saved] {wav_path}", file=sys.stderr)
    if cfg.play_audio:
        play_audio(wav_path)
    return response


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = RunConfig(
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        csm_model_id=args.csm_model_id,
        output_dir=args.output_dir,
        play_audio=not args.no_play,
        stream_tts=not args.no_stream_tts,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_text = resolve_user_text(args, cfg.output_dir).strip()
        if not user_text:
            raise ValueError("No user text was provided or transcribed.")

        print(f"\n[User] {user_text}\n", file=sys.stderr)
        print("Loading CSM TTS...", file=sys.stderr)
        tts = build_tts(args)
        client = OllamaClient(cfg.ollama_base_url)

        if cfg.stream_tts:
            run_streaming_tts(client, tts, cfg, args, user_text)
        else:
            run_single_tts(client, tts, cfg, args, user_text)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

