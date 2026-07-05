from __future__ import annotations

import os
import json
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio_io import convert_audio
from .ollama_client import OllamaClient
from .stt_whisper import transcribe_audio
from .text_chunks import SentenceChunker
from .tts_csm import CsmTts

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs" / "server"
STATE_DIR = ROOT / "state"
REFERENCE_RAW = STATE_DIR / "reference_upload"
REFERENCE_WAV = STATE_DIR / "reference_24k.wav"
REFERENCE_TEXT = STATE_DIR / "reference.txt"


@dataclass(frozen=True)
class AssistantConfig:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    csm_model_id: str = os.getenv("CSM_MODEL_ID", "sesame/csm-1b")
    system_prompt: str = os.getenv(
        "VOICE_SYSTEM_PROMPT",
        "You are a low-latency local voice assistant. Answer in one short sentence by default. "
        "Use at most 10 words unless the user explicitly asks for detail.",
    )
    tts_device: str = os.getenv("TTS_DEVICE", "auto")
    tts_dtype: str = os.getenv("TTS_DTYPE", "float16")
    speaker: str = os.getenv("CSM_SPEAKER", "0")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
    max_tts_chunks: int = int(os.getenv("MAX_TTS_CHUNKS", "1"))
    max_new_tokens: int = int(os.getenv("CSM_MAX_NEW_TOKENS", "80"))
    max_spoken_words: int = int(os.getenv("MAX_SPOKEN_WORDS", "10"))
    reference_seconds: float = float(os.getenv("REFERENCE_SECONDS", "3"))
    auto_warmup: bool = os.getenv("AUTO_WARMUP", "1") != "0"


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)
    session_id: str | None = None


class WarmupResponse(BaseModel):
    loaded: bool
    device: str


class VoiceAssistantService:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self.ollama = OllamaClient(config.ollama_base_url)
        self._tts: CsmTts | None = None
        self._tts_lock = threading.Lock()
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._session_lock = threading.Lock()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, object]:
        models = []
        try:
            tags = self.ollama.tags()
            models = [item.get("name", "") for item in tags.get("models", [])]
        except Exception:
            models = []

        return {
            "ollama_model": self.config.ollama_model,
            "ollama_models": models,
            "csm_model_id": self.config.csm_model_id,
            "tts_loaded": self._tts is not None,
            "voice_locked": self.voice_locked,
            "reference_text": self.reference_text or "",
        }

    @property
    def reference_audio(self) -> Path | None:
        return REFERENCE_WAV if REFERENCE_WAV.exists() else None

    @property
    def reference_text(self) -> str | None:
        if not REFERENCE_TEXT.exists():
            return None
        text = REFERENCE_TEXT.read_text(encoding="utf-8").strip()
        return text or None

    @property
    def voice_locked(self) -> bool:
        return self.reference_audio is not None and self.reference_text is not None

    def warmup(self) -> WarmupResponse:
        tts = self._ensure_tts()
        return WarmupResponse(loaded=True, device=tts.device)

    def set_reference(self, upload: UploadFile, transcript: str) -> dict[str, object]:
        clean_transcript = transcript.strip()
        if not clean_transcript:
            raise ValueError("Reference transcript is required.")

        suffix = Path(upload.filename or "reference.webm").suffix or ".webm"
        raw_path = REFERENCE_RAW.with_suffix(suffix)
        with raw_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        convert_audio(
            raw_path,
            REFERENCE_WAV,
            sample_rate=CsmTts.sample_rate,
            max_duration=self.config.reference_seconds,
        )
        REFERENCE_TEXT.write_text(clean_transcript, encoding="utf-8")
        return {
            "voice_locked": True,
            "reference_audio": str(REFERENCE_WAV),
            "reference_text": clean_transcript,
        }

    def stream_chat_events(self, text: str, session_id: str | None = None) -> Iterator[str]:
        clean_text = text.strip()
        if not clean_text:
            yield _sse("error", {"message": "Message text is required."})
            return

        session = session_id or uuid.uuid4().hex
        started = time.perf_counter()
        messages = self._messages_for_session(session, clean_text)
        response_parts: list[str] = []
        chunker = SentenceChunker(min_chars=90)
        work_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        chunk_count = 0
        spoken_word_count = 0
        tts_started: float | None = None
        saw_tts_done = False

        yield _sse("start", {"session_id": session, "input_text": clean_text})

        def enqueue_chunk(chunk: str) -> None:
            nonlocal chunk_count, spoken_word_count, tts_started
            if chunk_count >= self.config.max_tts_chunks:
                return
            remaining_words = self.config.max_spoken_words - spoken_word_count
            spoken_chunk = _clip_words(chunk, remaining_words)
            if not spoken_chunk:
                return
            spoken_word_count += len(spoken_chunk.split())
            chunk_count += 1
            if tts_started is None:
                tts_started = time.perf_counter()
            work_queue.put((chunk_count, spoken_chunk))
            event_queue.put({"type": "tts_queued", "index": chunk_count, "text": spoken_chunk})

        def tts_worker() -> None:
            try:
                tts = self._ensure_tts()
                while True:
                    item = work_queue.get()
                    if item is None:
                        return
                    index, chunk = item
                    stamp = int(time.time() * 1000)
                    filename = f"{session}_{stamp}_{index:02d}.wav"
                    output_path = OUTPUT_DIR / filename
                    chunk_started = time.perf_counter()
                    with self._tts_lock:
                        tts.synthesize(
                            chunk,
                            output_path,
                            reference_audio=self.reference_audio,
                            reference_text=self.reference_text,
                            speaker=self.config.speaker,
                            max_new_tokens=self.config.max_new_tokens,
                        )
                    event_queue.put(
                        {
                            "type": "audio",
                            "index": index,
                            "text": chunk,
                            "url": f"/audio/{filename}",
                            "seconds": round(time.perf_counter() - chunk_started, 3),
                        }
                    )
            except BaseException as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put({"type": "tts_done"})

        worker = threading.Thread(target=tts_worker, daemon=True)
        worker.start()

        def drain_events() -> Iterator[str]:
            nonlocal saw_tts_done
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    return
                saw_tts_done = saw_tts_done or event.get("type") == "tts_done"
                yield _sse(str(event.get("type", "event")), event)

        llm_started = time.perf_counter()
        try:
            for token in self.ollama.stream_messages(
                model=self.config.ollama_model,
                messages=messages,
                temperature=0.4,
            ):
                response_parts.append(token)
                yield _sse("text", {"delta": token})
                for chunk in chunker.push(token):
                    enqueue_chunk(chunk)
                yield from drain_events()

            llm_seconds = time.perf_counter() - llm_started
            for chunk in chunker.flush():
                enqueue_chunk(chunk)
            work_queue.put(None)

            while not saw_tts_done:
                event = event_queue.get()
                saw_tts_done = event.get("type") == "tts_done"
                yield _sse(str(event.get("type", "event")), event)

            response = "".join(response_parts).strip()
            self._append_session(session, clean_text, response)
            tts_seconds = 0.0 if tts_started is None else time.perf_counter() - tts_started
            yield _sse(
                "done",
                {
                    "session_id": session,
                    "response_text": response,
                    "timings": {
                        "llm_seconds": round(llm_seconds, 3),
                        "tts_seconds": round(tts_seconds, 3),
                        "total_seconds": round(time.perf_counter() - started, 3),
                    },
                    "voice_locked": self.voice_locked,
                },
            )
        except BaseException as exc:
            work_queue.put(None)
            yield _sse("error", {"message": str(exc)})

    def chat(self, text: str, session_id: str | None = None) -> dict[str, object]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Message text is required.")

        session = session_id or uuid.uuid4().hex
        started = time.perf_counter()
        messages = self._messages_for_session(session, clean_text)

        llm_started = time.perf_counter()
        response = self.ollama.chat_messages(
            model=self.config.ollama_model,
            messages=messages,
            temperature=0.4,
        )
        llm_seconds = time.perf_counter() - llm_started

        chunks = self._chunks_for_tts(response)
        tts_started = time.perf_counter()
        audio_urls = self._synthesize_chunks(session, chunks)
        tts_seconds = time.perf_counter() - tts_started

        self._append_session(session, clean_text, response)
        return {
            "session_id": session,
            "input_text": clean_text,
            "response_text": response,
            "chunks": chunks,
            "audio_urls": audio_urls,
            "timings": {
                "llm_seconds": round(llm_seconds, 3),
                "tts_seconds": round(tts_seconds, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
            },
            "voice_locked": self.voice_locked,
        }

    def transcribe_and_chat(self, upload: UploadFile, session_id: str | None = None) -> dict[str, object]:
        suffix = Path(upload.filename or "voice.webm").suffix or ".webm"
        input_path = OUTPUT_DIR / f"input_{uuid.uuid4().hex}{suffix}"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        transcribe_started = time.perf_counter()
        text = transcribe_audio(
            input_path,
            model_name=self.config.whisper_model,
            device=self.config.whisper_device,
        )
        stt_seconds = time.perf_counter() - transcribe_started
        result = self.chat(text, session_id=session_id)
        timings = dict(result["timings"])
        timings["stt_seconds"] = round(stt_seconds, 3)
        timings["total_seconds"] = round(timings["total_seconds"] + timings["stt_seconds"], 3)
        result["timings"] = timings
        return result

    def stream_transcribed_chat_events(self, input_path: Path, session_id: str | None = None) -> Iterator[str]:
        try:
            started = time.perf_counter()
            yield _sse("status", {"message": "transcribing"})
            text = transcribe_audio(
                input_path,
                model_name=self.config.whisper_model,
                device=self.config.whisper_device,
            )
            yield _sse(
                "transcript",
                {"text": text, "stt_seconds": round(time.perf_counter() - started, 3)},
            )
            yield from self.stream_chat_events(text, session_id=session_id)
        except BaseException as exc:
            yield _sse("error", {"message": str(exc)})

    def _ensure_tts(self) -> CsmTts:
        if self._tts is None:
            with self._tts_lock:
                if self._tts is None:
                    self._tts = CsmTts(
                        model_id=self.config.csm_model_id,
                        device=self.config.tts_device,
                        dtype=self.config.tts_dtype,
                        speaker=self.config.speaker,
                        reference_audio=None,
                        reference_text=None,
                        allow_cpu=False,
                        max_new_tokens=self.config.max_new_tokens,
                    )
        return self._tts

    def _messages_for_session(self, session_id: str, user_text: str) -> list[dict[str, str]]:
        with self._session_lock:
            history = list(self._sessions.get(session_id, []))[-self.config.max_history_messages :]
        return [{"role": "system", "content": self.config.system_prompt}, *history, {"role": "user", "content": user_text}]

    def _append_session(self, session_id: str, user_text: str, response: str) -> None:
        with self._session_lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response},
                ]
            )
            del history[: max(0, len(history) - self.config.max_history_messages)]

    def _chunks_for_tts(self, text: str) -> list[str]:
        chunker = SentenceChunker(min_chars=110)
        spoken_text = _clip_words(text, self.config.max_spoken_words)
        chunks = chunker.push(spoken_text) + chunker.flush()
        if not chunks:
            chunks = [spoken_text.strip()]
        return chunks[: self.config.max_tts_chunks]

    def _synthesize_chunks(self, session_id: str, chunks: list[str]) -> list[str]:
        tts = self._ensure_tts()
        stamp = int(time.time() * 1000)
        urls: list[str] = []
        with self._tts_lock:
            for index, chunk in enumerate(chunks, 1):
                filename = f"{session_id}_{stamp}_{index:02d}.wav"
                output_path = OUTPUT_DIR / filename
                tts.synthesize(
                    chunk,
                    output_path,
                    reference_audio=self.reference_audio,
                    reference_text=self.reference_text,
                    speaker=self.config.speaker,
                    max_new_tokens=self.config.max_new_tokens,
                )
                urls.append(f"/audio/{filename}")
        return urls


config = AssistantConfig()
service = VoiceAssistantService(config)
app = FastAPI(title="CSM Voice Console")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.on_event("startup")
def warmup_on_startup() -> None:
    if not config.auto_warmup:
        return

    def run() -> None:
        try:
            service.warmup()
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, object]:
    return service.status()


@app.post("/api/warmup")
def warmup() -> WarmupResponse:
    return service.warmup()


@app.post("/api/reference")
def set_reference(
    audio: Annotated[UploadFile, File()],
    transcript: Annotated[str, Form()],
) -> dict[str, object]:
    try:
        return service.set_reference(audio, transcript)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    try:
        return service.chat(request.text, session_id=request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        service.stream_chat_events(request.text, session_id=request.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/talk")
def talk(
    audio: Annotated[UploadFile, File()],
    session_id: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    try:
        return service.transcribe_and_chat(audio, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/talk/stream")
def talk_stream(
    audio: Annotated[UploadFile, File()],
    session_id: Annotated[str | None, Form()] = None,
) -> StreamingResponse:
    suffix = Path(audio.filename or "voice.webm").suffix or ".webm"
    input_path = OUTPUT_DIR / f"input_{uuid.uuid4().hex}{suffix}"
    with input_path.open("wb") as handle:
        shutil.copyfileobj(audio.file, handle)
    return StreamingResponse(
        service.stream_transcribed_chat_events(input_path, session_id=session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/audio/{filename}")
def audio(filename: str) -> FileResponse:
    output_root = OUTPUT_DIR.resolve()
    path = (OUTPUT_DIR / filename).resolve()
    if not path.exists() or path.parent != output_root:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav")


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clip_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = text.strip().split()
    if not words:
        return ""
    if len(words) <= max_words:
        return text.strip()
    clipped = " ".join(words[:max_words]).rstrip(",;:")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped
