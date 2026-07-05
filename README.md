# Local CSM Voice MVP

One-turn local voice assistant MVP for Mac:

```text
Whisper STT -> Ollama LLM -> CSM-1B TTS -> afplay
```

The first run downloads models and can take a while. CSM-1B is gated on Hugging Face, so you must request access at <https://huggingface.co/sesame/csm-1b>, be approved, and log in before TTS can load.

## Setup

```bash
./scripts/setup_macos.sh
source .venv/bin/activate
huggingface-cli login
ollama pull llama3.2:3b
```

If Ollama is not already running:

```bash
ollama serve
```

In another terminal, run the doctor:

```bash
./scripts/doctor.py
```

For a deeper Hugging Face/CSM check:

```bash
./scripts/doctor.py --check-csm-download
```

## Text-In MVP

This skips Whisper and sends typed text directly to Ollama:

```bash
./scripts/run_mvp.sh --text "Give me a short morning briefing."
```

By default the LLM response streams to the terminal, and each completed sentence is sent to CSM and played as a WAV chunk.

## Browser Voice Console

Run the FastAPI server and open the UI:

```bash
./scripts/run_server.sh
```

Then visit:

```text
http://127.0.0.1:8000
```

The server keeps CSM loaded after the first request, so later turns avoid repeated model-load time. Upload a short reference clip plus its exact transcript in the Voice panel to keep the same reference voice across responses.
Reference uploads are trimmed to the first 3 seconds by default, so the transcript should match that portion exactly.

Useful API checks:

```bash
curl -s http://127.0.0.1:8000/api/status
curl -s -X POST http://127.0.0.1:8000/api/warmup
```

Latency-oriented defaults can be overridden:

```bash
MAX_TTS_CHUNKS=1 CSM_MAX_NEW_TOKENS=80 MAX_SPOKEN_WORDS=10 REFERENCE_SECONDS=3 ./scripts/run_server.sh
```

## Voice-In MVP

Record from the default macOS microphone for 5 seconds:

```bash
./scripts/run_mvp.sh --record-seconds 5
```

Or transcribe an existing WAV/MP3:

```bash
./scripts/run_mvp.sh --audio-input path/to/input.wav
```

## Reference Voice

CSM is steadier with a short reference clip and matching transcript. Use a clean 3-10 second clip; it will be resampled to 24 kHz for CSM.

```bash
./scripts/run_mvp.sh \
  --text "Say this in the reference style." \
  --reference-audio path/to/reference.wav \
  --reference-text "Exact words spoken in the reference clip."
```

Do not use a real person's voice without explicit consent.

## Common Commands

List macOS capture devices:

```bash
./scripts/list_audio_devices.sh
```

Run without audio playback:

```bash
./scripts/run_mvp.sh --text "Hello" --no-play
```

Use a different Ollama model:

```bash
OLLAMA_MODEL=llama3.1:8b ./scripts/run_mvp.sh --text "Keep it brief."
```

## Notes

- CSM output is saved at 24 kHz.
- The default model is `sesame/csm-1b` through Transformers.
- The default Ollama model is `llama3.2:3b` because it is a practical first smoke test on Mac. Override it when your local 7B/8B model is pulled and working.
- Python 3.12 is used because Python 3.13 is still a frequent source of dependency friction in local audio/ML stacks.
- If `torch.backends.mps.is_available()` is false, this MVP will stop before loading CSM unless you explicitly pass `--allow-cpu-tts`.
