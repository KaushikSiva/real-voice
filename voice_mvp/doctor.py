from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import urllib.error
import urllib.request


def ok(name: str, value: str = "") -> None:
    suffix = f": {value}" if value else ""
    print(f"[ok]   {name}{suffix}")


def warn(name: str, value: str = "") -> None:
    suffix = f": {value}" if value else ""
    print(f"[warn] {name}{suffix}")


def fail(name: str, value: str = "") -> None:
    suffix = f": {value}" if value else ""
    print(f"[fail] {name}{suffix}")


def check_module(name: str) -> bool:
    if importlib.util.find_spec(name):
        ok(f"python module {name}")
        return True
    fail(f"python module {name}", "not installed")
    return False


def check_ollama(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        fail("ollama API", str(exc))
        return False

    models = [model.get("name", "<unknown>") for model in payload.get("models", [])]
    ok("ollama API", f"{len(models)} model(s)")
    if models:
        print("       " + ", ".join(models[:8]))
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        fail("torch", "not installed")
        return False

    ok("torch", torch.__version__)
    if torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            ok("torch mps", "available")
        else:
            warn("torch mps", "built but runtime unavailable")
    else:
        warn("torch mps", "not built")

    if torch.cuda.is_available():
        ok("torch cuda", torch.version.cuda or "available")
    else:
        warn("torch cuda", "not available")
    return True


def check_csm_download(model_id: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        fail("huggingface_hub", "not installed")
        return False

    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
    except Exception as exc:
        fail("huggingface model access", f"{model_id}: {exc}")
        return False

    ok("huggingface model access", path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local voice MVP prerequisites.")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--csm-model-id", default="sesame/csm-1b")
    parser.add_argument("--check-csm-download", action="store_true")
    args = parser.parse_args(argv)

    failures = 0
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()} / {platform.machine()}")

    if sys.version_info[:2] != (3, 12):
        warn("python version", "3.12 is recommended for this MVP")
    else:
        ok("python version", "3.12")

    executables = ["ffmpeg", "ollama"]
    if platform.system() == "Darwin":
        executables.append("afplay")

    for executable in executables:
        path = shutil.which(executable)
        if path:
            ok(executable, path)
        else:
            fail(executable, "not found on PATH")
            failures += 1

    for module in ("requests", "soundfile", "whisper", "transformers"):
        if not check_module(module):
            failures += 1

    if not check_torch():
        failures += 1

    try:
        from transformers import CsmForConditionalGeneration  # noqa: F401

        ok("transformers CSM class")
    except Exception as exc:
        fail("transformers CSM class", str(exc))
        failures += 1

    if not check_ollama(args.ollama_base_url):
        failures += 1

    if args.check_csm_download and not check_csm_download(args.csm_model_id):
        failures += 1

    if failures:
        print(f"\n{failures} check(s) failed.")
        return 1

    print("\nDoctor checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
