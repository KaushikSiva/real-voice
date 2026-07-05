from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioToolError(RuntimeError):
    pass


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AudioToolError(f"Required executable not found on PATH: {name}")
    return path


def record_audio(output_path: Path, seconds: float, audio_device: str = ":0") -> Path:
    require_executable("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        audio_device,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def play_audio(path: Path) -> None:
    require_executable("afplay")
    subprocess.run(["afplay", str(path)], check=True)


def convert_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 24000,
    max_duration: float | None = None,
) -> Path:
    require_executable("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
    ]
    if max_duration is not None:
        cmd.extend(["-t", str(max_duration)])
    cmd.extend(
        [
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)
    return output_path
