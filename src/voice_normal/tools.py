"""Пошук зовнішніх програм (ffmpeg, ffprobe, uv) і підказки, як їх поставити."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_TOOLS = ("ffmpeg", "ffprobe", "uv")

_INSTALL_HINTS: dict[str, dict[str, list[str]]] = {
    "win32": {
        "ffmpeg": ["winget install Gyan.FFmpeg"],
        "ffprobe": ["winget install Gyan.FFmpeg"],
        "uv": [
            "winget install astral-sh.uv",
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        ],
    },
    "darwin": {
        "ffmpeg": ["brew install ffmpeg"],
        "ffprobe": ["brew install ffmpeg"],
        "uv": ["brew install uv"],
    },
    "linux": {
        "ffmpeg": ["sudo apt install ffmpeg"],
        "ffprobe": ["sudo apt install ffmpeg"],
        "uv": ["curl -LsSf https://astral.sh/uv/install.sh | sh"],
    },
}


def find_tool(name: str) -> Path | None:
    # shutil.which сам додає .exe за PATHEXT на Windows.
    found = shutil.which(name)
    return Path(found) if found else None


def run_tool(args: list[str | Path], timeout: float | None = 30) -> subprocess.CompletedProcess[str]:
    """Запуск зовнішньої програми: лише список аргументів, без shell=True.

    Вивід декодуємо як UTF-8 із заміною: ffmpeg друкує шляхи з кирилицею в UTF-8.
    """
    return subprocess.run(
        [str(a) for a in args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def tool_version(path: Path) -> str | None:
    flag = "--version" if path.stem.lower() == "uv" else "-version"
    try:
        result = run_tool([path, flag], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    first = result.stdout.splitlines()[0].split()
    # "ffmpeg version 7.1 Copyright ..." / "uv 0.13.1 (...)"
    if "version" in first:
        idx = first.index("version")
        return first[idx + 1] if idx + 1 < len(first) else None
    return first[1] if len(first) > 1 else None


def install_hints(name: str, platform: str | None = None) -> list[str]:
    platform = platform or sys.platform
    key = platform if platform in _INSTALL_HINTS else "linux"
    return list(_INSTALL_HINTS[key].get(name, []))
