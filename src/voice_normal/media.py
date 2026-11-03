"""Читання аудіо й відео через ffmpeg/ffprobe.

Тільки списки аргументів, без shell. Шлях передається як є: ffmpeg на Windows сам
перетворює його в UTF-16 і додає префікс довгого шляху, тож кирилиця й пробіли
проходять без екранування.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_normal import tools


class MediaError(Exception):
    pass


@dataclass(frozen=True)
class MediaInfo:
    path: str
    size_bytes: int
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    has_video: bool


@dataclass(frozen=True)
class Loudness:
    integrated_lufs: float
    lra_lu: float
    true_peak_dbtp: float


def _exe(name: str) -> str:
    path = tools.find_tool(name)
    if path is None:
        raise MediaError(f"{name} not found in PATH")
    return str(path)


def probe(path: Path) -> MediaInfo:
    result = tools.run_tool(
        [_exe("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    if result.returncode != 0:
        raise MediaError(result.stderr.strip() or f"ffprobe failed on {path}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaError(f"no audio stream in {path}")
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams
    )
    duration = audio.get("duration") or data.get("format", {}).get("duration") or 0
    return MediaInfo(
        path=str(path),
        size_bytes=int(data.get("format", {}).get("size") or path.stat().st_size),
        duration_s=float(duration),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 1),
        codec=str(audio.get("codec_name") or "?"),
        has_video=has_video,
    )


def _decode_args(path: Path, sample_rate: int | None, channels: int | None) -> list[str]:
    args = [_exe("ffmpeg"), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn"]
    if channels is not None:
        args += ["-ac", str(channels)]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    return args + ["-f", "f32le", "-acodec", "pcm_f32le", "-"]


def decode(path: Path, sample_rate: int | None = None, channels: int | None = None) -> np.ndarray:
    """Увесь звук як float32 форми (семпли, канали)."""
    info_channels = channels or probe(path).channels
    result = subprocess.run(_decode_args(path, sample_rate, channels), capture_output=True, check=False)
    if result.returncode != 0:
        raise MediaError(result.stderr.decode("utf-8", "replace").strip() or f"ffmpeg failed on {path}")
    audio = np.frombuffer(result.stdout, dtype="<f4")
    return audio.reshape(-1, info_channels)


def iter_blocks(path: Path, channels: int, block_frames: int) -> Iterator[np.ndarray]:
    """Звук у рідній частоті блоками (кадри, канали) — без завантаження всього в пам'ять."""
    frame_bytes = 4 * channels
    process = subprocess.Popen(
        _decode_args(path, None, None), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        pending = b""
        while chunk := process.stdout.read(block_frames * frame_bytes):
            pending += chunk
            usable = len(pending) - len(pending) % frame_bytes
            if usable:
                yield np.frombuffer(pending[:usable], dtype="<f4").reshape(-1, channels)
                pending = pending[usable:]
    finally:
        process.stdout.close()
        stderr = process.stderr.read()
        process.stderr.close()
        code = process.wait()
    if code != 0:
        raise MediaError(stderr.decode("utf-8", "replace").strip() or f"ffmpeg failed on {path}")


_SUMMARY = re.compile(r"Summary:(.*)", re.S)
_NUMBER = r"(-?inf|-?\d+(?:\.\d+)?)"


def _value(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise MediaError("cannot parse ebur128 summary")
    raw = match.group(1)
    return -math.inf if raw.endswith("inf") else float(raw)


def loudness(path: Path) -> Loudness:
    """Гучність за EBU R128 / ITU-R BS.1770 і true peak — фільтром ebur128 самого ffmpeg."""
    result = tools.run_tool(
        [_exe("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-map", "0:a:0",
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        timeout=None,
    )
    if result.returncode != 0:
        raise MediaError(result.stderr.strip()[-500:] or f"ffmpeg failed on {path}")
    summary = _SUMMARY.search(result.stderr)
    if summary is None:
        raise MediaError("ebur128 printed no summary")
    text = summary.group(1)
    return Loudness(
        integrated_lufs=_value(rf"I:\s+{_NUMBER} LUFS", text),
        lra_lu=_value(rf"LRA:\s+{_NUMBER} LU\b", text),
        true_peak_dbtp=_value(rf"Peak:\s+{_NUMBER} dBFS", text),
    )
