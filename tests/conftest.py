import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


ML_READY = all(_have(m) for m in ("numpy", "soundfile", "onnxruntime", "faster_whisper"))


@pytest.fixture(scope="session")
def models_ready():
    """Справжні моделі з VOICE_NORMAL_HOME (на CI) або з папки користувача."""
    if not ML_READY:
        pytest.skip("ML-пакети не встановлено (voice-normal install)")
    from voice_normal import models

    manifest = models.read_manifest()
    if manifest is None or models.missing_files(manifest.get("device", "cpu")):
        pytest.skip("моделі не завантажено (voice-normal install)")
    return manifest


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("Тестові записи")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


SPEECH_TEXT = (
    "Good afternoon. This is a short test recording for the voice processing tool. "
    "I speak at a normal pace and pause between sentences, so there is silence in between. "
    "Imagine a video recorded in a room with bare walls and a laptop fan in the background. "
    "The words are clear, but the sound is tiring to listen to for a long time. "
    "The tool keeps the voice, removes everything else, and makes the speech sound close and even."
)


def tts(out: Path, text: str = SPEECH_TEXT, voice: str = "*") -> Path:
    """Мова від вбудованого синтезатора Windows (System.Speech), 48 кГц моно.

    voice — шаблон імені англійського голосу, наприклад "*David*"; без потрібного
    голосу чи без Windows тест пропускається.
    """
    if sys.platform != "win32" or shutil.which("ffmpeg") is None:
        pytest.skip("потрібні Windows (System.Speech) і ffmpeg")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' -and "
        "$_.VoiceInfo.Name -like $env:VN_TTS_VOICE } | Select-Object -First 1;"
        "if (-not $v) { exit 3 };"
        "$s.SelectVoice($v.VoiceInfo.Name);"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 48000,"
        "([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen),([System.Speech.AudioFormat.AudioChannel]::Mono);"
        "$s.SetOutputToWaveFile($env:VN_TTS_OUT, $f); $s.Speak($env:VN_TTS_TEXT); $s.Dispose()"
    )
    env = {**os.environ, "VN_TTS_OUT": str(out), "VN_TTS_TEXT": text, "VN_TTS_VOICE": voice}
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env, capture_output=True, check=False
    )
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"немає англійського голосу System.Speech ({voice})")
    return out


@pytest.fixture(scope="session")
def speech_wav(media_dir):
    """Справжня мова від вбудованого синтезатора Windows; без нього тести пропускаються."""
    return tts(media_dir / "мова чиста.wav")


@pytest.fixture(scope="session")
def noisy_wav(speech_wav, media_dir):
    out = media_dir / "мова з шумом.wav"
    ffmpeg(
        "-i", str(speech_wav),
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.08:sample_rate=48000:duration=120:seed=7",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0:duration=first",
        str(out),
    )
    return out


def write_wav(path: Path, data, sr: int) -> Path:
    import soundfile as sf

    sf.write(str(path), data, sr, subtype="FLOAT")
    return path
