import sys
from pathlib import Path

import pytest

from voice_normal import paths


def test_override_wins(monkeypatch, tmp_path):
    target = tmp_path / "Тест папка"
    monkeypatch.setenv("VOICE_NORMAL_HOME", str(target))
    assert paths.data_dir() == target
    assert paths.models_dir() == target / "models"


def test_windows_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICE_NORMAL_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.data_dir() == tmp_path / "voice-normal"


def test_linux_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICE_NORMAL_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.data_dir() == tmp_path / "voice-normal"


def test_temp_dir_is_unique():
    first, second = paths.make_temp_dir(), paths.make_temp_dir()
    try:
        assert first != second and first.is_dir() and second.is_dir()
    finally:
        first.rmdir()
        second.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="MAX_PATH існує лише у Windows")
def test_short_path_unchanged(tmp_path):
    assert paths.long_path(tmp_path) == str(tmp_path.absolute())


@pytest.mark.skipif(sys.platform != "win32", reason="MAX_PATH існує лише у Windows")
def test_long_cyrillic_path_roundtrip(tmp_path):
    deep = tmp_path
    while len(str(deep)) < 300:
        deep = deep / "довга папка з пробілами"
    target = deep / "звіт.md"
    assert len(str(target)) > 260
    assert paths.long_path(target).startswith("\\\\?\\")

    import os

    os.makedirs(paths.long_path(deep), exist_ok=True)
    from voice_normal.console import write_text

    write_text(target, "Привіт — тест\n")
    with open(paths.long_path(target), "rb") as fh:
        assert fh.read() == "Привіт — тест\n".encode("utf-8")


def test_report_is_utf8_with_lf(tmp_path):
    from voice_normal.console import write_text

    target = tmp_path / "Тест папка" / "report.md"
    target.parent.mkdir()
    write_text(target, "рядок 1\nстрока 2\n")
    assert Path(target).read_bytes() == "рядок 1\nстрока 2\n".encode("utf-8")
