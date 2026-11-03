import json
import os
import subprocess
import sys

import pytest

from voice_normal import cli, doctor, tools


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    home = tmp_path / "Дані з пробілами"
    monkeypatch.setenv("VOICE_NORMAL_HOME", str(home))
    return home


def _run_json(capsys, *argv):
    code = cli.main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_doctor_json(capsys, isolated_home):
    code, payload = _run_json(capsys, "doctor")
    ids = [c["id"] for c in payload["checks"]]
    assert {"python", "os", "ffmpeg", "ffprobe", "uv", "data_dir", "disk", "gpu"} <= set(ids)
    assert isolated_home.is_dir()
    assert code == (0 if payload["ok"] else 1)


@pytest.mark.parametrize("argv", [["--lang", "ru", "doctor"], ["doctor", "--lang", "ru"]])
def test_lang_before_or_after_command(capsys, argv):
    _, payload = _run_json(capsys, *argv)
    data_dir = next(c for c in payload["checks"] if c["id"] == "data_dir")
    assert data_dir["message"].startswith("Папка данных")


def test_missing_ffmpeg_fails_with_winget_hint(capsys, monkeypatch):
    real = tools.find_tool
    monkeypatch.setattr(tools, "find_tool", lambda name: None if name == "ffmpeg" else real(name))
    monkeypatch.setattr(sys, "platform", "win32")
    check = doctor.check_tool("ffmpeg")
    assert check.status == doctor.FAIL
    assert "winget install Gyan.FFmpeg" in check.hints


def test_no_gpu_is_not_an_error(monkeypatch):
    real = tools.find_tool
    monkeypatch.setattr(tools, "find_tool", lambda name: None if name == "nvidia-smi" else real(name))
    check = doctor.check_gpu()
    assert check.status == doctor.OK and check.data["device"] == "cpu"


def test_missing_input_is_usage_error(capsys, tmp_path):
    code, payload = _run_json(capsys, "process", str(tmp_path / "немає такого.wav"))
    assert code == cli.EXIT_USAGE and payload["ok"] is False


def test_unfinished_commands_say_so(capsys, tmp_path):
    source = tmp_path / "запис камера.wav"
    source.write_bytes(b"")
    code, _ = _run_json(capsys, "process", str(source))
    assert code == cli.EXIT_NOT_READY


def _module_run(cwd, env_extra):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "voice_normal", "doctor", "--json"],
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
    )


def test_subprocess_from_cyrillic_cwd_utf8(tmp_path):
    cwd = tmp_path / "Тест папка"
    cwd.mkdir()
    result = _module_run(cwd, {"PYTHONIOENCODING": "utf-8", "VOICE_NORMAL_LANG": "uk"})
    payload = json.loads(result.stdout.decode("utf-8"))
    assert result.returncode in (0, 1)
    assert any("Папка даних" in c["message"] for c in payload["checks"])


def test_subprocess_with_legacy_codepage_does_not_crash(tmp_path):
    # Без PYTHONIOENCODING pipe отримує кодову сторінку ANSI (на CI це cp1252 без
    # кирилиці). Символи можуть замінитися на "?", але процес не має впасти.
    result = _module_run(tmp_path, {"VOICE_NORMAL_LANG": "uk"})
    assert result.returncode in (0, 1), result.stderr
    assert b"UnicodeEncodeError" not in result.stderr
