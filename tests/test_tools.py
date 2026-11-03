import json
import sys

from voice_normal import tools


def test_windows_hints_use_winget():
    assert tools.install_hints("ffmpeg", "win32") == ["winget install Gyan.FFmpeg"]
    assert tools.install_hints("uv", "win32")[0] == "winget install astral-sh.uv"
    assert "install.ps1" in tools.install_hints("uv", "win32")[1]


def test_arguments_pass_through_without_shell(tmp_path):
    # Пробіли, кирилиця й символи, які shell інтерпретував би, мають дійти як є.
    tricky = ["камера запис.wav", "a&b", "50%", 'кав"ички', str(tmp_path / "Тест папка" / "x.wav")]
    script = "import json, sys; sys.stdout.reconfigure(encoding='utf-8'); print(json.dumps(sys.argv[1:]))"
    result = tools.run_tool([sys.executable, "-c", script, *tricky])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == tricky


def test_missing_tool_is_none():
    assert tools.find_tool("definitely-not-a-real-tool-xyz") is None
