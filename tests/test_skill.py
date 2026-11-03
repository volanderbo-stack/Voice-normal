import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from voice_normal import __version__

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "voice-normal" / "SKILL.md"


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_skill_frontmatter():
    meta = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert meta["name"] == "voice-normal" == SKILL.parent.name
    # Agent Skills: name — до 64 символів [a-z0-9-], description — до 1024.
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    assert 50 < len(meta["description"]) <= 1024


def test_skill_mentions_every_command_and_exit_code():
    text = SKILL.read_text(encoding="utf-8")
    for command in ("doctor", "install", "plan", "process"):
        assert f"{command}" in text
    for code in ("`0`", "`1`", "`2`", "`3`", "`130`"):
        assert code in text
    assert "voice-normal.cmd" in text and "${CLAUDE_PLUGIN_ROOT}" in text and "repo-root.txt" in text


def test_plugin_and_marketplace_manifests_agree():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["version"] == __version__
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", plugin["name"])
    assert not plugin["name"].startswith(("claude-", "anthropic-"))
    assert {"name", "owner", "plugins"} <= set(market) and market["owner"]["name"]
    (entry,) = market["plugins"]
    # Ім'я в маркетплейсі має збігатися з ім'ям у plugin.json, інакше install не знайде плагін.
    assert entry["name"] == plugin["name"]
    assert entry["source"] == "./" and ".." not in entry["source"]
    assert (ROOT / "skills" / plugin["name"] / "SKILL.md").is_file()


# ---------- install-skill.ps1 ----------

windows = pytest.mark.skipif(sys.platform != "win32", reason="install-skill.ps1 — для Windows")


def _install(user_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(ROOT / "scripts" / "install-skill.ps1"), "-UserDir", str(user_dir), *args],
        capture_output=True, text=True, check=False,
    )


@pytest.fixture
def user_dir(tmp_path):
    yield tmp_path / "Профіль з пробілами"
    marker = SKILL.parent / "repo-root.txt"
    # Не лишаємо після тестів посилань на тимчасову папку.
    if marker.exists() and not (Path.home() / ".claude" / "skills" / "voice-normal").exists():
        marker.unlink()


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


@windows
def test_junction_install_rerun_and_uninstall(user_dir):
    result = _install(user_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    for agent in (".claude", ".agents"):
        target = user_dir / agent / "skills" / "voice-normal"
        assert _is_link(target)
        assert (target / "SKILL.md").read_text(encoding="utf-8") == SKILL.read_text(encoding="utf-8")
        assert Path((target / "repo-root.txt").read_text(encoding="utf-8").strip()) == ROOT
    assert _install(user_dir).returncode == 0  # повторний запуск нічого не ламає

    result = _install(user_dir, "-Uninstall")
    assert result.returncode == 0, result.stdout
    assert not (user_dir / ".claude" / "skills" / "voice-normal").exists()
    # Видалення посилання не зачепило файли в репозиторії.
    assert SKILL.is_file()


@windows
def test_copy_mode(user_dir):
    result = _install(user_dir, "-Mode", "copy", "-Agent", "codex")
    assert result.returncode == 0, result.stdout + result.stderr
    target = user_dir / ".agents" / "skills" / "voice-normal"
    assert target.is_dir() and not _is_link(target)
    assert (target / "SKILL.md").is_file()
    assert Path((target / "repo-root.txt").read_text(encoding="utf-8").strip()) == ROOT
    assert not (user_dir / ".claude").exists()
    assert _install(user_dir, "-Uninstall", "-Agent", "codex").returncode == 0
    assert not target.exists()


@windows
def test_refuses_to_overwrite_someone_elses_skill(user_dir):
    foreign = user_dir / ".claude" / "skills" / "voice-normal"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("чужий skill", encoding="utf-8")
    result = _install(user_dir, "-Agent", "claude")
    assert result.returncode == 1
    assert (foreign / "SKILL.md").read_text(encoding="utf-8") == "чужий skill"
    shutil.rmtree(foreign)
