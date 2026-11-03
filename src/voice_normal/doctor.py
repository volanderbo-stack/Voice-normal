"""voice-normal doctor: перевірка середовища перед роботою."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voice_normal import __version__, paths, tools
from voice_normal.i18n import t

OK, WARN, FAIL, PENDING = "ok", "warn", "fail", "pending"

PYTHON_MIN = (3, 11)
PYTHON_MAX_EXCLUSIVE = (3, 14)

# Пакети ~1 ГБ + моделі ~1.5 ГБ + проміжні файли обробки (година моно 48 кГц float32
# займає ~0.7 ГБ на одну копію).
DISK_NEED_GB = 3
DISK_WANT_GB = 6

ML_PACKAGES = ("numpy", "soundfile", "torch", "torchaudio", "transformers", "onnxruntime", "faster_whisper")


@dataclass
class Check:
    id: str
    status: str
    message: str
    hints: list[str] = field(default_factory=list)
    data: dict[str, object] = field(default_factory=dict)


def check_python() -> Check:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if PYTHON_MIN <= (v.major, v.minor) < PYTHON_MAX_EXCLUSIVE:
        return Check("python", OK, t("doctor.python", version=text), data={"version": text})
    required = f">={PYTHON_MIN[0]}.{PYTHON_MIN[1]},<{PYTHON_MAX_EXCLUSIVE[0]}.{PYTHON_MAX_EXCLUSIVE[1]}"
    return Check(
        "python",
        FAIL,
        t("doctor.python.bad", version=text, required=required),
        hints=["uv python install 3.12"],
        data={"version": text},
    )


def check_os() -> Check:
    machine = platform.machine()
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        major = sys.getwindowsversion().major
        # Windows 11 досі звітує major=10; відрізняємо за номером збірки.
        name = f"Windows {'11' if build >= 22000 else major} (build {build}, {machine})"
        is_64 = machine.upper() in ("AMD64", "ARM64")
        if major < 10 or not is_64:
            return Check("os", FAIL, t("doctor.os.old", name=name), data={"build": build, "machine": machine})
        return Check("os", OK, t("doctor.os", name=name), data={"build": build, "machine": machine})
    name = f"{platform.system()} {platform.release()} ({machine})"
    return Check("os", OK, t("doctor.os", name=name), data={"machine": machine})


def check_tool(name: str) -> Check:
    path = tools.find_tool(name)
    if path is None:
        hints = tools.install_hints(name)
        if sys.platform == "win32":
            hints.append(t("doctor.tool.hint_new_terminal"))
        return Check(name, FAIL, t("doctor.tool.missing", tool=name), hints=hints)
    version = tools.tool_version(path) or "?"
    return Check(
        name,
        OK,
        t("doctor.tool.found", tool=name, version=version, path=path),
        data={"path": str(path), "version": version},
    )


def check_data_dir() -> Check:
    root = paths.data_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(dir=paths.long_path(root), prefix=".probe-")
        # Windows не дасть видалити відкритий файл — спершу закриваємо дескриптор.
        os.close(fd)
        os.remove(probe)
    except OSError as exc:
        return Check("data_dir", FAIL, t("doctor.data_dir.bad", path=root, error=exc), data={"path": str(root)})
    return Check("data_dir", OK, t("doctor.data_dir", path=root), data={"path": str(root)})


def _existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path.cwd()


def check_disk() -> Check:
    usage = shutil.disk_usage(_existing_ancestor(paths.data_dir()))
    free = usage.free / 1024**3
    data = {"free_gb": round(free, 2)}
    if free < DISK_NEED_GB:
        return Check("disk", FAIL, t("doctor.disk.too_low", free=free, need=DISK_NEED_GB), data=data)
    if free < DISK_WANT_GB:
        return Check("disk", WARN, t("doctor.disk.low", free=free, want=DISK_WANT_GB), data=data)
    return Check("disk", OK, t("doctor.disk", free=free), data=data)


def check_long_paths() -> Check | None:
    enabled = paths.long_paths_enabled()
    if enabled is None:
        return None
    if enabled:
        return Check("long_paths", OK, t("doctor.longpaths.on"), data={"enabled": True})
    return Check("long_paths", WARN, t("doctor.longpaths.off"), data={"enabled": False})


def check_gpu() -> Check:
    smi = tools.find_tool("nvidia-smi")
    if smi is not None:
        try:
            result = tools.run_tool([smi, "--query-gpu=name", "--format=csv,noheader"], timeout=15)
            name = result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else ""
        except (OSError, subprocess.TimeoutExpired, IndexError):
            name = ""
        if name:
            return Check("gpu", OK, t("doctor.gpu.nvidia", name=name), data={"device": "cuda", "name": name})
    # Відсутність GPU — не помилка: CPU є шляхом за замовчуванням.
    return Check("gpu", OK, t("doctor.gpu.none"), data={"device": "cpu"})


def check_ml_packages() -> Check:
    missing = [name for name in ML_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        return Check(
            "ml_packages", PENDING, t("doctor.ml.missing"), hints=[t("doctor.next_install")], data={"missing": missing}
        )
    return Check("ml_packages", OK, t("doctor.ml.ok", packages=", ".join(ML_PACKAGES)))


def check_models() -> Check:
    # Реєстр моделей з sha256 з'явиться на етапі 2; до того моделей просто немає.
    return Check("models", PENDING, t("doctor.models.missing"), hints=[t("doctor.next_install")])


def run_checks() -> list[Check]:
    checks = [check_python(), check_os()]
    checks += [check_tool(name) for name in tools.REQUIRED_TOOLS]
    checks += [check_data_dir(), check_disk()]
    long_paths = check_long_paths()
    if long_paths is not None:
        checks.append(long_paths)
    checks += [check_gpu(), check_ml_packages(), check_models()]
    return checks


def summarize(checks: list[Check]) -> tuple[str, int]:
    failed = [c for c in checks if c.status == FAIL]
    if failed:
        return t("doctor.summary.fail", count=len(failed)), 1
    if any(c.status == PENDING for c in checks):
        return t("doctor.summary.pending"), 0
    return t("doctor.summary.ok"), 0


_MARKS = {OK: "[OK]", WARN: "[??]", FAIL: "[!!]", PENDING: "[..]"}


def format_human(checks: list[Check], summary: str) -> str:
    # Лише ASCII-позначки: їх відобразить будь-яка кодова сторінка консолі.
    lines = [t("doctor.title", version=__version__), ""]
    for check in checks:
        lines.append(f"  {_MARKS[check.status]} {check.message}")
        lines.extend(f"         -> {hint}" for hint in check.hints)
    lines += ["", summary]
    return "\n".join(lines)


def as_json(checks: list[Check], summary: str, exit_code: int) -> dict[str, object]:
    return {
        "version": __version__,
        "ok": exit_code == 0,
        "summary": summary,
        "checks": [asdict(c) for c in checks],
    }
