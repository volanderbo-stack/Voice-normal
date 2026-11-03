"""voice-normal install: ML-пакети, моделі й перевірка, що все це запускається."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voice_normal import download, models, paths, tools
from voice_normal.i18n import t

MB = 1024**2
DISK_RESERVE = 512 * MB


@dataclass
class InstallResult:
    ok: bool = True
    device: str = "cpu"
    packages: str = "skipped"
    models: list[dict[str, object]] = field(default_factory=list)
    smoke: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None


def project_root() -> Path | None:
    """Корінь репозиторію, якщо пакет запущено з git clone (editable-встановлення)."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "voice_normal").is_dir():
        return root
    return None


def detect_device() -> str:
    smi = tools.find_tool("nvidia-smi")
    if smi is None:
        return "cpu"
    try:
        result = tools.run_tool([smi, "--query-gpu=name", "--format=csv,noheader"], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"
    return "cuda" if result.returncode == 0 and result.stdout.strip() else "cpu"


def sync_packages(device: str, root: Path) -> int:
    uv = tools.find_tool("uv")
    if uv is None:
        raise RuntimeError(t("doctor.tool.missing", tool="uv"))
    # --inexact: нічого не видаляємо з оточення; --no-dev: користувачам не потрібен pytest.
    # --no-install-project: сам voice-normal уже запущений з цього оточення, а Windows
    # не дає перезаписати .exe, що виконується.
    args = [
        uv, "sync", "--project", root, "--extra", "ml", "--extra", device,
        "--inexact", "--no-dev", "--no-install-project",
    ]
    # Вивід uv іде в stderr, щоб stdout лишався чистим JSON для агентів.
    return subprocess.run([str(a) for a in args], stdout=sys.stderr, stderr=sys.stderr, check=False).returncode


class _ProgressPrinter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.tty = sys.stderr.isatty()
        self.last = 0.0

    def __call__(self, done: int, total: int | None) -> None:
        if not self.tty:
            return
        now = time.monotonic()
        if now - self.last < 0.3 and (total is None or done < total):
            return
        self.last = now
        tail = f"{done / MB:7.1f}/{total / MB:.1f} MB {100 * done // total:3d}%" if total else f"{done / MB:7.1f} MB"
        sys.stderr.write(f"\r    {self.label:<44} {tail}")
        sys.stderr.flush()

    def finish(self) -> None:
        if self.tty:
            sys.stderr.write("\r" + " " * 80 + "\r")
            sys.stderr.flush()


def _free_bytes(path: Path) -> int:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free
    return shutil.disk_usage(Path.cwd()).free


def ensure_models(device: str, verify: bool, log) -> list[dict[str, object]]:
    root = paths.models_dir()
    root.mkdir(parents=True, exist_ok=True)
    plan = models.required_files(device)

    still_needed = 0
    for model, file in plan:
        target = models.file_path(model, file)
        if os.path.exists(paths.long_path(target)):
            continue
        part = download.part_path(target)
        have = os.path.getsize(paths.long_path(part)) if os.path.exists(paths.long_path(part)) else 0
        still_needed += file.size - have
    free = _free_bytes(root)
    if still_needed + DISK_RESERVE > free:
        raise RuntimeError(t("install.disk", need=(still_needed + DISK_RESERVE) / 1024**3, free=free / 1024**3))

    results = []
    for model, file in plan:
        target = models.file_path(model, file)
        label = f"{model.id}/{file.name}"
        status = "present"
        if os.path.exists(paths.long_path(target)):
            wrong_size = os.path.getsize(paths.long_path(target)) != file.size
            if wrong_size or (verify and download.sha256_file(target) != file.sha256):
                os.remove(paths.long_path(target))
            elif verify:
                status = "verified"
        if not os.path.exists(paths.long_path(target)):
            log(t("install.downloading", name=label, size=file.size / MB))
            printer = _ProgressPrinter(label)
            try:
                download.download(file.url, target, file.sha256, file.size, progress=printer)
            finally:
                printer.finish()
            status = "downloaded"
        results.append({"file": label, "status": status, "size": file.size})
    models.write_manifest(device)
    return results


def run_smoke(device: str) -> list[dict[str, object]]:
    """Перевірка в окремому процесі: щойно встановлені пакети вантажаться начисто."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-m", "voice_normal.smoke", device],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    try:
        return json.loads(result.stdout)
    except ValueError:
        detail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        return [{"name": "smoke", "ok": False, "detail": " | ".join(detail)}]


def run_install(requested_device: str, verify: bool, skip_packages: bool, log) -> InstallResult:
    device = detect_device() if requested_device == "auto" else requested_device
    result = InstallResult(device=device)
    log(t("install.device." + device))

    root = project_root()
    if skip_packages:
        log(t("install.packages.skipped"))
    elif root is None:
        result.packages = "manual"
        log(t("install.packages.manual", device=device))
    else:
        log(t("install.step.packages"))
        code = sync_packages(device, root)
        if code != 0:
            result.ok, result.packages, result.error = False, "failed", t("install.packages.failed", code=code)
            return result
        result.packages = "installed"

    log(t("install.step.models", size=models.total_size(device) / 1024**3, path=paths.models_dir()))
    result.models = ensure_models(device, verify, log)

    log(t("install.step.smoke"))
    result.smoke = run_smoke(device)
    for item in result.smoke:
        log(f"    [{'OK' if item['ok'] else '!!'}] {item['name']}: {item['detail']}")
    result.ok = all(item["ok"] for item in result.smoke)
    if not result.ok:
        result.error = t("install.smoke.failed")
    return result


def as_json(result: InstallResult) -> dict[str, object]:
    return asdict(result)
