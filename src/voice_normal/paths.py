"""Де лежать дані, моделі й тимчасові файли.

Windows: %LOCALAPPDATA%\\voice-normal. macOS/Linux мають розумні запасні варіанти,
але основна платформа — Windows. Змінна VOICE_NORMAL_HOME перекриває все (для тестів і CI).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

APP_NAME = "voice-normal"

# MAX_PATH у Win32 — 260 символів разом із нульовим символом; каталоги обмежені ще
# сильніше (248), бо до них дописується ім'я файлу 8.3.
_WIN_MAX_DIR = 248


def data_dir() -> Path:
    override = os.environ.get("VOICE_NORMAL_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / APP_NAME


def models_dir() -> Path:
    return data_dir() / "models"


def make_temp_dir(prefix: str = "voice-normal-") -> Path:
    """Окрема тимчасова папка на один запуск; прибирає її той, хто створив."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def long_path(path: Path) -> str:
    """Рядок шляху, який Win32 відкриє навіть якщо він довший за 260 символів.

    Префікс \\\\?\\ вимикає ліміт MAX_PATH незалежно від LongPathsEnabled у реєстрі,
    але вимагає абсолютного шляху з бекслешами. На інших ОС повертає шлях як є.
    """
    if sys.platform != "win32":
        return str(path)
    text = str(Path(path).absolute())
    if text.startswith("\\\\?\\") or len(text) < _WIN_MAX_DIR:
        return text
    if text.startswith("\\\\"):
        # Мережевий шлях \\server\share\... має окрему форму префікса.
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def long_paths_enabled() -> bool | None:
    """Чи ввімкнено LongPathsEnabled у Windows; None, якщо не Windows або невідомо.

    Лише читаємо реєстр — змінювати системні налаштування ми не будемо.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except OSError:
        return None
