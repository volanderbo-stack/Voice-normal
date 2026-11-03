"""Безпечний текстовий вивід у консолі Windows.

У справжній консолі Python пише через WriteConsoleW і кирилиця проходить завжди.
Проблеми починаються, коли вивід перенаправлено в pipe чи файл: тоді діє кодова
сторінка cp1251/cp866, і будь-який символ поза нею кидає UnicodeEncodeError.
Тому замінюємо такі символи, а не падаємо. Обгортки .cmd/.ps1 додатково вмикають
UTF-8 через PYTHONIOENCODING.
"""

from __future__ import annotations

import sys


def setup_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _json_default(value):
    # numpy-скаляри й масиви (float32, int64, ndarray) -> звичайні типи Python.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def dumps(data) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n"


def write_text(path, text: str) -> None:
    """Звіти завжди в UTF-8 з LF — однаково на всіх ОС."""
    from voice_normal.paths import long_path

    with open(long_path(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
