"""Завантаження файлів моделей з докачуванням і перевіркою sha256.

Лише стандартна бібліотека: install має працювати ще до того, як поставлено будь-які
пакети. Недокачаний файл лежить поруч як <ім'я>.part і продовжується з місця обриву
(HTTP Range); у робоче ім'я файл потрапляє тільки після збігу sha256.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voice_normal import __version__
from voice_normal.paths import long_path

CHUNK = 1 << 20
USER_AGENT = f"voice-normal/{__version__}"

Progress = Callable[[int, int | None], None]


class DownloadError(Exception):
    pass


class ChecksumError(DownloadError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def download(
    url: str,
    dest: Path,
    sha256: str,
    size: int | None = None,
    progress: Progress | None = None,
    retries: int = 4,
    timeout: float = 60,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(dest)
    for attempt in range(retries + 1):
        try:
            _fetch(url, part, size, progress, timeout)
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            if attempt == retries:
                raise DownloadError(f"{url}: {exc}") from exc
            # Мережа на ноутбуках часто "моргає": чекаємо й докачуємо з того ж місця.
            time.sleep(min(2**attempt, 15))

    actual = sha256_file(part)
    if actual != sha256:
        os.remove(long_path(part))
        raise ChecksumError(f"{dest.name}: sha256 {actual} != {sha256}")
    # os.replace атомарний і на Windows перезаписує ціль; обидва файли вже закриті.
    os.replace(long_path(part), long_path(dest))


def _fetch(url: str, part: Path, size: int | None, progress: Progress | None, timeout: float) -> None:
    have = os.path.getsize(long_path(part)) if os.path.exists(long_path(part)) else 0
    if size is not None and have > size:
        os.remove(long_path(part))
        have = 0
    if size is not None and have == size:
        return

    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            # Сервер каже, що такого діапазону немає — .part зіпсований, починаємо заново.
            os.remove(long_path(part))
        raise

    with response:
        if have and response.status != 206:
            # Сервер проігнорував Range і віддає файл з початку.
            have = 0
        total = size
        if total is None:
            length = response.headers.get("Content-Length")
            total = have + int(length) if length else None
        mode = "ab" if have else "wb"
        with open(long_path(part), mode) as fh:
            while block := response.read(CHUNK):
                fh.write(block)
                have += len(block)
                if progress is not None:
                    progress(have, total)
    if size is not None and have != size:
        raise http.client.IncompleteRead(b"", size - have)
