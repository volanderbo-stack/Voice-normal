"""Відновлення довгого запису Sidon'ом фрагментами з точним складанням назад.

Фрагменти ~30 с ріжемо в паузах між фразами: дві обробки того самого місця з різним
контекстом дають помітно різні хвилі, тож шов посеред слова було б чути. Кожен
фрагмент подається з контекстом MARGIN_S з обох боків, його зсув у часі вимірюється
окремо, а склеювання — коротким косинусним переходом точно на межі.

Позиції рахуються в семплах 16 кГц і множаться на 3 для 48 кГц, тож межі точні до семпла.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from voice_normal import align

SR16, SR48, RATIO = 16_000, 48_000, 3
CHUNK_S = 30.0
SEARCH_S = 5.0
MARGIN_S = 1.5
CROSSFADE_MS = 20.0
MIN_ALIGN_CORRELATION = 0.3
# Типова затримка Sidon v0.1 (мс), заміряна на чистій мові; потрібна лише для
# фрагментів, де мови замало для власного вимірювання.
DEFAULT_LATENCY_SAMPLES = 300
FRAME16 = 160  # кадр маски мови — 10 мс


@dataclass
class Chunk:
    start: int  # межі в семплах 16 кГц
    end: int
    cut_in_silence: bool
    offset_samples: int = 0
    offset_measured: bool = False
    correlation: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_s"], d["end_s"] = self.start / SR16, self.end / SR16
        return d


def plan_chunks(n16: int, speech: np.ndarray, chunk_s: float = CHUNK_S) -> list[Chunk]:
    """Межі фрагментів посередині найдовшої паузи біля кожної кратної chunk_s позиції."""
    cuts, flags = [0], [True]
    target = chunk_s
    total_s = n16 / SR16
    while target < total_s - chunk_s / 3:
        lo = int((target - SEARCH_S) * SR16 / FRAME16)
        hi = min(int((target + SEARCH_S) * SR16 / FRAME16), speech.size)
        best = _longest_silence(speech, max(lo, 0), hi, int(target * SR16 / FRAME16))
        if best is None:
            cuts.append(int(target * SR16))
            flags.append(False)
        else:
            cuts.append(int(best) * FRAME16)
            flags.append(True)
        target = cuts[-1] / SR16 + chunk_s
    cuts.append(n16)
    return [Chunk(cuts[i], cuts[i + 1], flags[i + 1] if i + 1 < len(cuts) - 1 else True) for i in range(len(cuts) - 1)]


def _longest_silence(speech: np.ndarray, lo: int, hi: int, center: int) -> int | None:
    silent = ~speech[lo:hi]
    if not silent.any():
        return None
    edges = np.diff(np.concatenate([[0], silent.astype(np.int8), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    lengths = ends - starts
    # Найдовша пауза, при рівних — ближча до цілі.
    order = sorted(range(lengths.size), key=lambda i: (-lengths[i], abs(lo + (starts[i] + ends[i]) // 2 - center)))
    i = order[0]
    if lengths[i] < 3:  # коротше за 30 мс — це не пауза
        return None
    return int(lo + (starts[i] + ends[i]) // 2)


def restore(
    x48: np.ndarray,
    speech: np.ndarray,
    model,
    chunks: list[Chunk] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[np.ndarray, list[Chunk]]:
    """x48 — моно 48 кГц; speech — маска мови по 10 мс; model — об'єкт із методом restore()."""
    from scipy.signal import resample_poly

    n48 = x48.size
    peak = float(np.max(np.abs(x48))) or 1.0
    # Як в офіційному демо: вхід Sidon нормуємо за піком один раз на весь файл —
    # нормування по фрагментах дало б стрибки рівня на швах.
    scale = 0.9 / peak
    x16 = resample_poly(x48.astype(np.float64) * scale, 1, RATIO).astype(np.float32)
    chunks = chunks or plan_chunks(x16.size, speech)
    margin = int(MARGIN_S * SR16)
    fade = int(CROSSFADE_MS * SR48 / 1000) // 2 * 2
    out = np.zeros(n48, dtype=np.float64)
    weight = np.zeros(n48, dtype=np.float64)
    measured: list[int] = []

    for index, chunk in enumerate(chunks):
        started = time.perf_counter()
        a, b = max(0, chunk.start - margin), min(x16.size, chunk.end + margin)
        y = model.restore(x16[a:b]).astype(np.float64) / scale
        reference = x48[a * RATIO : min(b * RATIO, n48)]
        y = align.fit_length(y, reference.size)
        offset = align.measure(y, reference)
        if offset.correlation >= MIN_ALIGN_CORRELATION:
            chunk.offset_samples, chunk.offset_measured = int(round(offset.samples)), True
            measured.append(chunk.offset_samples)
        else:
            chunk.offset_samples = int(np.median(measured)) if measured else DEFAULT_LATENCY_SAMPLES
        chunk.correlation = round(offset.correlation, 3)
        y = align.shift(y, chunk.offset_samples)

        # Вага: 1 усередині фрагмента, косинусні переходи шириною fade навколо меж.
        lo, hi = chunk.start * RATIO, min(chunk.end * RATIO, n48)
        w = np.zeros(reference.size)
        core_lo, core_hi = lo - a * RATIO, hi - a * RATIO
        w[max(core_lo - fade // 2, 0) : min(core_hi + fade // 2, w.size)] = 1.0
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
        if chunk.start > 0:
            w[core_lo - fade // 2 : core_lo + fade // 2] = ramp
        if chunk.end < x16.size:
            w[core_hi - fade // 2 : core_hi + fade // 2] = ramp[::-1]
        span = slice(a * RATIO, a * RATIO + reference.size)
        out[span] += w * y
        weight[span] += w
        chunk.seconds = round(time.perf_counter() - started, 2)
        if progress is not None:
            progress(index + 1, len(chunks), chunk.seconds)

    covered = weight > 0
    out[covered] /= weight[covered]
    return out.astype(np.float32), chunks
