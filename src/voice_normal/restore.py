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


def recut(chunks: list[Chunk], seam: int, speech: np.ndarray) -> list[Chunk] | None:
    """Перенести межу між фрагментами seam-1 і seam в іншу паузу (для ремонту шва)."""
    prev, cur = chunks[seam - 1], chunks[seam]
    old = cur.start // FRAME16
    lo = (prev.start + int(SEARCH_S * SR16)) // FRAME16
    hi = (cur.end - int(SEARCH_S * SR16)) // FRAME16
    if hi - lo < 10:
        return None
    candidates = speech.copy()
    # Не ближче 1.5 с до старої межі — інакше вийде той самий шов.
    keep_out = int(1.5 * SR16) // FRAME16
    candidates[max(old - keep_out, 0) : old + keep_out] = True
    best = _longest_silence(candidates, lo, hi, old)
    if best is None:
        return None
    moved = list(chunks)
    cut = int(best) * FRAME16
    moved[seam - 1] = Chunk(prev.start, cut, prev.cut_in_silence)
    moved[seam] = Chunk(cut, cur.end, True)
    return moved


class Restorer:
    """Відновлення одного запису: усе разом або лише вибрані фрагменти (для ремонту)."""

    def __init__(self, x48: np.ndarray, model):
        from scipy.signal import resample_poly

        self.x48 = x48
        self.model = model
        peak = float(np.max(np.abs(x48))) or 1.0
        # Як в офіційному демо: вхід Sidon нормуємо за піком один раз на весь файл —
        # нормування по фрагментах дало б стрибки рівня на швах.
        self.scale = 0.9 / peak
        self.x16 = resample_poly(x48.astype(np.float64) * self.scale, 1, RATIO).astype(np.float32)
        self.fade = int(CROSSFADE_MS * SR48 / 1000) // 2 * 2
        self.measured: list[int] = []

    def _render(self, chunk: Chunk, margin_s: float) -> tuple[slice, np.ndarray, np.ndarray]:
        """Відновлений фрагмент із компенсованим зсувом і його вага для склеювання."""
        n48, margin = self.x48.size, int(margin_s * SR16)
        started = time.perf_counter()
        a, b = max(0, chunk.start - margin), min(self.x16.size, chunk.end + margin)
        y = self.model.restore(self.x16[a:b]).astype(np.float64) / self.scale
        reference = self.x48[a * RATIO : min(b * RATIO, n48)]
        y = align.fit_length(y, reference.size)
        offset = align.measure(y, reference)
        if offset.correlation >= MIN_ALIGN_CORRELATION:
            chunk.offset_samples, chunk.offset_measured = int(round(offset.samples)), True
            self.measured.append(chunk.offset_samples)
        else:
            chunk.offset_samples = int(np.median(self.measured)) if self.measured else DEFAULT_LATENCY_SAMPLES
            chunk.offset_measured = False
        chunk.correlation = round(offset.correlation, 3)
        y = align.shift(y, chunk.offset_samples)

        # Вага: 1 усередині фрагмента, косинусні переходи шириною fade навколо меж.
        fade = self.fade
        w = np.zeros(reference.size)
        core_lo, core_hi = chunk.start * RATIO - a * RATIO, min(chunk.end * RATIO, n48) - a * RATIO
        w[max(core_lo - fade // 2, 0) : min(core_hi + fade // 2, w.size)] = 1.0
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
        if chunk.start > 0:
            w[core_lo - fade // 2 : core_lo + fade // 2] = ramp
        if chunk.end < self.x16.size:
            w[core_hi - fade // 2 : core_hi + fade // 2] = ramp[::-1]
        chunk.seconds = round(time.perf_counter() - started, 2)
        return slice(a * RATIO, a * RATIO + reference.size), y, w

    def render(
        self,
        chunks: list[Chunk],
        margin_s: float = MARGIN_S,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        out = np.zeros(self.x48.size)
        weight = np.zeros(self.x48.size)
        for index, chunk in enumerate(chunks):
            span, y, w = self._render(chunk, margin_s)
            out[span] += w * y
            weight[span] += w
            if progress is not None:
                progress(index + 1, len(chunks), chunk.seconds)
        covered = weight > 0
        out[covered] /= weight[covered]
        return out.astype(np.float32)

    def redo(
        self,
        previous: np.ndarray,
        chunks: list[Chunk],
        indices: list[int],
        margin_s: float,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        """Відновити заново лише фрагменти indices і вклеїти їх у previous.

        Зовнішні межі вибраних фрагментів — ті самі паузи, що й раніше, тож новий шматок
        входить у старий результат тим самим косинусним переходом.
        """
        new = np.zeros(self.x48.size)
        weight = np.zeros(self.x48.size)
        for done, index in enumerate(indices, 1):
            span, y, w = self._render(chunks[index], margin_s)
            new[span] += w * y
            weight[span] += w
            if progress is not None:
                progress(done, len(indices), chunks[index].seconds)
        weight = np.minimum(weight, 1.0)
        return (previous.astype(np.float64) * (1.0 - weight) + new).astype(np.float32)


def restore(
    x48: np.ndarray,
    speech: np.ndarray,
    model,
    chunks: list[Chunk] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[np.ndarray, list[Chunk]]:
    """x48 — моно 48 кГц; speech — маска мови по 10 мс; model — об'єкт із методом restore()."""
    restorer = Restorer(x48, model)
    chunks = chunks or plan_chunks(restorer.x16.size, speech)
    return restorer.render(chunks, progress=progress), chunks
