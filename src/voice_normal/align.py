"""Вимірювання й компенсація зсуву в часі між відновленим і оригінальним звуком.

Sidon синтезує голос наново, тож форма хвилі з оригіналом майже не корелює. Надійно
збігаються моменти початку звуків, тому зсув шукаємо крос-кореляцією спектрального
потоку (spectral flux) з кроком 1 мс і параболічним уточненням між кроками.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, stft

SR = 48_000
HOP = 48  # 1 мс
NPERSEG = 1024


@dataclass(frozen=True)
class Offset:
    samples: float  # >0: перевірюваний сигнал запізнюється відносно еталона
    correlation: float

    @property
    def ms(self) -> float:
        return self.samples * 1000 / SR


def flux(x: np.ndarray) -> np.ndarray:
    _, _, spec = stft(x.astype(np.float32), fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    log = np.log1p(np.abs(spec) * 100.0)
    rise = np.maximum(np.diff(log, axis=1), 0.0).sum(axis=0)
    return rise - rise.mean()


def measure(test: np.ndarray, reference: np.ndarray, max_ms: float = 40.0) -> Offset:
    """Зсув test відносно reference (обидва 48 кГц, вирівняні за початком)."""
    n = min(test.size, reference.size)
    a, b = flux(test[:n]), flux(reference[:n])
    if a.size < 8 or not np.any(a) or not np.any(b):
        return Offset(0.0, 0.0)
    max_lag = int(max_ms * SR / 1000 / HOP)
    full = correlate(a, b, mode="full", method="fft")
    mid = b.size - 1
    window = full[mid - max_lag : mid + max_lag + 1]
    k = int(np.argmax(window))
    shift = 0.0
    if 0 < k < window.size - 1:
        y0, y1, y2 = window[k - 1 : k + 2]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom else 0.0
    norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
    coarse = Offset((k - max_lag + shift) * HOP, float(window[k] / norm) if norm else 0.0)
    return _refine(test[:n], reference[:n], coarse)


# Уточнення по самій хвилі має сенс, лише коли хвилі справді схожі (без Sidon, або
# Sidon на дуже чистому вході); для згенерованого голосу лишається оцінка за потоком.
WAVEFORM_MIN_CORRELATION = 0.5


def _refine(test: np.ndarray, reference: np.ndarray, coarse: Offset) -> Offset:
    lag0 = int(round(coarse.samples))
    span = 2 * HOP
    a = test.astype(np.float64)
    b = reference.astype(np.float64)
    full = correlate(a, b, mode="full", method="fft")
    mid = b.size - 1
    lo, hi = mid + lag0 - span, mid + lag0 + span + 1
    if lo < 1 or hi > full.size - 1:
        return coarse
    window = full[lo:hi]
    k = int(np.argmax(np.abs(window)))
    norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
    corr = float(window[k] / norm) if norm else 0.0
    if corr < WAVEFORM_MIN_CORRELATION:
        return coarse
    frac = 0.0
    if 0 < k < window.size - 1:
        y0, y1, y2 = window[k - 1 : k + 2]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom else 0.0
    return Offset(lag0 - span + k + frac, corr)


def shift(x: np.ndarray, samples: int) -> np.ndarray:
    """Зсунути вперед (samples > 0 — прибрати запізнення), довжина та сама."""
    if samples == 0:
        return x
    out = np.zeros_like(x)
    if samples > 0:
        out[: x.size - samples] = x[samples:]
    else:
        out[-samples:] = x[: x.size + samples]
    return out


def fit_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.size >= length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - x.size, dtype=x.dtype)])


def windowed_offsets(test: np.ndarray, reference: np.ndarray, seconds: float = 10.0) -> list[tuple[float, Offset]]:
    """Зсув по вікнах — щоб побачити, чи не "пливе" синхрон усередині файлу."""
    step = int(seconds * SR)
    result = []
    for start in range(0, min(test.size, reference.size) - step + 1, step):
        result.append((start / SR, measure(test[start : start + step], reference[start : start + step])))
    return result
