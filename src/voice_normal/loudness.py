"""Гучність за ITU-R BS.1770-4, true peak і лімітер на піки — без компресора.

Зведення лише змінює загальне підсилення до цілі; лімітер торкається тільки окремих
піків вище стелі. Остаточні цифри для звіту міряє ffmpeg (ebur128), а цей модуль —
для швидких ітерацій усередині обробки; тести звіряють їх між собою.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter, resample_poly

SR = 48_000
# Коефіцієнти K-фільтра для 48 кГц з BS.1770-4, табл. 1 і 2.
_SHELF_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_SHELF_A = [1.0, -1.69065929318241, 0.73248077421585]
_HP_B = [1.0, -2.0, 1.0]
_HP_A = [1.0, -1.99004745483398, 0.99007225036621]

TRUE_PEAK_OVERSAMPLE = 4
# Запас під стелею: квантування в 24 біти й різниця між вимірювачами.
CEILING_MARGIN_DB = 0.5


def integrated_lufs(x: np.ndarray) -> float:
    """Інтегральна гучність моно-сигналу 48 кГц з абсолютним і відносним гейтом."""
    y = lfilter(_HP_B, _HP_A, lfilter(_SHELF_B, _SHELF_A, x.astype(np.float64)))
    block, step = int(0.4 * SR), int(0.1 * SR)
    if y.size < block:
        return -np.inf
    power = np.square(y)
    cumsum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, y.size - block + 1, step)
    z = (cumsum[starts + block] - cumsum[starts]) / block
    loud = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    z = z[loud > -70.0]
    if z.size == 0:
        return -np.inf
    relative = -0.691 + 10 * np.log10(np.mean(z)) - 10.0
    z = z[-0.691 + 10 * np.log10(z) > relative]
    return float(-0.691 + 10 * np.log10(np.mean(z)))


def true_peak_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(_oversampled(x)))) if x.size else 0.0
    return 20 * np.log10(max(peak, 1e-12))


def _oversampled(x: np.ndarray) -> np.ndarray:
    return resample_poly(x.astype(np.float64), TRUE_PEAK_OVERSAMPLE, 1)


def peak_limit(
    x: np.ndarray, ceiling_db: float, lookahead_ms: float = 5.0
) -> tuple[np.ndarray, float, float]:
    """Лімітер без затримки: крива підсилення рахується наперед по всьому сигналу.

    Ковзний мінімум потрібного підсилення з подальшим ковзним середнім тієї ж ширини
    гарантує, що в точці піку підсилення не більше потрібного, а зміни плавні (атака й
    відпускання по lookahead_ms). Повертає сигнал, найбільше зниження в дБ і частку часу
    (у відсотках), коли зниження перевищує 1 дБ — щоб відрізнити окремі піки від
    постійного стискання.
    """
    ceiling = 10 ** (ceiling_db / 20)
    over = np.abs(_oversampled(x))
    n = x.size
    peaks = over[: n * TRUE_PEAK_OVERSAMPLE].reshape(n, TRUE_PEAK_OVERSAMPLE).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if need.min() >= 1.0:
        return x, 0.0, 0.0
    half = max(1, int(lookahead_ms * SR / 1000))
    gain = uniform_filter1d(minimum_filter1d(need, 2 * half + 1, mode="nearest"), 2 * half + 1, mode="nearest")
    gain = np.minimum(gain, need)
    busy = float(100 * np.mean(gain < 10 ** (-1 / 20)))
    return (x * gain).astype(np.float32), float(-20 * np.log10(gain.min())), busy


def master(
    x: np.ndarray, target_lufs: float, ceiling_dbtp: float, iterations: int = 5
) -> tuple[np.ndarray, dict[str, float]]:
    """Підсилення до target_lufs і лімітер під ceiling_dbtp; повторюємо, бо лімітер трохи
    знижує гучність."""
    internal_ceiling = ceiling_dbtp - CEILING_MARGIN_DB
    source = x.astype(np.float64)
    start_lufs = integrated_lufs(source)
    if not np.isfinite(start_lufs):
        return x.astype(np.float32), {"gain_db": 0.0, "limiter_db": 0.0, "lufs": start_lufs}
    gain_db = target_lufs - start_lufs
    out, reduction, busy = source, 0.0, 0.0
    for _ in range(iterations):
        out, reduction, busy = peak_limit((source * 10 ** (gain_db / 20)).astype(np.float32), internal_ceiling)
        error = target_lufs - integrated_lufs(out)
        if abs(error) < 0.05:
            break
        gain_db += error
    return out.astype(np.float32), {
        "gain_db": round(float(gain_db), 2),
        "limiter_db": round(reduction, 2),
        "limited_percent": round(busy, 3),
        "lufs": round(integrated_lufs(out), 2),
        "true_peak_dbtp": round(float(true_peak_db(out)), 2),
    }
