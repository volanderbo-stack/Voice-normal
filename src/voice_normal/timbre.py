"""Підбір тембру: повернути відновленому голосу забарвлення самого мовця.

Sidon тягне голос до "усередненого" студійного звучання. Порівнюємо довгостроковий
спектр мови в оригіналі (з відніманням спектра шуму з пауз) і після відновлення та
коригуємо різницю лише частково й у вузьких межах: повністю копіювати оригінал не
можна, бо він зіпсований мікрофоном і кімнатою. Фільтр лінійно-фазовий і симетричний,
тож затримки немає і синхрон не змінюється.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, welch

SR = 48_000
NPERSEG = 4096
STRENGTH = 0.5
MAX_DB = 3.0
LOW_HZ = 80.0
HIGH_LIMIT_HZ = 12_000.0
TAPS = 2047
FRAME48 = 480  # кадр маски мови — 10 мс


def _regions(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    keep = np.repeat(mask, FRAME48)[: x.size]
    keep = np.concatenate([keep, np.zeros(x.size - keep.size, dtype=bool)])
    return x[keep]


def _psd(x: np.ndarray) -> np.ndarray | None:
    if x.size < NPERSEG * 4:
        return None
    return welch(x.astype(np.float64), fs=SR, nperseg=NPERSEG)[1]


def _third_octave_smooth(freqs: np.ndarray, db: np.ndarray) -> np.ndarray:
    out = np.empty_like(db)
    for i, f in enumerate(freqs):
        if f <= 0:
            out[i] = db[i]
            continue
        sel = (freqs >= f / 2 ** (1 / 6)) & (freqs <= f * 2 ** (1 / 6))
        out[i] = db[sel].mean()
    return out


def correction_curve(restored: np.ndarray, original: np.ndarray, speech: np.ndarray, bandwidth_hz: float):
    p_rest = _psd(_regions(restored, speech))
    p_orig = _psd(_regions(original, speech))
    p_noise = _psd(_regions(original, ~speech))
    if p_rest is None or p_orig is None:
        return None, None
    freqs = np.fft.rfftfreq(NPERSEG, 1 / SR)
    noise = p_noise if p_noise is not None else np.zeros_like(p_orig)
    clean = np.maximum(p_orig - noise, 0.05 * p_orig)
    diff = _third_octave_smooth(freqs, 10 * np.log10(np.maximum(clean, 1e-20) / np.maximum(p_rest, 1e-20)))
    # Довіряємо оригіналу лише там, де мова помітно над шумом: від SNR 3 дБ вага 0,
    # від 12 дБ — повна. Інакше "тембр" оригіналу — це просто його шум.
    band_snr = _third_octave_smooth(freqs, 10 * np.log10(np.maximum(clean, 1e-20) / np.maximum(noise, 1e-20)))
    reliable = np.clip((band_snr - 3.0) / 9.0, 0.0, 1.0)
    # Загальний рівень не чіпаємо — його задасть зведення.
    core = (freqs >= 300) & (freqs <= 3000)
    diff -= np.average(diff[core], weights=reliable[core] + 1e-6)
    diff = np.clip(STRENGTH * diff * reliable, -MAX_DB, MAX_DB)
    high = min(0.9 * bandwidth_hz, HIGH_LIMIT_HZ)
    # Плавно до нуля за третину октави від меж: поза смугою оригінал нічого не знає про тембр.
    taper = np.clip(np.log2(np.maximum(freqs, 1) / LOW_HZ) * 3, 0, 1) * np.clip(np.log2(high / np.maximum(freqs, 1)) * 3, 0, 1)
    return freqs, diff * taper


def apply_curve(x: np.ndarray, freqs: np.ndarray, curve_db: np.ndarray) -> np.ndarray:
    grid = np.fft.rfftfreq(TAPS + 1, 1 / SR)
    magnitude = 10 ** (np.interp(grid, freqs, curve_db) / 20)
    kernel = np.fft.irfft(magnitude, n=TAPS + 1)
    kernel = np.roll(kernel, TAPS // 2)[:TAPS] * np.hanning(TAPS)
    kernel /= kernel.sum() / magnitude[0] if kernel.sum() else 1.0
    # Непарна симетрична довжина + mode="same" = нульова фаза, жодного зсуву.
    return fftconvolve(x.astype(np.float64), kernel, mode="same").astype(np.float32)


def match(restored: np.ndarray, original: np.ndarray, speech: np.ndarray, bandwidth_hz: float):
    freqs, curve = correction_curve(restored, original, speech, bandwidth_hz)
    if curve is None:
        return restored, {"applied": False}
    bands = {f"{int(f)}": round(float(np.interp(f, freqs, curve)), 2) for f in (125, 250, 500, 1000, 2000, 4000, 8000)}
    return apply_curve(restored, freqs, curve), {"applied": True, "max_db": round(float(np.abs(curve).max()), 2), "bands_db": bands}
