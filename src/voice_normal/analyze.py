"""Аналіз запису: гучність, мова/тиша, шум, SNR, кліпінг, смуга, реверберація, DNSMOS.

Усі оцінки, крім гучності (ffmpeg ebur128) і DNSMOS (модель Microsoft), — евристики.
Вони потрібні, щоб обрати маршрут і найважчі фрагменти для проби, а не як вимірювання
лабораторної точності.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from voice_normal import media, models, paths

SR = 16_000
FRAME = 320  # 20 мс
HOP = 160  # 10 мс
CLIP_LEVEL = 0.999
# Між цими кутами спектр рахуємо для оцінки смуги.
BANDWIDTH_DROP_DB = 60.0

DNSMOS_SECONDS = 9.01
DNSMOS_HOP_SECONDS = 4.5
# Поліноміальне калібрування з dnsmos_local.py (не персоналізований варіант).
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

GRID_SECONDS = 10.0
_EPS = 1e-12


@dataclass
class Window:
    start_s: float
    end_s: float
    speech_ratio: float
    snr_db: float | None
    ovrl: float | None
    bak: float | None
    clipped: int


@dataclass
class Analysis:
    channel_strategy: str
    channel_levels_db: list[float]
    integrated_lufs: float
    lra_lu: float
    true_peak_dbtp: float
    speech_ratio: float
    speech_seconds: float
    speech_level_db: float | None
    noise_floor_db: float | None
    snr_db: float | None
    clipped_samples: int
    clipped_fraction: float
    bandwidth_hz: float
    rt60_s: float | None
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    dnsmos_ovrl: float | None
    dnsmos_windows: int
    windows: list[Window] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _db(power: float | np.ndarray) -> float | np.ndarray:
    return 10 * np.log10(np.maximum(power, _EPS))


# ---------- рідна частота: кліпінг, рівні каналів, спектр ----------


@dataclass
class _NativeStats:
    channel_power: np.ndarray
    clipped: int
    clipped_per_second: np.ndarray
    total_frames: int
    bandwidth_hz: float


def native_pass(path: Path, info: media.MediaInfo) -> _NativeStats:
    sr = info.sample_rate
    nfft = 4096
    window = np.hanning(nfft).astype(np.float32)
    psd = np.zeros(nfft // 2 + 1)
    psd_count = 0
    power = np.zeros(info.channels)
    clipped = 0
    clipped_seconds: list[int] = []
    total = 0
    for block in media.iter_blocks(path, info.channels, sr):
        total += block.shape[0]
        power += np.sum(block.astype(np.float64) ** 2, axis=0)
        hits = int(np.count_nonzero(np.abs(block) >= CLIP_LEVEL))
        clipped += hits
        clipped_seconds.append(hits)
        mono = block.mean(axis=1)
        for start in range(0, mono.size - nfft + 1, nfft):
            spectrum = np.fft.rfft(mono[start : start + nfft] * window)
            psd += np.abs(spectrum) ** 2
            psd_count += 1
    power /= max(total, 1)
    return _NativeStats(
        channel_power=power,
        clipped=clipped,
        clipped_per_second=np.asarray(clipped_seconds, dtype=np.int64),
        total_frames=total,
        bandwidth_hz=_bandwidth(psd / max(psd_count, 1), sr, nfft),
    )


def _bandwidth(psd: np.ndarray, sr: int, nfft: int) -> float:
    """Найвища частота, де спектр ще не впав на BANDWIDTH_DROP_DB від піку.

    Розпізнає телефонну смугу (~3.4–4 кГц) і зрізаний низьким бітрейтом верх (11–16 кГц).
    """
    if not np.any(psd > 0):
        return 0.0
    freqs = np.arange(psd.size) * sr / nfft
    # Згладжуємо ~200 Гц, щоб окремі гармоніки не вдавали смугу.
    width = max(1, int(200 / (sr / nfft)))
    smooth = np.convolve(psd, np.ones(width) / width, mode="same")
    level = _db(smooth)
    above = np.nonzero(level > level.max() - BANDWIDTH_DROP_DB)[0]
    return float(freqs[above[-1]]) if above.size else 0.0


# ---------- 16 кГц: канали, VAD, SNR, реверберація ----------


def choose_channels(audio: np.ndarray, channel_power: np.ndarray) -> tuple[str, np.ndarray]:
    """Як звести канали в моно, не втративши голос.

    Мертвий або значно тихіший канал (типово для петлички в одному каналі камери)
    не змішуємо; протифазні канали теж — середнє їх би погасило.
    """
    if audio.shape[1] == 1:
        return "mono", audio[:, 0]
    levels = _db(channel_power)
    loudest = int(np.argmax(levels))
    if levels.max() - np.sort(levels)[-2] > 15:
        return f"channel{loudest + 1}", audio[:, loudest]
    if audio.shape[1] == 2:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
        denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
        if denom > 0 and np.sum(left * right) / denom < -0.5:
            return f"channel{loudest + 1}", audio[:, loudest]
    return "mean", audio.mean(axis=1)


def frame_power(mono: np.ndarray) -> np.ndarray:
    if mono.size < FRAME:
        return np.array([float(np.mean(mono.astype(np.float64) ** 2))]) if mono.size else np.zeros(1)
    count = 1 + (mono.size - FRAME) // HOP
    strides = (mono.strides[0] * HOP, mono.strides[0])
    frames = np.lib.stride_tricks.as_strided(mono, shape=(count, FRAME), strides=strides)
    return np.mean(frames.astype(np.float64) ** 2, axis=1)


def speech_mask(mono: np.ndarray, n_frames: int) -> np.ndarray:
    """Маска кадрів із мовою за Silero VAD (поставляється разом із faster-whisper)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=300, speech_pad_ms=30)
    mask = np.zeros(n_frames, dtype=bool)
    for segment in get_speech_timestamps(mono.astype(np.float32), options, sampling_rate=SR):
        first = segment["start"] // HOP
        last = max(first + 1, (segment["end"] - FRAME) // HOP + 1)
        mask[first:last] = True
    return mask


def levels(power: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Рівень мови, шуму й SNR у дБ (відносно повної шкали)."""
    if not mask.any():
        return None, None, None
    speech = power[mask]
    if (~mask).sum() >= 20:
        noise = power[~mask]
    else:
        # Мова без пауз: шум оцінюємо за найтихішими кадрами між словами.
        noise = speech[speech <= np.percentile(speech, 10)]
    p_noise = float(np.mean(noise))
    p_speech = float(np.mean(speech))
    snr = _db(max(p_speech - p_noise, _EPS)) - _db(p_noise)
    return float(_db(p_speech)), float(_db(p_noise)), float(snr)


def estimate_rt60(power: np.ndarray, mask: np.ndarray, noise_db: float | None) -> float | None:
    """Грубо: швидкість згасання енергії після кінців фраз -> RT60 = 60 дБ / нахил."""
    if noise_db is None:
        return None
    level = _db(power)
    ends = np.nonzero(mask[:-1] & ~mask[1:])[0]
    estimates = []
    for end in ends:
        if end < 5:
            continue
        tail = level[end - 5 : end + 40]
        if tail.size < 20 or mask[end + 1 : end + 30].any():
            continue
        peak = int(np.argmax(tail[:10]))
        decay = tail[peak:]
        top, bottom = decay[0] - 5, noise_db + 5
        if top - bottom < 10:
            continue
        idx = np.nonzero((decay <= top) & (decay >= bottom))[0]
        if idx.size < 4:
            continue
        slope = np.polyfit(idx * HOP / SR, decay[idx], 1)[0]
        if slope < -1:
            estimates.append(-60.0 / slope)
    if len(estimates) < 3:
        return None
    return float(np.clip(np.median(estimates), 0.05, 3.0))


# ---------- DNSMOS ----------


def dnsmos(mono: np.ndarray, root: Path | None = None) -> list[tuple[float, float, float, float]]:
    """(початок_с, SIG, BAK, OVRL) для кожного вікна 9.01 с з кроком 4.5 с."""
    import onnxruntime as ort

    model = models.get_model("dnsmos")
    session = ort.InferenceSession(
        paths.long_path(models.file_path(model, model.files[0], root)), providers=["CPUExecutionProvider"]
    )
    length = int(DNSMOS_SECONDS * SR)
    audio = mono.astype(np.float32)
    while audio.size < length:
        # Як в еталонному скрипті: короткий запис повторюємо до довжини вікна.
        audio = np.concatenate([audio, audio])
    hop = int(DNSMOS_HOP_SECONDS * SR)
    results = []
    for start in range(0, audio.size - length + 1, hop):
        segment = audio[start : start + length][np.newaxis, :]
        sig, bak, ovr = session.run(None, {"input_1": segment})[0][0]
        results.append((start / SR, float(_P_SIG(sig)), float(_P_BAK(bak)), float(_P_OVR(ovr))))
    return results


# ---------- вікна для вибору проби ----------


def grid_windows(
    duration_s: float,
    power: np.ndarray,
    mask: np.ndarray,
    mos: list[tuple[float, float, float, float]],
    clipped_per_second: np.ndarray,
) -> list[Window]:
    frames_per_s = SR / HOP
    windows = []
    start = 0.0
    while start < duration_s:
        end = min(start + GRID_SECONDS, duration_s)
        a, b = int(start * frames_per_s), int(end * frames_per_s)
        w_mask, w_power = mask[a:b], power[a:b]
        ratio = float(w_mask.mean()) if w_mask.size else 0.0
        _, _, snr = levels(w_power, w_mask) if w_mask.size else (None, None, None)
        overlapping = [m for m in mos if m[0] < end and m[0] + DNSMOS_SECONDS > start]
        ovrl = float(np.mean([m[3] for m in overlapping])) if overlapping else None
        bak = float(np.mean([m[2] for m in overlapping])) if overlapping else None
        clipped = int(clipped_per_second[int(start) : int(np.ceil(end))].sum())
        windows.append(
            Window(round(start, 3), round(end, 3), round(ratio, 3), _round(snr), _round(ovrl), _round(bak), clipped)
        )
        start = end
    return windows


def analyze(path: Path, root: Path | None = None) -> tuple[media.MediaInfo, Analysis]:
    info = media.probe(path)
    loud = media.loudness(path)
    native = native_pass(path, info)

    audio = media.decode(path, sample_rate=SR, channels=info.channels)
    strategy, mono = choose_channels(audio, native.channel_power)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    del audio

    power = frame_power(mono)
    mask = speech_mask(mono, power.size)
    speech_db, noise_db, snr = levels(power, mask)
    rt60 = estimate_rt60(power, mask, noise_db)

    # DNSMOS рахуємо лише по вікнах, де є мова: тиша дає беззмістовні оцінки.
    mos_all = dnsmos(mono, root)
    frames_per_s = SR / HOP
    mos = [
        m for m in mos_all
        if mask[int(m[0] * frames_per_s) : int((m[0] + DNSMOS_SECONDS) * frames_per_s)].mean() >= 0.3
    ] or mos_all

    duration = mono.size / SR
    windows = grid_windows(duration, power, mask, mos, native.clipped_per_second)
    total_samples = native.total_frames * info.channels
    analysis = Analysis(
        channel_strategy=strategy,
        channel_levels_db=[round(float(x), 2) for x in _db(native.channel_power)],
        integrated_lufs=loud.integrated_lufs,
        lra_lu=loud.lra_lu,
        true_peak_dbtp=loud.true_peak_dbtp,
        speech_ratio=round(float(mask.mean()), 4),
        speech_seconds=round(float(mask.sum()) * HOP / SR, 2),
        speech_level_db=_round(speech_db),
        noise_floor_db=_round(noise_db),
        snr_db=_round(snr),
        clipped_samples=native.clipped,
        clipped_fraction=native.clipped / max(total_samples, 1),
        bandwidth_hz=round(native.bandwidth_hz),
        rt60_s=_round(rt60, 3),
        dnsmos_sig=_round(float(np.mean([m[1] for m in mos]))) if mos else None,
        dnsmos_bak=_round(float(np.mean([m[2] for m in mos]))) if mos else None,
        dnsmos_ovrl=_round(float(np.mean([m[3] for m in mos]))) if mos else None,
        dnsmos_windows=len(mos),
        windows=windows,
    )
    return info, analysis


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)
