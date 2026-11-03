import pytest

from conftest import ML_READY, ffmpeg, write_wav

pytestmark = pytest.mark.skipif(not ML_READY, reason="ML-пакети не встановлено")

if ML_READY:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from voice_normal import align, loudness, media, process, restore, timbre

SR = 48_000


def _speechlike(seconds: float, seed: int = 0) -> "np.ndarray":
    # Фрази по 1.5–3 с з паузами між ними: гармонічний тон із модуляцією і шумом.
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    pos = int(0.5 * SR)
    while pos < out.size - SR:
        n = int(rng.uniform(1.5, 3.0) * SR)
        t = np.arange(n) / SR
        f0 = rng.uniform(110, 220) * (1 + 0.1 * np.sin(2 * np.pi * 0.8 * t))
        phase = 2 * np.pi * np.cumsum(f0) / SR
        voice = sum(np.sin(k * phase) / k for k in range(1, 15))
        env = np.clip(np.sin(np.pi * t / t[-1]), 0, 1) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)
        seg = (0.1 * voice * env).astype(np.float32)[: out.size - pos]
        out[pos : pos + seg.size] += seg
        pos += n + int(rng.uniform(0.3, 0.8) * SR)
    return out + rng.normal(0, 1e-4, out.size).astype(np.float32)


# ---------- гучність і лімітер ----------


def test_lufs_matches_ffmpeg(media_dir):
    x = _speechlike(20)
    path = write_wav(media_dir / "гучність.wav", x, SR)
    assert loudness.integrated_lufs(x) == pytest.approx(media.loudness(path).integrated_lufs, abs=0.15)


def test_master_hits_targets_without_compressing(media_dir):
    x = _speechlike(30) * 0.05
    y, stats = loudness.master(x, -16.0, -1.0)
    path = write_wav(media_dir / "зведення.wav", y, SR)
    measured = media.loudness(path)
    assert measured.integrated_lufs == pytest.approx(-16.0, abs=0.3)
    assert measured.true_peak_dbtp <= -1.0
    # Лімітер працює лише на окремих піках, а не постійно.
    assert stats["limited_percent"] < 5


def test_limiter_leaves_quiet_signal_alone():
    x = _speechlike(5) * 0.1
    y, reduction, busy = loudness.peak_limit(x, -1.0)
    assert reduction == 0.0 and busy == 0.0 and np.array_equal(x, y)


def test_limiter_has_no_delay():
    x = _speechlike(10) * 10  # гучно (піки ~+5 dBFS), щоб лімітер працював
    y, reduction, _ = loudness.peak_limit(x, -1.0)
    assert reduction > 0
    assert loudness.true_peak_db(y) <= -1.0 + 0.05
    peak = int(np.argmax(np.abs(np.correlate(y[SR : 2 * SR], x[SR - 100 : 2 * SR + 100], mode="valid"))))
    assert peak == 100


# ---------- зсув і складання ----------


def test_measure_finds_known_shift():
    x = _speechlike(20, seed=3)
    for lag in (0, 48, 150, -96):
        delayed = np.roll(x, lag)
        assert align.measure(delayed, x).samples == pytest.approx(lag, abs=8)


def test_shift_and_fit_length_keep_length():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(align.shift(x, 3), [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])
    assert np.array_equal(align.shift(x, -2), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    assert align.fit_length(x, 4).size == 4 and align.fit_length(x, 12).size == 12


def test_chunks_cover_everything_and_cut_in_pauses():
    x = _speechlike(95, seed=5)
    x16 = resample_poly(x.astype(np.float64), 1, 3)
    power = np.array([np.mean(x16[i : i + 160] ** 2) for i in range(0, x16.size - 160, 160)])
    speech = power > 1e-5
    chunks = restore.plan_chunks(x16.size, speech)
    assert chunks[0].start == 0 and chunks[-1].end == x16.size
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))
    for c in chunks[:-1]:
        assert c.cut_in_silence and not speech[c.end // 160]


class FakeSidon:
    """Імітує Sidon: 16 кГц -> 48 кГц, затримка 300 семплів і на 26 семплів коротше."""

    latency = 300

    def restore(self, wav16):
        y = resample_poly(wav16.astype(np.float64), 3, 1)
        y = np.concatenate([np.zeros(self.latency), y])[: 3 * wav16.size - 26]
        return y.astype(np.float32)


def test_assembly_is_sample_exact_with_latency_removed():
    x = _speechlike(70, seed=7)
    speech = np.ones(x.size // 480, dtype=bool)
    speech[::300] = False  # паузи, де можна різати
    y, chunks = restore.restore(x, speech, FakeSidon())
    assert y.size == x.size
    assert len(chunks) >= 2
    assert all(c.offset_measured and abs(c.offset_samples - FakeSidon.latency) <= 4 for c in chunks)
    # Після компенсації затримки й склейки результат майже дорівнює входу (ресемплінг 16 кГц
    # зрізає верх, тому порівнюємо до 7 кГц).
    from scipy.signal import butter, sosfiltfilt

    sos = butter(8, 7000, fs=SR, output="sos")
    a, b = sosfiltfilt(sos, x[SR:-SR]), sosfiltfilt(sos, y[SR:-SR])
    assert np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a**2)) < 0.15


def test_timbre_filter_is_zero_phase():
    x = _speechlike(10)
    freqs = np.linspace(0, SR / 2, 100)
    curve = np.where(freqs > 2000, 2.0, -1.0)
    y = timbre.apply_curve(x, freqs, curve)
    assert y.size == x.size
    assert align.measure(y, x).samples == pytest.approx(0, abs=2)


def test_flat_curve_changes_nothing():
    x = _speechlike(5)
    y = timbre.apply_curve(x, np.array([0.0, SR / 2]), np.zeros(2))
    assert np.max(np.abs(y - x)) < 1e-3


# ---------- process цілком ----------


def test_master_only_route_end_to_end(speech_wav, models_ready, tmp_path):
    out = tmp_path / "Результат з пробілами"
    result = process.run(speech_wav, out, lambda m: None, force_route="master_only")
    data, sr = sf.read(result.outputs.audio)
    info = media.probe(speech_wav)
    assert sr == SR and data.size == round(info.duration_s * SR)
    assert sf.info(result.outputs.audio).subtype == "PCM_24"
    measured = media.loudness(__import__("pathlib").Path(result.outputs.audio))
    assert measured.integrated_lufs == pytest.approx(-16, abs=0.5)
    assert measured.true_peak_dbtp <= -1.0


def test_video_gets_audio_back(noisy_wav, models_ready, tmp_path):
    video = tmp_path / "відео камера.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=20", "-i", str(noisy_wav),
           "-t", "20", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video))
    result = process.run(video, tmp_path / "out", lambda m: None, force_route="master_only")
    assert result.outputs.video and result.outputs.video.endswith(".cleaned.mp4")
    original, cleaned = media.probe(video), media.probe(__import__("pathlib").Path(result.outputs.video))
    assert cleaned.has_video and abs(cleaned.duration_s - original.duration_s) < 0.05


def test_sidon_route_end_to_end(speech_wav, models_ready, tmp_path):
    result = process.run(speech_wav, tmp_path / "Результат", lambda m: None, force_route="restore")
    data, _ = sf.read(result.outputs.audio)
    assert data.size == result.length["output_samples"]
    assert abs(result.sync["offset_ms"]) <= 1.0
    assert result.chunks and all(c["cut_in_silence"] for c in result.chunks)
