import pytest

from conftest import ML_READY, ffmpeg, write_wav

pytestmark = pytest.mark.skipif(not ML_READY, reason="ML-пакети не встановлено")

if ML_READY:
    import numpy as np

    from voice_normal import analyze, media, plan


def test_loudness_of_reference_sine(media_dir):
    # BS.1770: синус 1 кГц з піком 0 dBFS має -3.01 LUFS; амплітуда 0.1 -> -23.01 LUFS.
    path = media_dir / "синус 1 кГц.wav"
    # Не "sine": той генератор видає амплітуду 1/8, а нам потрібна точно 0.1.
    ffmpeg("-f", "lavfi", "-i", "aevalsrc=0.1*sin(2*PI*1000*t):s=48000:d=10", "-c:a", "pcm_f32le", str(path))
    loud = media.loudness(path)
    assert loud.integrated_lufs == pytest.approx(-23.0, abs=0.2)
    assert loud.true_peak_dbtp == pytest.approx(-20.0, abs=0.3)


def test_probe_reads_cyrillic_path_with_spaces(media_dir):
    path = write_wav(media_dir / "запис з камери.wav", np.zeros((48000, 2), dtype=np.float32), 48000)
    info = media.probe(path)
    assert (info.sample_rate, info.channels) == (48000, 2)
    assert info.duration_s == pytest.approx(1.0, abs=1e-3)


def test_clipping_is_counted(media_dir):
    rng = np.random.default_rng(1)
    data = rng.uniform(-0.5, 0.5, 48000).astype(np.float32)
    data[1000:1100] = 1.0
    path = write_wav(media_dir / "кліпінг.wav", data, 48000)
    stats = analyze.native_pass(path, media.probe(path))
    assert stats.clipped == 100
    assert stats.clipped_per_second[0] == 100


def test_phone_band_is_detected(media_dir):
    path = media_dir / "телефон.wav"
    # Як справжній телефонний тракт: 8 кГц і назад до 48 кГц — над 4 кГц нічого немає.
    ffmpeg("-f", "lavfi", "-i", "anoisesrc=color=white:sample_rate=48000:duration=5:seed=3",
           "-af", "aresample=8000,aresample=48000", str(path))
    stats = analyze.native_pass(path, media.probe(path))
    assert 3000 < stats.bandwidth_hz < 6000


def test_dead_channel_is_not_mixed_in():
    voice = np.random.default_rng(0).normal(0, 0.1, (16000, 1))
    audio = np.hstack([voice, voice * 0.001]).astype(np.float32)
    power = np.mean(audio.astype(np.float64) ** 2, axis=0)
    strategy, mono = analyze.choose_channels(audio, power)
    assert strategy == "channel1"
    assert np.array_equal(mono, audio[:, 0])


def test_antiphase_channels_are_not_averaged():
    voice = np.random.default_rng(0).normal(0, 0.1, (16000, 1))
    audio = np.hstack([voice, -voice]).astype(np.float32)
    strategy, _ = analyze.choose_channels(audio, np.mean(audio.astype(np.float64) ** 2, axis=0))
    assert strategy.startswith("channel")


def test_similar_channels_are_averaged():
    voice = np.random.default_rng(0).normal(0, 0.1, (16000, 1))
    audio = np.hstack([voice, voice * 0.9]).astype(np.float32)
    strategy, _ = analyze.choose_channels(audio, np.mean(audio.astype(np.float64) ** 2, axis=0))
    assert strategy == "mean"


def test_snr_from_known_levels():
    power = np.full(1000, 1e-5)
    mask = np.zeros(1000, dtype=bool)
    mask[200:800] = True
    power[mask] = 1e-2 + 1e-5
    speech_db, noise_db, snr = analyze.levels(power, mask)
    assert noise_db == pytest.approx(-50, abs=0.01)
    assert snr == pytest.approx(30, abs=0.01)


def _analysis(**overrides):
    base = dict(
        channel_strategy="mono", channel_levels_db=[-30.0], integrated_lufs=-20.0, lra_lu=5.0,
        true_peak_dbtp=-3.0, speech_ratio=0.8, speech_seconds=80.0, speech_level_db=-25.0,
        noise_floor_db=-70.0, snr_db=45.0, clipped_samples=0, clipped_fraction=0.0, bandwidth_hz=20000,
        rt60_s=0.2, dnsmos_sig=3.6, dnsmos_bak=4.1, dnsmos_ovrl=3.4, dnsmos_windows=10,
    )
    base.update(overrides)
    return analyze.Analysis(**base)


def test_clean_recording_skips_sidon():
    route, reasons, _ = plan.choose_route(_analysis(), 100)
    assert route == "master_only" and reasons[0].code == "clean"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"snr_db": 12.0}, "noise"),
        ({"rt60_s": 0.6}, "reverb"),
        ({"clipped_fraction": 0.01, "clipped_samples": 4800}, "clipping"),
        ({"bandwidth_hz": 3900}, "bandwidth"),
        ({"dnsmos_ovrl": 2.5}, "quality"),
    ],
)
def test_each_problem_routes_to_restore(overrides, code):
    route, reasons, _ = plan.choose_route(_analysis(**overrides), 100)
    assert route == "restore"
    assert code in [r.code for r in reasons]


def test_low_speech_warns():
    _, _, warnings = plan.choose_route(_analysis(speech_ratio=0.1), 100)
    assert "low_speech" in [w.code for w in warnings]


def _windows(scores):
    # Складність задаємо через OVRL: що нижче, то важче.
    return [analyze.Window(i * 10.0, (i + 1) * 10.0, 0.9, 30.0, ovrl, 3.0, 0) for i, ovrl in enumerate(scores)]


def test_probe_takes_hardest_150_seconds():
    scores = [3.5] * 40
    for i in range(10, 25):
        scores[i] = 1.5
    probe, whole = plan.choose_probe(_windows(scores), 400)
    assert not whole
    assert probe == [(100.0, 250.0)]


def test_short_recording_probes_whole_file():
    probe, whole = plan.choose_probe(_windows([2.0] * 15), 150)
    assert whole and probe == [(0.0, 150)]


def test_probe_ignores_silent_windows():
    windows = _windows([3.5] * 30)
    windows[5] = analyze.Window(50.0, 60.0, 0.0, None, 1.0, 1.0, 0)
    probe, _ = plan.choose_probe(windows, 300)
    assert all(not (s <= 50.0 < e) for s, e in probe)


def test_real_speech_noise_is_detected_and_routed(speech_wav, noisy_wav, models_ready):
    _, clean = analyze.analyze(speech_wav)
    _, noisy = analyze.analyze(noisy_wav)
    assert clean.speech_ratio > 0.5
    assert clean.snr_db > noisy.snr_db + 10
    assert clean.dnsmos_bak > noisy.dnsmos_bak
    info, _ = analyze.analyze(noisy_wav)
    result = plan.build_plan(info, noisy)
    assert result.route == "restore"
    assert "noise" in [r.code for r in result.reasons]


def test_plan_files_are_utf8(speech_wav, models_ready, tmp_path):
    info, a = analyze.analyze(speech_wav)
    result = plan.build_plan(info, a)
    out = tmp_path / "Результат з пробілами"
    json_path, md_path = plan.write(result, speech_wav, out)
    assert json_path.name == "мова чиста.plan.json"
    text = md_path.read_bytes().decode("utf-8")
    assert "мова чиста.wav" in text and "\r\n" not in text
