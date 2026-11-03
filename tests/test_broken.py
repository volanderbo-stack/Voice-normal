"""Зіпсований результат має не пройти саме той гейт, що відповідає за цю ваду.

Оригінал — справжня мова (System.Speech) із тихим шумом кімнати. "Добрий" результат —
його зведення до −16 LUFS: він проходить усі гейти. Далі кожну ваду вносимо окремо.
"""

import pytest

from conftest import ML_READY, SPEECH_TEXT, tts

pytestmark = pytest.mark.skipif(not ML_READY, reason="ML-пакети не встановлено")

if ML_READY:
    import numpy as np

    from voice_normal import align, loudness, media, plan, process, restore, verify

SR = 48_000


def _read(path):
    x, _ = process.read_mono_48k(path, media.probe(path), "mean")
    return x


def _with_room_noise(x, seed=0, level_dbfs=-60.0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 10 ** (level_dbfs / 20), x.size)
    return (x + noise).astype(np.float32)


def _master(x):
    y, _ = loudness.master(x, plan.TARGET_LUFS, plan.TARGET_TRUE_PEAK)
    return y


# Два різні голоси, щоб перевірити гейт голосу; обидва є в Windows 10/11 і на windows-latest.
VOICE, OTHER_VOICE = "*Zira*", "*David*"


@pytest.fixture(scope="module")
def setup(media_dir, models_ready):
    x = _with_room_noise(_read(tts(media_dir / "оригінал.wav", voice=VOICE)))
    speech = process.speech_mask_48k(x)
    verifier = verify.Verifier("cpu")
    ref = verifier.reference(x, speech)
    # Шви — як у справжній обробці: фрагменти по ~8 с, межі в паузах.
    chunks = [c.to_dict() for c in restore.plan_chunks(x.size // 3, speech, chunk_s=8.0)]
    assert len(chunks) >= 3
    return {"x": x, "y": _master(x), "ref": ref, "verifier": verifier, "chunks": chunks, "speech": speech}


def _check(setup, y, path=None, content_changed=True):
    gates = setup["verifier"].check(
        setup["ref"], y, audio_path=path, chunks=setup["chunks"], content_changed=content_changed,
        target_lufs=plan.TARGET_LUFS, ceiling_dbtp=plan.TARGET_TRUE_PEAK,
    )
    return {g.id: g for g in gates}


def test_good_result_passes_every_gate(setup, tmp_path):
    path = tmp_path / "добрий результат.wav"
    process.write_wav(path, setup["y"])
    gates = _check(setup, setup["y"], path)
    assert {gid: g.status for gid, g in gates.items() if g.status == verify.FAIL} == {}
    assert gates["words"].status == verify.PASS and gates["voice"].values["similarity"] > 0.9


def test_echo_fails(setup):
    y = setup["y"].copy()
    k = int(0.15 * SR)
    y[k:] += 0.4 * setup["y"][:-k]
    g = setup["verifier"].echo(setup["ref"], _master(y))
    assert g.status == verify.FAIL, g.values


def test_50ms_shift_fails(setup):
    late = align.shift(setup["y"], -int(0.05 * SR))  # звук запізнюється на 50 мс
    g = setup["verifier"].sync(setup["ref"], late, setup["chunks"])
    assert g.status == verify.FAIL and g.values["offset_ms"] == pytest.approx(50, abs=1), g.values


def test_plus_3_lu_fails(setup, tmp_path):
    path = tmp_path / "гучніше.wav"
    louder = setup["y"] * 10 ** (3 / 20)
    process.write_wav(path, louder)
    g = setup["verifier"].loudness(path, louder, plan.TARGET_LUFS, plan.TARGET_TRUE_PEAK)
    assert g.status == verify.FAIL and g.values["lufs"] > plan.TARGET_LUFS + 2, g.values


def test_substituted_word_fails(setup, media_dir):
    changed = tts(media_dir / "інше слово.wav", SPEECH_TEXT.replace("in a room", "in a kitchen"), voice=VOICE)
    y = _master(align.fit_length(_with_room_noise(_read(changed), seed=1), setup["x"].size))
    from voice_normal import metrics

    g = setup["verifier"].words(setup["ref"], metrics.to16(y), setup["chunks"])
    assert g.status == verify.FAIL, g.values
    assert any(d["changed"] and "kitchen" in d["after"] for d in g.details), g.details


def test_different_voice_fails(setup, media_dir):
    other = tts(media_dir / "інший голос.wav", voice=OTHER_VOICE)
    y = _master(align.fit_length(_with_room_noise(_read(other), seed=2), setup["x"].size))
    from voice_normal import metrics

    g = setup["verifier"].voice(setup["ref"], metrics.to16(y), setup["chunks"])
    assert g.status == verify.FAIL, g.values


def test_click_at_seam_fails(setup):
    y = setup["y"].copy()
    at = int(round(setup["chunks"][1]["start_s"] * SR))
    y[at] += 0.05
    g = setup["verifier"].seams(setup["ref"], y, setup["chunks"])
    assert g.status == verify.FAIL and g.details[0]["status"] == verify.FAIL, g.details


def test_background_jump_at_seam_fails(setup):
    y = setup["y"].copy()
    at = int(round(setup["chunks"][1]["start_s"] * SR))
    end = int(round(setup["chunks"][1]["end_s"] * SR))
    # Фон одного фрагмента на ~10 дБ гучніший, ніж у сусідів.
    floor = 10 ** ((verify.FLOOR_FAIL_DB + 5) / 20)
    rng = np.random.default_rng(3)
    base = setup["y"][: at][~np.repeat(setup["speech"], 480)[: at]]
    sigma = float(np.std(base)) or 1e-4
    y[at:end] += rng.normal(0, sigma * np.sqrt(floor**2 - 1), end - at).astype(np.float32)
    g = setup["verifier"].seams(setup["ref"], y, setup["chunks"])
    assert g.status == verify.FAIL, g.details
    assert any(d["floor_jump_db"] is not None and abs(d["floor_jump_db"]) > verify.FLOOR_FAIL_DB for d in g.details)
