import pytest

from conftest import ML_READY

pytestmark = pytest.mark.skipif(not ML_READY, reason="ML-пакети не встановлено")

if ML_READY:
    import numpy as np
    from scipy.signal import resample_poly

    from voice_normal import i18n, metrics, process, report, restore, verify

SR = 48_000


def _speechlike(seconds, seed=0):
    from test_process import _speechlike as make

    return make(seconds, seed)


# ---------- вимірювання ----------


def test_echo_is_found_at_any_delay():
    x = _speechlike(30, seed=1)
    clean = verify.echo_level(x)
    assert clean < verify.ECHO_WARN
    for delay_ms in (40, 150, 400):
        k = delay_ms * SR // 1000
        echoed = x.copy()
        echoed[k:] += 0.4 * x[:-k]
        assert verify.echo_level(echoed) > verify.ECHO_FAIL


def test_click_stands_out_in_a_pause_and_near_speech():
    x = _speechlike(10, seed=2)
    level = np.convolve(np.abs(x), np.ones(4800), "same")
    pause = SR + int(np.argmin(level[SR:-SR]))  # пауза всередині запису, не на краю
    for at in (pause, SR * 3):
        score, _ = metrics.click_score(x, at)
        assert score < verify.CLICK_WARN_DB
        broken = x.copy()
        broken[at] += 0.05
        score, level = metrics.click_score(broken, at)
        assert score > verify.CLICK_FAIL_DB and level > verify.CLICK_MIN_LEVEL_DBFS


def test_background_floor_sees_added_noise():
    rng = np.random.default_rng(0)
    x = _speechlike(20, seed=3) + rng.normal(0, 10 ** (-60 / 20), 20 * SR).astype(np.float32)
    power = np.mean(x.reshape(-1, 480).astype(np.float64) ** 2, axis=1)
    mask = np.convolve(power > 1e-5, np.ones(5), "same") > 0  # грубо: мова там, де гучно
    quiet = metrics.background_floor(x, mask, 0, 10 * SR)
    louder = x.copy()
    louder[10 * SR :] += rng.normal(0, 10 ** (-50 / 20), 10 * SR).astype(np.float32)
    assert abs(metrics.background_floor(x, mask, 10 * SR, 20 * SR) - quiet) < 2
    assert metrics.background_floor(louder, mask, 10 * SR, 20 * SR) - quiet > verify.FLOOR_FAIL_DB


def _words(text, probability=0.95):
    return [metrics.Word(i, i + 0.5, w, probability) for i, w in enumerate(text.split())]


def test_changed_word_versus_changed_ending():
    before = _words("сьогодні ми говоримо про голоса і про звук у кімнаті")
    ending = _words("сьогодні ми говоримо про голос і про звук у кімнаті")
    other = _words("сьогодні ми говоримо про машини і про звук у кімнаті")
    _, diffs = metrics.compare_words(before, ending, verify.CONFIDENT_PROBABILITY)
    assert diffs and not verify._changed_words(diffs)
    _, diffs = metrics.compare_words(before, other, verify.CONFIDENT_PROBABILITY)
    assert [d.after for d in verify._changed_words(diffs)] == ["машини"]
    # Невпевнена розшифровка шумного оригіналу — не привід вважати слово заміненим.
    _, diffs = metrics.compare_words(_words("сьогодні ми говоримо про голоса", 0.4), other[:5], 0.8)
    assert not verify._changed_words(diffs)


def test_normalize_word_ignores_case_and_punctuation():
    assert metrics.normalize_word(" Привіт,") == "привіт"
    assert metrics.normalize_word("м’ята") == metrics.normalize_word("м'ята") == "мята"


def test_fbank_shape_and_mean_removed():
    x16 = _speechlike(3)[::3]
    feats = metrics.fbank(x16)
    assert feats.shape == (1 + (x16.size - 400) // 160, 80)
    assert np.allclose(feats.mean(axis=0), 0, atol=1e-3)


# ---------- рішення про ремонт ----------


def _gate(gid, status="fail", **values):
    return verify.Gate(gid, status, values)


def test_repair_decisions():
    plan = process.plan_repair
    loud = _gate("loudness", lufs=-15.2)
    assert plan([loud, _gate("echo", "pass")], "restore", 0) == ["remaster"]
    seam = verify.Gate("seams", "fail", details=[{"seam": 2, "status": "fail"}], chunks=[1, 2])
    assert plan([seam], "restore", 0) == ["recut"]
    words = verify.Gate("words", "fail", {"wer": 0.05}, chunks=[3])
    assert plan([words], "restore", 0) == ["redo"]
    assert plan([words], "restore", 1) == ["redo"]
    # Останній ремонт — відмова від відновлення, щоб не віддати зіпсований голос.
    assert plan([words], "restore", 2) == ["master_only"]
    worse = _gate("quality", ovrl_before=3.0, ovrl_after=2.5)
    assert plan([worse], "restore", 0) == ["master_only"]
    assert plan([_gate("echo")], "restore", 0) == ["master_only"]
    assert plan([_gate("sync", offset_ms=3.0)], "restore", 0) == ["resync"]
    # Лише підсилення: ремонтувати можна тільки гучність.
    assert plan([_gate("quality", ovrl_before=3.0, ovrl_after=2.5)], "master_only", 0) == []


# ---------- переробка фрагментів ----------


class FakeSidon:
    latency = 300

    def __init__(self, gain=1.0):
        self.gain = gain

    def restore(self, wav16):
        y = resample_poly(wav16.astype(np.float64), 3, 1) * self.gain
        y = np.concatenate([np.zeros(self.latency), y])[: 3 * wav16.size - 26]
        return y.astype(np.float32)


def _mask_with_pauses(n48):
    speech = np.ones(n48 // 480, dtype=bool)
    for start in range(100, speech.size, 250):  # паузи по 50 мс кожні 2.5 с
        speech[start : start + 5] = False
    return speech


def test_redo_replaces_only_chosen_chunks():
    x = _speechlike(80, seed=4)
    speech = _mask_with_pauses(x.size)
    r = restore.Restorer(x, FakeSidon())
    chunks = restore.plan_chunks(r.x16.size, speech)
    first = r.render(chunks)
    r.model = FakeSidon(gain=0.5)
    second = r.redo(first, chunks, [1], margin_s=3.0)
    assert second.size == first.size
    a, b = chunks[1].start * 3, chunks[1].end * 3
    fade = r.fade
    # Поза вибраним фрагментом (і його переходами) — ті самі семпли.
    assert np.array_equal(second[: a - fade], first[: a - fade])
    assert np.array_equal(second[b + fade :], first[b + fade :])
    inside = slice(a + fade, b - fade)
    assert np.allclose(second[inside], 0.5 * first[inside], atol=1e-4)


def test_recut_moves_seam_to_another_pause():
    x = _speechlike(80, seed=5)
    speech = _mask_with_pauses(x.size)
    chunks = restore.plan_chunks(x.size // 3, speech)
    moved = restore.recut(chunks, 1, speech)
    assert moved is not None and len(moved) == len(chunks)
    assert abs(moved[1].start - chunks[1].start) >= 1.5 * 16_000
    assert moved[0].start == chunks[0].start and moved[1].end == chunks[1].end
    assert moved[0].end == moved[1].start and not speech[moved[1].start // 160]


# ---------- звіт ----------


def _fake_result(verdict="fail"):
    gates = [
        verify.Gate("quality", "pass", {"ovrl_before": 1.2, "ovrl_after": 3.4, "bak_before": 1.2, "bak_after": 4.1,
                                        "sig_before": 1.3, "sig_after": 3.6, "worst_window_s": 0, "worst_window_change": 1.0},
                    {"max_drop": 0.1, "max_window_drop": 0.5, "target_ovrl": 3.0, "target_bak": 3.5}),
        verify.Gate("words", "fail", {"wer": 0.04, "words_before": 100, "words_after": 100, "changed_words": 1, "language": "uk"},
                    {"wer_fail": 0.15, "wer_warn": 0.08, "changed_words": 0}, [12.5], [0],
                    [{"start_s": 12.5, "end_s": 13.0, "kind": "replace", "before": "кіт", "after": "пес", "changed": True}]),
        verify.Gate("voice", "skip", reason="too_little_speech"),
        verify.Gate("seams", "warn", {"seams": 1, "worst_click_db": 11.0, "worst_floor_jump_db": None},
                    {"click_fail_db": 15, "click_warn_db": 10, "floor_fail_db": 5, "floor_warn_db": 3}, [], [],
                    [{"seam": 1, "at_s": 30.0, "click_db": 11.0, "click_level_dbfs": -40, "floor_jump_db": None, "status": "warn"}]),
    ]
    return {
        "input": {"path": "D:\\Відео\\запис.mp4"}, "verdict": verdict, "route": "master_only",
        "notes": ["restore_rejected"], "probe": {}, "gates": [g.to_dict() for g in gates],
        "rounds": [{"round": 0, "route": "restore", "actions": ["initial"], "gates": {"words": "fail"}},
                   {"round": 1, "route": "master_only", "actions": ["master_only"], "gates": {"words": "skip"}}],
        "chunks": [], "timbre": {"applied": True, "bands_db": {"250": -1.0, "2000": 1.5}},
        "mastering": {"gain_db": 12.0, "limiter_db": 0.5, "limited_percent": 0.01},
        "seconds": {"total": 125, "verify": 40},
        "outputs": {"audio": "a.wav", "video": "a.mp4", "summary": "a.report.json"},
    }


@pytest.mark.parametrize("lang", ["uk", "ru", "en"])
def test_report_renders_in_every_language(lang):
    i18n.set_lang(lang)
    try:
        text = report.render(_fake_result())
    finally:
        i18n.set_lang("uk")
    assert text.startswith("# ") and "запис.mp4" in text
    assert "«кіт» → «пес»" in text and "0:12" in text and "0:30" in text
    assert "{" not in text  # усі підстановки заповнено


# ---------- process із ремонтом ----------


class ScriptedVerifier(verify.Verifier):
    """Повертає наперед задані результати гейтів по раундах, рахує виклики."""

    def __init__(self, script):
        super().__init__("cpu")
        self.script = list(script)
        self.calls = []

    def reference(self, x48, mask):
        return verify.Reference(x48=x48, x16=x48[::3], mask=mask, mos=[], echo=0.0)

    def check(self, ref, y48, *, audio_path, chunks, content_changed, target_lufs, ceiling_dbtp, video=None):
        self.calls.append({"chunks": [c["start_s"] for c in chunks], "content_changed": content_changed, "y": y48.copy()})
        failing = self.script.pop(0) if self.script else []
        gates = []
        for gid in ("quality", "words", "voice", "sync", "loudness", "seams", "echo", "length"):
            if gid in failing:
                gates.append(failing[gid] if isinstance(failing, dict) else verify.Gate(gid, verify.FAIL))
            else:
                gates.append(verify.Gate(gid, verify.PASS))
        return gates


def _long_speech_file(tmp_path):
    from conftest import write_wav

    return write_wav(tmp_path / "довгий запис.wav", _speechlike(70, seed=9), SR)


def test_seam_failure_is_repaired_by_moving_the_cut(models_ready, tmp_path, monkeypatch):
    import voice_normal.sidon

    monkeypatch.setattr(voice_normal.sidon, "Sidon", lambda device: FakeSidon())
    source = _long_speech_file(tmp_path)
    seam = verify.Gate("seams", verify.FAIL, details=[{"seam": 1, "status": "fail"}], chunks=[0, 1])
    verifier = ScriptedVerifier([{"seams": seam}])
    result = process.run(source, tmp_path / "out", lambda m: None, force_route="restore", verifier=verifier)
    assert [r["actions"] for r in result.rounds] == [["initial"], ["recut"]]
    assert result.verdict == verify.PASS and result.route == "restore"
    assert verifier.calls[0]["chunks"][1] != verifier.calls[1]["chunks"][1]
    assert open(result.outputs.report, encoding="utf-8").read().count("## ") >= 4


def test_unfixable_words_fall_back_to_master_only(models_ready, tmp_path, monkeypatch):
    import voice_normal.sidon

    monkeypatch.setattr(voice_normal.sidon, "Sidon", lambda device: FakeSidon())
    source = _long_speech_file(tmp_path)
    bad_words = verify.Gate("words", verify.FAIL, {"wer": 0.3}, chunks=[1])
    verifier = ScriptedVerifier([{"words": bad_words}] * 3)
    result = process.run(source, tmp_path / "out", lambda m: None, force_route="restore", verifier=verifier)
    assert [r["actions"] for r in result.rounds] == [["initial"], ["redo"], ["redo"], ["master_only"]]
    assert result.route == "master_only" and "restore_rejected" in result.notes
    assert verifier.calls[-1]["content_changed"] is False
    assert result.ok
