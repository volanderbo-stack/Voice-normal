"""Перевірка результату за гейтами: якість, слова, голос, синхрон, гучність, шви, луна, довжина.

Кожен гейт порівнює результат з оригіналом і має статус pass / warn / fail / skip.
Пороги підібрані на тестових записах (див. tests/test_verify.py): нормальний результат
проходить із запасом, а навмисно зіпсований — ні.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from voice_normal import align, analyze, media, metrics

SR48 = 48_000
PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

# Якість (DNSMOS P.835): не гірше за оригінал — ні в цілому, ні в окремому вікні.
QUALITY_MAX_DROP = 0.1
QUALITY_MAX_WINDOW_DROP = 0.5
QUALITY_TARGET_OVRL = 3.0
QUALITY_TARGET_BAK = 3.5
# Слова: розбіжність розшифровок і "впевнені" заміни на зовсім інше слово.
WER_FAIL, WER_WARN = 0.15, 0.08
CONFIDENT_PROBABILITY = 0.8
# Заміна закінчення ("голоса" -> "голос") — звичайне коливання Whisper, а не нове слово.
CHANGED_WORD_SIMILARITY = 0.5
# Голос: косинус ембедингів ECAPA між оригіналом і результатом.
VOICE_FAIL, VOICE_WINDOW_FAIL, VOICE_WARN = 0.45, 0.35, 0.6
VOICE_WINDOW_S = 10.0
# Синхрон.
SYNC_MAX_MS = 1.0
SYNC_WINDOW_WARN_MS, SYNC_WINDOW_FAIL_MS = 5.0, 20.0
SYNC_SEARCH_MS = 250.0
SYNC_WINDOW_MIN_CORRELATION = 0.3
# Гучність.
LOUDNESS_TOLERANCE_LU = 0.5
# Шви: клацання і стрибок рівня фону між сусідніми фрагментами.
CLICK_FAIL_DB, CLICK_WARN_DB = 15.0, 10.0
CLICK_MIN_LEVEL_DBFS = -70.0
FLOOR_FAIL_DB, FLOOR_WARN_DB = 5.0, 3.0
# Луна: автокореляція (похідної) хвилі на затримках 20–500 мс.
ECHO_FAIL, ECHO_WARN, ECHO_OVER_INPUT = 0.15, 0.08, 0.05
ECHO_MIN_MS, ECHO_MAX_MS = 20.0, 500.0
# Довжина відео після заміни звуку (AAC додає кілька десятків семплів на початку).
VIDEO_DURATION_TOLERANCE_S = 0.05


@dataclass
class Gate:
    id: str
    status: str
    values: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    # Де саме проблема: секунди й номери фрагментів — для ремонту і для звіту.
    where_s: list[float] = field(default_factory=list)
    chunks: list[int] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def worst(gates: list[Gate]) -> str:
    statuses = {g.status for g in gates}
    for status in (FAIL, WARN):
        if status in statuses:
            return status
    return PASS


def failing(gates: list[Gate]) -> list[Gate]:
    return [g for g in gates if g.status == FAIL]


def _chunks_for(chunks: list[dict], spans: list[tuple[float, float]]) -> list[int]:
    found = set()
    for start, end in spans:
        for i, c in enumerate(chunks):
            if c["start_s"] < end and start < c["end_s"]:
                found.add(i)
    return sorted(found)


# ---------- окремі вимірювання ----------


def echo_level(x48: np.ndarray, block_s: float = 30.0) -> float:
    """Найбільша нормована автокореляція похідної сигналу на затримках 20–500 мс.

    Луна — це затримана копія того самого звуку, тож на своїй затримці дає пік;
    у звичайній мові на таких затримках кореляції майже немає. Рахуємо блоками, щоб
    годинний запис не займав гігабайти пам'яті.
    """
    max_lag = int(ECHO_MAX_MS * SR48 / 1000)
    acc = np.zeros(max_lag + 1)
    d = np.diff(x48.astype(np.float64))
    step = int(block_s * SR48)
    for start in range(0, d.size, step):
        block = d[start : start + step + max_lag]
        if block.size <= max_lag * 2:
            continue
        acc += fftconvolve(block, block[::-1], mode="full")[block.size - 1 : block.size + max_lag]
    if acc[0] <= 0:
        return 0.0
    acc /= acc[0]
    return float(np.max(np.abs(acc[int(ECHO_MIN_MS * SR48 / 1000) :])))


def _changed_words(diffs: list[metrics.WordDiff]) -> list[metrics.WordDiff]:
    return [
        d for d in diffs
        if d.confident and difflib.SequenceMatcher(a=d.before, b=d.after).ratio() < CHANGED_WORD_SIMILARITY
    ]


# ---------- перевірка ----------


@dataclass
class Reference:
    """Усе про оригінал, що не змінюється між раундами ремонту."""

    x48: np.ndarray
    x16: np.ndarray
    mask: np.ndarray  # маска мови по 10 мс
    mos: list
    echo: float
    words: list | None = None
    language: str | None = None


class Verifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._transcriber = None
        self._encoder = None

    @property
    def transcriber(self) -> metrics.Transcriber:
        if self._transcriber is None:
            self._transcriber = metrics.Transcriber(self.device)
        return self._transcriber

    @property
    def encoder(self) -> metrics.SpeakerEncoder:
        if self._encoder is None:
            self._encoder = metrics.SpeakerEncoder()
        return self._encoder

    def reference(self, x48: np.ndarray, mask: np.ndarray) -> Reference:
        x16 = metrics.to16(x48)
        return Reference(x48=x48, x16=x16, mask=mask, mos=analyze.dnsmos(x16), echo=echo_level(x48))

    def check(
        self,
        ref: Reference,
        y48: np.ndarray,
        *,
        audio_path: Path | None,
        chunks: list[dict],
        content_changed: bool,
        target_lufs: float,
        ceiling_dbtp: float,
        video: tuple[Path, Path] | None = None,
    ) -> list[Gate]:
        y16 = metrics.to16(y48)
        gates = [self.quality(ref, y16, chunks, content_changed)]
        if content_changed:
            gates += [self.words(ref, y16, chunks), self.voice(ref, y16, chunks)]
        else:
            gates += [Gate("words", SKIP, reason="gain_only"), Gate("voice", SKIP, reason="gain_only")]
        gates += [
            self.sync(ref, y48, chunks),
            self.loudness(audio_path, y48, target_lufs, ceiling_dbtp),
            self.seams(ref, y48, chunks),
            self.echo(ref, y48),
            self.length(ref, y48, audio_path, video),
        ]
        return gates

    # --- гейти ---

    def quality(self, ref: Reference, y16: np.ndarray, chunks: list[dict], content_changed: bool) -> Gate:
        after = analyze.dnsmos(y16)
        if not ref.mos or not after:
            return Gate("quality", SKIP, reason="too_short")

        def mean(scores, i):
            return round(float(np.mean([s[i] for s in scores])), 2)

        drops = [(a[0], a[3] - b[3]) for a, b in zip(after, ref.mos)]
        worst_start, worst_drop = min(drops, key=lambda d: d[1])
        values = {
            "ovrl_before": mean(ref.mos, 3), "ovrl_after": mean(after, 3),
            "bak_before": mean(ref.mos, 2), "bak_after": mean(after, 2),
            "sig_before": mean(ref.mos, 1), "sig_after": mean(after, 1),
            "worst_window_s": round(worst_start, 1), "worst_window_change": round(worst_drop, 2),
        }
        limits = {"max_drop": QUALITY_MAX_DROP, "max_window_drop": QUALITY_MAX_WINDOW_DROP,
                  "target_ovrl": QUALITY_TARGET_OVRL, "target_bak": QUALITY_TARGET_BAK}
        bad = [(s, s + analyze.DNSMOS_SECONDS) for s, d in drops if d < -QUALITY_MAX_WINDOW_DROP]
        if values["ovrl_after"] < values["ovrl_before"] - QUALITY_MAX_DROP:
            status = FAIL
        elif bad:
            status = FAIL
        elif content_changed and (values["ovrl_after"] < QUALITY_TARGET_OVRL or values["bak_after"] < QUALITY_TARGET_BAK):
            status = WARN
        else:
            status = PASS
        return Gate("quality", status, values, limits, [round(s, 1) for s, _ in bad], _chunks_for(chunks, bad))

    def words(self, ref: Reference, y16: np.ndarray, chunks: list[dict]) -> Gate:
        if ref.words is None:
            ref.words, ref.language = self.transcriber.words(ref.x16)
        after, _ = self.transcriber.words(y16, ref.language)
        if len(ref.words) < 5:
            return Gate("words", SKIP, {"words_before": len(ref.words)}, reason="too_few_words")
        wer, diffs = metrics.compare_words(ref.words, after, CONFIDENT_PROBABILITY)
        changed = _changed_words(diffs)
        values = {"wer": round(wer, 3), "words_before": len(ref.words), "words_after": len(after),
                  "changed_words": len(changed), "language": ref.language}
        limits = {"wer_fail": WER_FAIL, "wer_warn": WER_WARN, "changed_words": 0}
        if changed or wer > WER_FAIL:
            status = FAIL
        elif wer > WER_WARN:
            status = WARN
        else:
            status = PASS
        # Для ремонту — місця впевнених замін; якщо їх немає, а WER високий, — усі розбіжності.
        spots = changed if changed else (diffs if status == FAIL else [])
        details = [
            {"start_s": round(d.start, 2), "end_s": round(d.end, 2), "kind": d.kind,
             "before": d.before, "after": d.after, "changed": d in changed}
            for d in diffs[:50]
        ]
        spans = [(d.start, d.end) for d in spots]
        return Gate("words", status, values, limits, [round(d.start, 1) for d in spots], _chunks_for(chunks, spans), details)

    def voice(self, ref: Reference, y16: np.ndarray, chunks: list[dict]) -> Gate:
        overall, windows = metrics.voice_similarity(self.encoder, ref.x16, y16, ref.mask, VOICE_WINDOW_S)
        if overall is None:
            return Gate("voice", SKIP, reason="too_little_speech")
        low = [(s, c) for s, c in windows if c < VOICE_WINDOW_FAIL]
        values = {"similarity": round(overall, 3),
                  "window_min": round(min((c for _, c in windows), default=overall), 3)}
        limits = {"fail": VOICE_FAIL, "window_fail": VOICE_WINDOW_FAIL, "warn": VOICE_WARN}
        if overall < VOICE_FAIL or low:
            status = FAIL
        elif overall < VOICE_WARN:
            status = WARN
        else:
            status = PASS
        spans = [(s, s + VOICE_WINDOW_S) for s, _ in low]
        details = [{"start_s": round(s, 1), "similarity": round(c, 3)} for s, c in windows]
        return Gate("voice", status, values, limits, [round(s, 1) for s, _ in low], _chunks_for(chunks, spans), details)

    def sync(self, ref: Reference, y48: np.ndarray, chunks: list[dict]) -> Gate:
        overall = align.measure(y48, ref.x48, max_ms=SYNC_SEARCH_MS)
        windows = []
        step = int(10 * SR48)
        for start in range(0, min(y48.size, ref.x48.size) - step + 1, step):
            o = align.measure(y48[start : start + step], ref.x48[start : start + step], max_ms=SYNC_SEARCH_MS)
            if o.correlation >= SYNC_WINDOW_MIN_CORRELATION:
                windows.append((start / SR48, o.ms))
        worst_ms = max((abs(ms) for _, ms in windows), default=0.0)
        values = {"offset_ms": round(overall.ms, 2), "correlation": round(overall.correlation, 3),
                  "window_max_ms": round(worst_ms, 2), "windows": len(windows)}
        limits = {"max_ms": SYNC_MAX_MS, "window_warn_ms": SYNC_WINDOW_WARN_MS, "window_fail_ms": SYNC_WINDOW_FAIL_MS}
        far = [(s, s + 10) for s, ms in windows if abs(ms) > SYNC_WINDOW_FAIL_MS]
        if abs(overall.ms) > SYNC_MAX_MS or far:
            status = FAIL
        elif worst_ms > SYNC_WINDOW_WARN_MS:
            status = WARN
        else:
            status = PASS
        details = [{"start_s": round(s, 1), "offset_ms": round(ms, 2)} for s, ms in windows]
        return Gate("sync", status, values, limits, [round(s, 1) for s, _ in far], _chunks_for(chunks, far), details)

    def loudness(self, path: Path | None, y48: np.ndarray, target: float, ceiling: float) -> Gate:
        limits = {"target_lufs": target, "tolerance_lu": LOUDNESS_TOLERANCE_LU, "max_true_peak_dbtp": ceiling}
        if path is not None:
            # Остаточні цифри — від ffmpeg (EBU R128) по записаному файлу, а не з нашого коду.
            measured = media.loudness(path)
            lufs, tp, source = measured.integrated_lufs, measured.true_peak_dbtp, "ffmpeg"
        else:
            from voice_normal import loudness as lm

            lufs, tp, source = lm.integrated_lufs(y48), lm.true_peak_db(y48), "internal"
        values = {"lufs": round(lufs, 2), "true_peak_dbtp": round(tp, 2), "measured_by": source}
        ok = abs(lufs - target) <= LOUDNESS_TOLERANCE_LU and tp <= ceiling
        return Gate("loudness", PASS if ok else FAIL, values, limits)

    def seams(self, ref: Reference, y48: np.ndarray, chunks: list[dict]) -> Gate:
        if len(chunks) < 2:
            return Gate("seams", SKIP, reason="single_piece")
        details, status, where, bad_chunks = [], PASS, [], set()
        for i in range(1, len(chunks)):
            prev, cur = chunks[i - 1], chunks[i]
            at = int(round(cur["start_s"] * SR48))
            click, level = metrics.click_score(y48, at)
            floor_a = metrics.background_floor(y48, ref.mask, int(prev["start_s"] * SR48), at)
            floor_b = metrics.background_floor(y48, ref.mask, at, int(cur["end_s"] * SR48))
            jump = None
            if floor_a is not None and floor_b is not None:
                jump = floor_b - floor_a
                orig_a = metrics.background_floor(ref.x48, ref.mask, int(prev["start_s"] * SR48), at)
                orig_b = metrics.background_floor(ref.x48, ref.mask, at, int(cur["end_s"] * SR48))
                # Якщо фон змінювався вже в оригіналі (людина пересіла), це не вада шва.
                if orig_a is not None and orig_b is not None and min(orig_a, orig_b) > -90:
                    jump -= orig_b - orig_a
            seam_status = PASS
            audible = level >= CLICK_MIN_LEVEL_DBFS
            if (audible and click >= CLICK_FAIL_DB) or (jump is not None and abs(jump) > FLOOR_FAIL_DB):
                seam_status = FAIL
            elif (audible and click >= CLICK_WARN_DB) or (jump is not None and abs(jump) > FLOOR_WARN_DB):
                seam_status = WARN
            if seam_status == FAIL:
                status = FAIL
                where.append(round(cur["start_s"], 2))
                bad_chunks.update({i - 1, i})
            elif seam_status == WARN and status == PASS:
                status = WARN
            details.append({"seam": i, "at_s": round(cur["start_s"], 2), "click_db": round(click, 1),
                            "click_level_dbfs": round(level, 1),
                            "floor_jump_db": None if jump is None else round(jump, 1), "status": seam_status})
        values = {"seams": len(chunks) - 1,
                  "worst_click_db": max(d["click_db"] for d in details),
                  "worst_floor_jump_db": max((abs(d["floor_jump_db"]) for d in details if d["floor_jump_db"] is not None), default=None)}
        limits = {"click_fail_db": CLICK_FAIL_DB, "click_warn_db": CLICK_WARN_DB,
                  "floor_fail_db": FLOOR_FAIL_DB, "floor_warn_db": FLOOR_WARN_DB}
        return Gate("seams", status, values, limits, where, sorted(bad_chunks), details)

    def echo(self, ref: Reference, y48: np.ndarray) -> Gate:
        level = echo_level(y48)
        values = {"echo": round(level, 3), "echo_before": round(ref.echo, 3)}
        limits = {"fail": ECHO_FAIL, "warn": ECHO_WARN, "over_input": ECHO_OVER_INPUT}
        added = level > ref.echo + ECHO_OVER_INPUT
        if added and level >= ECHO_FAIL:
            status = FAIL
        elif added and level >= ECHO_WARN:
            status = WARN
        else:
            status = PASS
        return Gate("echo", status, values, limits)

    def length(self, ref: Reference, y48: np.ndarray, path: Path | None, video: tuple[Path, Path] | None) -> Gate:
        written = y48.size
        if path is not None:
            import soundfile as sf

            from voice_normal.paths import long_path

            written = sf.info(long_path(path)).frames
        values = {"expected_samples": int(ref.x48.size), "samples": int(written)}
        ok = written == ref.x48.size
        if video is not None:
            original, cleaned = media.probe(video[0]), media.probe(video[1])
            values["video_duration_s"] = round(cleaned.duration_s, 3)
            values["original_duration_s"] = round(original.duration_s, 3)
            ok = ok and abs(cleaned.duration_s - original.duration_s) <= VIDEO_DURATION_TOLERANCE_S
        return Gate("length", PASS if ok else FAIL, values, {"video_tolerance_s": VIDEO_DURATION_TOLERANCE_S})
