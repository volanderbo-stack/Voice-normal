"""Вимірювання для гейтів: слова (Whisper), голос (ECAPA), шви, фон.

Кожна функція порівнює оригінал і результат (моно, 48 кГц, однакова довжина) і
повертає числа; рішення "пройдено / ні" ухвалює verify.py.
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from voice_normal import models, paths

SR48, SR16 = 48_000, 16_000


def to16(x48: np.ndarray) -> np.ndarray:
    return resample_poly(x48.astype(np.float64), 1, 3).astype(np.float32)


# ---------- слова ----------


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float


class Transcriber:
    """faster-whisper small; вантажиться один раз на всі раунди перевірки."""

    def __init__(self, device: str = "cpu"):
        from faster_whisper import WhisperModel

        model = models.get_model("whisper-small")
        root = models.file_path(model, model.files[0]).parent
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(str(root), device=device, compute_type=compute)

    def words(self, x16: np.ndarray, language: str | None = None) -> tuple[list[Word], str]:
        segments, info = self.model.transcribe(
            x16,
            language=language,
            beam_size=5,
            word_timestamps=True,
            # Без опори на попередній текст: інакше одна помилка тягне за собою наступні,
            # і порівняння двох розшифровок показувало б різницю, якої в звуці немає.
            condition_on_previous_text=False,
            vad_filter=True,
        )
        result = []
        for segment in segments:
            for w in segment.words or []:
                text = normalize_word(w.word)
                if text:
                    result.append(Word(float(w.start), float(w.end), text, float(w.probability)))
        return result, info.language


def normalize_word(word: str) -> str:
    # Регістр, пунктуація й апострофи різного вигляду не мають значення для порівняння.
    kept = [c for c in word.lower() if unicodedata.category(c)[0] in "LN"]
    return "".join(kept).replace("ё", "е")


@dataclass(frozen=True)
class WordDiff:
    kind: str  # replace | delete | insert
    start: float
    end: float
    before: str
    after: str
    confident: bool


def compare_words(before: list[Word], after: list[Word], confident_probability: float) -> tuple[float, list[WordDiff]]:
    """Наближений WER (через difflib, лінійно за довжиною) і перелік розбіжностей."""
    a, b = [w.text for w in before], [w.text for w in after]
    if not a:
        return (0.0 if not b else 1.0), []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    errors, diffs = 0, []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        errors += max(i2 - i1, j2 - j1)
        span = before[i1:i2] or after[j1:j2]
        # "Впевнена" заміна: обидві розшифровки чують своє слово чітко — отже, змінився
        # сам звук, а не розпізнавання шумного оригіналу.
        confident = (
            tag == "replace"
            and min(w.probability for w in before[i1:i2]) >= confident_probability
            and min(w.probability for w in after[j1:j2]) >= confident_probability
        )
        diffs.append(WordDiff(tag, span[0].start, span[-1].end, " ".join(a[i1:i2]), " ".join(b[j1:j2]), confident))
    return errors / len(a), diffs


# ---------- голос ----------

_MEL_BINS = 80
_FRAME, _SHIFT, _NFFT = 400, 160, 512


def _mel_matrix() -> np.ndarray:
    # Трикутні фільтри в мел-шкалі 1127·ln(1 + f/700), 20 Гц..8 кГц — як у Kaldi fbank,
    # на якому вчилась модель WeSpeaker.
    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f) / 700.0)

    bins = _NFFT // 2
    freqs = np.arange(bins) * SR16 / _NFFT
    lo, hi = mel(20.0), mel(SR16 / 2)
    edges = np.linspace(lo, hi, _MEL_BINS + 2)
    m = mel(freqs)
    left, center, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    weights = np.maximum(0.0, np.minimum(up, down))
    return np.concatenate([weights, np.zeros((_MEL_BINS, 1))], axis=1)  # стовпчик Найквіста


_MEL = None


def fbank(x16: np.ndarray) -> np.ndarray:
    global _MEL
    if _MEL is None:
        _MEL = _mel_matrix()
    x = x16.astype(np.float64) * 32768.0
    count = 1 + (x.size - _FRAME) // _SHIFT
    if count < 1:
        return np.zeros((0, _MEL_BINS), dtype=np.float32)
    idx = np.arange(_FRAME)[None, :] + _SHIFT * np.arange(count)[:, None]
    frames = x[idx]
    frames -= frames.mean(axis=1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames *= np.hamming(_FRAME)
    power = np.abs(np.fft.rfft(frames, n=_NFFT)) ** 2
    feats = np.log(np.maximum(power @ _MEL.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


class SpeakerEncoder:
    def __init__(self):
        import onnxruntime as ort

        model = models.get_model("ecapa")
        self.session = ort.InferenceSession(
            paths.long_path(models.file_path(model, model.files[0])), providers=["CPUExecutionProvider"]
        )

    def embed(self, x16: np.ndarray) -> np.ndarray | None:
        feats = fbank(x16)
        if feats.shape[0] < 100:  # менше секунди — ембединг ненадійний
            return None
        emb = self.session.run(None, {"feats": feats[None]})[0][0]
        return emb / (np.linalg.norm(emb) or 1.0)


def speech_only(x16: np.ndarray, mask: np.ndarray, frame: int = 160) -> np.ndarray:
    keep = np.repeat(mask, frame)[: x16.size]
    return x16[: keep.size][keep]


def voice_similarity(
    encoder: SpeakerEncoder, before16: np.ndarray, after16: np.ndarray, mask: np.ndarray, window_s: float
) -> tuple[float | None, list[tuple[float, float]]]:
    """Косинус ембедингів мовця по всьому запису та по вікнах (початок_с, косинус)."""
    whole_a = encoder.embed(speech_only(before16, mask))
    whole_b = encoder.embed(speech_only(after16, mask))
    overall = float(whole_a @ whole_b) if whole_a is not None and whole_b is not None else None
    step = int(window_s * SR16)
    frames = step // 160
    windows = []
    for start in range(0, before16.size - step // 2, step):
        m = mask[start // 160 : start // 160 + frames]
        if m.mean() < 0.3:
            continue
        a = encoder.embed(speech_only(before16[start : start + step], m))
        b = encoder.embed(speech_only(after16[start : start + step], m))
        if a is not None and b is not None:
            windows.append((start / SR16, float(a @ b)))
    return overall, windows


# ---------- шви й фон ----------


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-20)))


_HP_SOS = None
CLICK_FRAME = 48  # 1 мс


def _highpass(x: np.ndarray) -> np.ndarray:
    global _HP_SOS
    from scipy.signal import butter, sosfiltfilt

    if _HP_SOS is None:
        _HP_SOS = butter(4, 4000, btype="highpass", fs=SR48, output="sos")
    return sosfiltfilt(_HP_SOS, x.astype(np.float64))


def click_score(y48: np.ndarray, at: int, half_ms: float = 10.0) -> tuple[float, float]:
    """Наскільки 1-мс кадр вище 4 кГц біля семпла at гучніший за сусідні кадри (дБ),
    і рівень цього кадру (дБFS).

    Клацання — короткий широкосмуговий імпульс, ізольований у часі. Шиплячі теж мають
    багато високих частот, але тягнуться десятки мілісекунд, тож відносно сусідів
    (2–8 мс з кожного боку) не виділяються.
    """
    pad = int(0.05 * SR48)
    lo, hi = max(at - pad, 0), min(at + pad, y48.size)
    hp = _highpass(y48[lo:hi])
    n = hp.size // CLICK_FRAME
    if n < 30:
        return 0.0, -200.0
    energy = np.mean(hp[: n * CLICK_FRAME].reshape(n, CLICK_FRAME) ** 2, axis=1)
    k0 = (at - lo) // CLICK_FRAME
    h = max(1, int(half_ms * SR48 / 1000) // CLICK_FRAME)
    best, level = -200.0, -200.0
    for k in range(max(k0 - h, 8), min(k0 + h + 1, n - 8)):
        neighbours = np.concatenate([energy[k - 8 : k - 1], energy[k + 2 : k + 9]])
        ratio = _db(float(energy[k])) - _db(float(np.median(neighbours)))
        if ratio > best:
            best, level = ratio, _db(float(energy[k]))
    return best, level


def _frame_db(y48: np.ndarray, frame: int = 480) -> np.ndarray:
    count = y48.size // frame
    power = np.mean(y48[: count * frame].astype(np.float64).reshape(count, frame) ** 2, axis=1)
    return 10 * np.log10(np.maximum(power, 1e-20))


def background_floor(y48: np.ndarray, mask: np.ndarray, start: int, end: int) -> float | None:
    """Рівень фону (дБFS) на відрізку [start, end) семплів 48 кГц: медіана кадрів 10 мс
    у паузах, без 50 мс біля країв фраз, де ще звучить хвіст голосу."""
    db = _frame_db(y48[start:end])
    m = np.zeros(db.size, dtype=bool)
    first = start // 480
    part = mask[first : first + db.size]
    m[: part.size] = part
    # Розширюємо мову на 5 кадрів в обидва боки.
    grown = np.convolve(m.astype(int), np.ones(11, dtype=int), mode="same") > 0
    pauses = db[~grown]
    if pauses.size < 50:  # менше пів секунди пауз — оцінка ненадійна
        return None
    return float(np.median(pauses))
