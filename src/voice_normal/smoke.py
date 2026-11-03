"""Перевірка після встановлення: кожна модель вантажиться й видає результат.

Запускається окремим процесом (`python -m voice_normal.smoke cpu`) і друкує JSON-список.
"""

from __future__ import annotations

import json
import sys
import time
import traceback


def _check(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        return {"name": name, "ok": True, "detail": detail, "seconds": round(time.perf_counter() - started, 2)}
    except Exception as exc:  # noqa: BLE001 — будь-яка помилка тут означає "не працює"
        last = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return {"name": name, "ok": False, "detail": last, "seconds": round(time.perf_counter() - started, 2)}


def _test_tone(seconds: float, sr: int):
    import numpy as np

    # Схоже на голос: основний тон із гармоніками, повільна амплітудна модуляція й трохи шуму.
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    noise = np.random.default_rng(0).normal(0, 0.02, t.size)
    return (0.3 * voice * envelope / 3 + noise).astype(np.float32)


def check_torch(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__}: CUDA is not available")
    return f"torch {torch.__version__}, threads={torch.get_num_threads()}" + (
        f", {torch.cuda.get_device_name(0)}" if device == "cuda" else ""
    )


def check_onnx(model_id):
    def run():
        import onnxruntime as ort

        from voice_normal import models, paths

        model = models.get_model(model_id)
        session = ort.InferenceSession(
            paths.long_path(models.file_path(model, model.files[0])), providers=["CPUExecutionProvider"]
        )
        inputs = ", ".join(f"{i.name}{i.shape}" for i in session.get_inputs())
        return f"onnxruntime {ort.__version__}: {inputs}"

    return run


def check_whisper(device):
    from faster_whisper import WhisperModel

    from voice_normal import models

    model = models.get_model("whisper-small")
    root = models.file_path(model, model.files[0]).parent
    compute = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel(str(root), device=device, compute_type=compute)
    segments, info = whisper.transcribe(_test_tone(3.0, 16_000), language="uk", beam_size=1)
    list(segments)
    return f"faster-whisper ({compute}), {info.duration:.1f} s"


def check_sidon(device):
    from voice_normal.sidon import INPUT_SR, OUTPUT_SR, Sidon

    started = time.perf_counter()
    sidon = Sidon(device)
    load = time.perf_counter() - started
    seconds = 5.0
    wav = _test_tone(seconds, INPUT_SR)
    started = time.perf_counter()
    out = sidon.restore(wav)
    elapsed = time.perf_counter() - started
    ratio = out.size / wav.size
    if abs(ratio - OUTPUT_SR / INPUT_SR) > 0.1:
        raise RuntimeError(f"unexpected output length {out.size} for {wav.size} samples")
    return f"load {load:.1f} s, {seconds:.0f} s audio in {elapsed:.1f} s (x{seconds / elapsed:.2f} realtime), out/in={ratio:.4f}"


def main(argv: list[str]) -> int:
    device = argv[0] if argv else "cpu"
    results = [
        _check("torch", lambda: check_torch(device)),
        _check("dnsmos", check_onnx("dnsmos")),
        _check("ecapa", check_onnx("ecapa")),
        _check("whisper", lambda: check_whisper(device)),
        _check("sidon", lambda: check_sidon(device)),
    ]
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
