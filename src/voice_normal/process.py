"""voice-normal process: проба -> відновлення -> тембр -> зведення -> перевірка -> ремонт -> файли.

Після кожного зведення результат проходить гейти (verify.py). Якщо щось не так, до трьох
раундів ремонту: переробити окремі фрагменти з ширшим контекстом, перенести шов в іншу
паузу, звести ще раз, а якщо й це не допомогло — лишити тільки зведення без відновлення.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from voice_normal import align, analyze, loudness, media, models, report, restore, timbre, tools, verify
from voice_normal import plan as planner
from voice_normal.console import dumps, write_text
from voice_normal.i18n import t
from voice_normal.paths import long_path

SR48 = 48_000
PROBE_MIN_GAIN = 0.1  # наскільки OVRL проби має вирости, щоб відновлення мало сенс
MAX_REPAIR_ROUNDS = 3
REPAIR_MARGINS_S = (3.0, 5.0)  # контекст фрагмента в 1-му і 2-му раундах ремонту
REMASTER_STEP_DB = 0.3


@dataclass
class Outputs:
    audio: str
    video: str | None = None
    plan: str | None = None
    summary: str | None = None
    report: str | None = None


@dataclass
class ProcessResult:
    ok: bool
    route: str
    input: dict
    length: dict
    probe: dict = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    sync: dict = field(default_factory=dict)
    timbre: dict = field(default_factory=dict)
    mastering: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    outputs: Outputs | None = None
    notes: list[str] = field(default_factory=list)
    verdict: str = verify.FAIL
    gates: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def output_names(source: Path, out_dir: Path) -> dict[str, Path]:
    stem = source.stem
    return {
        "audio": out_dir / f"{stem}.cleaned.wav",
        "video": out_dir / f"{stem}.cleaned{source.suffix.lower()}",
        "summary": out_dir / f"{stem}.report.json",
        "report": out_dir / f"{stem}.report.md",
    }


def load_or_make_plan(source: Path, out_dir: Path, log) -> tuple[media.MediaInfo, planner.Plan, Path]:
    json_path, _ = planner.output_paths(source, out_dir)
    info = media.probe(source)
    try:
        with open(long_path(json_path), encoding="utf-8") as fh:
            data = json.load(fh)
        same = data["input"]["size_bytes"] == info.size_bytes and abs(data["input"]["duration_s"] - info.duration_s) < 1e-3
        if same and data.get("version") == planner.PLAN_VERSION:
            log(t("process.plan_reused", path=json_path))
            return info, _plan_from_dict(data), json_path
    except (OSError, ValueError, KeyError):
        pass
    log(t("process.analyzing"))
    info, analysis = analyze.analyze(source)
    plan = planner.build_plan(info, analysis)
    json_path, _ = planner.write(plan, source, out_dir)
    return info, plan, json_path


def _plan_from_dict(data: dict) -> planner.Plan:
    fields = dict(data)
    fields["reasons"] = [planner.Reason(**r) for r in data["reasons"]]
    fields["warnings"] = [planner.Reason(**r) for r in data["warnings"]]
    fields["probe"] = [tuple(p) for p in data["probe"]]
    return planner.Plan(**fields)


def read_mono_48k(source: Path, info: media.MediaInfo, strategy: str) -> tuple[np.ndarray, int]:
    """Моно 48 кГц і точна кількість семплів входу в його рідній частоті."""
    audio = media.decode(source)
    native = audio.shape[0]
    if strategy.startswith("channel"):
        mono = audio[:, int(strategy.removeprefix("channel")) - 1]
    elif audio.shape[1] > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    if info.sample_rate == SR48:
        return mono.astype(np.float32), native
    # scipy, а не ffmpeg: однаковий результат на будь-якій ОС і версії ffmpeg.
    g = np.gcd(SR48, info.sample_rate)
    return resample_poly(mono, SR48 // g, info.sample_rate // g).astype(np.float32), native


def speech_mask_48k(x48: np.ndarray) -> np.ndarray:
    x16 = resample_poly(x48.astype(np.float64), 1, 3).astype(np.float32)
    return analyze.speech_mask(x16, x16.size // analyze.HOP)


def _mean_ovrl(x48: np.ndarray) -> float | None:
    x16 = resample_poly(x48.astype(np.float64), 1, 3).astype(np.float32)
    scores = analyze.dnsmos(x16)
    return float(np.mean([s[3] for s in scores])) if scores else None


def run_probe(x48: np.ndarray, speech: np.ndarray, segments, model, log) -> dict:
    originals, restored = [], []
    for start_s, end_s in segments:
        a, b = int(start_s * SR48), min(int(end_s * SR48), x48.size)
        fa, fb = a // restore.FRAME16 // 3, b // restore.FRAME16 // 3
        part, _ = restore.restore(x48[a:b], speech[fa:fb], model)
        originals.append(x48[a:b])
        restored.append(part)
    before = _mean_ovrl(np.concatenate(originals))
    after = _mean_ovrl(np.concatenate(restored))
    log(t("process.probe_result", before=before or 0, after=after or 0))
    better = before is None or after is None or after >= before + PROBE_MIN_GAIN
    return {"segments": [list(s) for s in segments], "ovrl_before": before, "ovrl_after": after, "better": better}


def write_wav(path: Path, data: np.ndarray) -> None:
    import soundfile as sf

    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(long_path(tmp), np.clip(data, -1.0, 1.0), SR48, subtype="PCM_24")
    # Файл закрито всередині sf.write; лише тепер Windows дасть його перейменувати.
    os.replace(long_path(tmp), long_path(path))


def mux_video(video: Path, wav: Path, target: Path) -> None:
    ffmpeg = tools.find_tool("ffmpeg")
    codec = ["-c:a", "libopus", "-b:a", "192k"] if target.suffix.lower() == ".webm" else ["-c:a", "aac", "-b:a", "256k"]
    tmp = target.with_name(target.stem + ".part" + target.suffix)
    result = tools.run_tool(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(wav),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *codec, "-movflags", "+faststart", str(tmp)],
        timeout=None,
    )
    if result.returncode != 0:
        if os.path.exists(long_path(tmp)):
            os.remove(long_path(tmp))
        raise media.MediaError(result.stderr.strip()[-500:])
    os.replace(long_path(tmp), long_path(target))


def device() -> str:
    return (models.read_manifest() or {}).get("device", "cpu")


def _chunk_dicts(chunks: list[restore.Chunk]) -> list[dict]:
    return [c.to_dict() for c in chunks]


def plan_repair(gates: list[verify.Gate], route: str, round_index: int) -> list[str]:
    """Що робити з гейтами, які не пройшли: перелік дій для наступного раунду."""
    bad = {g.id: g for g in verify.failing(gates)}
    if set(bad) <= {"loudness"}:
        return ["remaster"]
    if route != "restore":
        # Лише підсилення: крім гучності, обробка тут нічого не могла зламати — ремонтувати нічого.
        return ["remaster"] if "loudness" in bad else []
    quality = bad.get("quality")
    globally_worse = quality is not None and quality.values["ovrl_after"] < quality.values["ovrl_before"] - verify.QUALITY_MAX_DROP
    if globally_worse or "echo" in bad or "length" in bad or round_index >= MAX_REPAIR_ROUNDS - 1:
        return ["master_only"]
    actions = []
    sync = bad.get("sync")
    if sync is not None and abs(sync.values["offset_ms"]) > verify.SYNC_MAX_MS:
        actions.append("resync")
    if "seams" in bad:
        actions.append("recut")
    if any(g.chunks for k, g in bad.items() if k not in ("seams", "loudness")) or "words" in bad or "voice" in bad:
        actions.append("redo")
    if "loudness" in bad:
        actions.append("remaster")
    return actions or ["master_only"]


def run(
    source: Path,
    out_dir: Path,
    log,
    force_route: str | None = None,
    chunk_s: float | None = None,
    verifier: verify.Verifier | None = None,
) -> ProcessResult:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    info, plan, plan_path = load_or_make_plan(source, out_dir, log)
    timings: dict[str, float] = {}

    x48, native_samples = read_mono_48k(source, info, plan.analysis["channel_strategy"])
    length = {"input_samples": native_samples, "input_rate": info.sample_rate,
              "output_samples": int(x48.size), "output_rate": SR48}
    route = force_route or plan.route
    result = ProcessResult(ok=False, route=route, input=asdict(info), length=length)
    speech = speech_mask_48k(x48)
    model = None

    if route == "restore":
        from voice_normal.sidon import Sidon

        t0 = time.perf_counter()
        log(t("process.loading_model", device=device().upper()))
        model = Sidon(device())
        timings["load_model"] = round(time.perf_counter() - t0, 1)

        if not plan.probe_whole_file and force_route is None:
            log(t("process.probe", seconds=round(sum(e - s for s, e in plan.probe))))
            t0 = time.perf_counter()
            result.probe = run_probe(x48, speech, plan.probe, model, log)
            timings["probe"] = round(time.perf_counter() - t0, 1)
            if not result.probe["better"]:
                route = result.route = "master_only"
                result.notes.append("probe_not_better")
                log(t("process.probe_fallback"))

    def progress(done: int, total: int, seconds: float) -> None:
        log(t("process.chunk", done=done, total=total, seconds=seconds))

    chunks: list[restore.Chunk] = []
    restored = restorer = None
    if route == "restore":
        t0 = time.perf_counter()
        restorer = restore.Restorer(x48, model)
        chunks = restore.plan_chunks(restorer.x16.size, speech, chunk_s or restore.CHUNK_S)
        log(t("process.restoring", count=len(chunks)))
        restored = restorer.render(chunks, progress=progress)
        timings["restore"] = round(time.perf_counter() - t0, 1)

    names = output_names(source, out_dir)
    verifier = verifier or verify.Verifier(device())
    reference = verifier.reference(x48, speech)
    ceiling = planner.TARGET_TRUE_PEAK
    actions: list[str] = ["initial"]
    timings["verify"] = 0.0

    for round_index in range(MAX_REPAIR_ROUNDS + 1):
        if route == "restore":
            signal, result.timbre = timbre.match(restored, x48, speech, plan.analysis["bandwidth_hz"])
        else:
            signal, result.timbre, chunks = x48, {}, []
        # Довжина виходу дорівнює довжині входу в семплах 48 кГц — завжди, за будь-якого маршруту.
        signal = align.fit_length(signal, x48.size)
        sync = align.measure(signal, x48)
        result.sync = {"offset_ms": round(sync.ms, 3), "correlation": round(sync.correlation, 3)}

        log(t("process.mastering"))
        mastered, result.mastering = loudness.master(signal, planner.TARGET_LUFS, ceiling)
        write_wav(names["audio"], mastered)

        log(t("process.verifying"))
        t0 = time.perf_counter()
        gates = verifier.check(
            reference, mastered, audio_path=names["audio"], chunks=_chunk_dicts(chunks),
            content_changed=route == "restore",
            target_lufs=planner.TARGET_LUFS, ceiling_dbtp=planner.TARGET_TRUE_PEAK,
        )
        timings["verify"] = round(timings["verify"] + time.perf_counter() - t0, 1)
        for gate in gates:
            log(t("process.gate", mark=report.MARK[gate.status], gate=t(f"gate.{gate.id}")))
        result.rounds.append({"round": round_index, "route": route, "actions": actions,
                              "gates": {g.id: g.status for g in gates}})
        if not verify.failing(gates) or round_index == MAX_REPAIR_ROUNDS:
            break
        actions = plan_repair(gates, route, round_index)
        if not actions:
            break
        log(t("process.repair", round=round_index + 1, actions=", ".join(t(f"repair.{a}") for a in actions)))
        t0 = time.perf_counter()
        bad = {g.id: g for g in verify.failing(gates)}
        if "master_only" in actions:
            route = result.route = "master_only"
            result.notes.append("restore_rejected")
            continue
        if "remaster" in actions:
            ceiling -= REMASTER_STEP_DB
        if "resync" in actions:
            restored = align.shift(restored, int(round(bad["sync"].values["offset_ms"] * SR48 / 1000)))
        redo: set[int] = set()
        if "recut" in actions:
            for seam in (d["seam"] for d in bad["seams"].details if d["status"] == verify.FAIL):
                moved = restore.recut(chunks, seam, speech)
                if moved is not None:
                    chunks = moved
                redo.update({seam - 1, seam})
        if "redo" in actions:
            for gate in bad.values():
                if gate.id not in ("seams", "loudness", "sync") or gate.chunks:
                    redo.update(gate.chunks or range(len(chunks)))
        if redo:
            margin = REPAIR_MARGINS_S[min(round_index, len(REPAIR_MARGINS_S) - 1)]
            restored = restorer.redo(restored, chunks, sorted(redo), margin, progress)
        timings[f"repair_{round_index + 1}"] = round(time.perf_counter() - t0, 1)

    result.chunks = _chunk_dicts(chunks)
    outputs = Outputs(audio=str(names["audio"]), plan=str(plan_path), summary=str(names["summary"]),
                      report=str(names["report"]))
    if info.has_video:
        log(t("process.muxing"))
        mux_video(source, names["audio"], names["video"])
        outputs.video = str(names["video"])
        gates = [verifier.length(reference, mastered, names["audio"], (source, names["video"])) if g.id == "length" else g
                 for g in gates]
    result.outputs = outputs
    result.gates = [g.to_dict() for g in gates]
    result.verdict = verify.worst(gates)
    result.ok = result.verdict != verify.FAIL
    timings["total"] = round(time.perf_counter() - started, 1)
    result.seconds = timings
    write_text(names["summary"], dumps(result.to_dict()))
    write_text(names["report"], report.render(result.to_dict()))
    return result
