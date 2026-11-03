"""voice-normal plan: маршрут обробки, найважчі 150 с для проби, оцінка часу."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voice_normal import __version__, models
from voice_normal.analyze import Analysis, Window
from voice_normal.i18n import t
from voice_normal.media import MediaInfo
from voice_normal.paths import long_path

PLAN_VERSION = 1
TARGET_LUFS = -16.0
TARGET_TRUE_PEAK = -1.0
PROBE_SECONDS = 150.0
# Якщо мови не набагато більше за пробу, пробувати окремо немає сенсу.
WHOLE_FILE_PROBE_LIMIT = 180.0

# Пороги, за якими запис уже досить чистий і Sidon лише зашкодить
# (генеративне відновлення завжди трохи змінює голос).
CLEAN_OVRL = 3.3
CLEAN_BAK = 3.9
CLEAN_SNR_DB = 35.0
CLIPPING_FRACTION = 1e-4
NARROW_BANDWIDTH_HZ = 12_000
REVERB_RT60_S = 0.35
NOISY_SNR_DB = 25.0
NOISY_BAK = 3.5
LOW_SPEECH_RATIO = 0.2


@dataclass
class Reason:
    code: str
    values: dict[str, float | str] = field(default_factory=dict)


@dataclass
class Plan:
    version: int
    tool: str
    input: dict
    targets: dict
    analysis: dict
    route: str
    reasons: list[Reason]
    warnings: list[Reason]
    probe: list[tuple[float, float]]
    probe_whole_file: bool
    estimate: dict

    def to_dict(self) -> dict:
        return asdict(self)


def choose_route(a: Analysis, duration_s: float) -> tuple[str, list[Reason], list[Reason]]:
    reasons: list[Reason] = []
    if a.snr_db is not None and a.snr_db < NOISY_SNR_DB:
        reasons.append(Reason("noise", {"snr": a.snr_db}))
    elif a.dnsmos_bak is not None and a.dnsmos_bak < NOISY_BAK:
        reasons.append(Reason("background", {"bak": a.dnsmos_bak}))
    if a.rt60_s is not None and a.rt60_s >= REVERB_RT60_S:
        reasons.append(Reason("reverb", {"rt60": a.rt60_s}))
    if a.clipped_fraction >= CLIPPING_FRACTION:
        reasons.append(Reason("clipping", {"percent": round(100 * a.clipped_fraction, 3)}))
    if a.bandwidth_hz < NARROW_BANDWIDTH_HZ:
        reasons.append(Reason("bandwidth", {"khz": round(a.bandwidth_hz / 1000, 1)}))
    if a.dnsmos_ovrl is not None and a.dnsmos_ovrl < CLEAN_OVRL:
        reasons.append(Reason("quality", {"ovrl": a.dnsmos_ovrl}))

    clean = (
        not reasons
        and (a.dnsmos_bak or 0) >= CLEAN_BAK
        and (a.snr_db if a.snr_db is not None else 0) >= CLEAN_SNR_DB
    )
    if clean:
        route = "master_only"
        reasons = [Reason("clean", {"ovrl": a.dnsmos_ovrl or 0, "snr": a.snr_db or 0})]
    else:
        route = "restore"
        if not reasons:
            reasons.append(Reason("not_clean_enough", {"ovrl": a.dnsmos_ovrl or 0, "snr": a.snr_db or 0}))

    warnings: list[Reason] = []
    if a.speech_ratio < LOW_SPEECH_RATIO:
        warnings.append(Reason("low_speech", {"percent": round(100 * a.speech_ratio)}))
    if duration_s < 3:
        warnings.append(Reason("short", {"seconds": round(duration_s, 1)}))
    if a.channel_strategy.startswith("channel"):
        warnings.append(Reason("one_channel", {"channel": a.channel_strategy.removeprefix("channel")}))
    if a.dnsmos_ovrl is None:
        warnings.append(Reason("no_dnsmos"))
    return route, reasons, warnings


def difficulty(w: Window) -> float:
    score = 0.0
    if w.ovrl is not None:
        score += 4.5 - w.ovrl
    if w.snr_db is not None:
        score += max(0.0, 30.0 - w.snr_db) / 10.0
    if w.clipped:
        score += 2.0
    return score


def choose_probe(windows: list[Window], duration_s: float) -> tuple[list[tuple[float, float]], bool]:
    speechy = [w for w in windows if w.speech_ratio >= 0.4]
    if not speechy or sum(w.end_s - w.start_s for w in speechy) <= WHOLE_FILE_PROBE_LIMIT:
        return [(0.0, round(duration_s, 3))], True
    picked, total = [], 0.0
    # При однаковій складності беремо раніші — результат стабільний між запусками.
    for w in sorted(speechy, key=lambda w: (-difficulty(w), w.start_s)):
        if total >= PROBE_SECONDS:
            break
        picked.append(w)
        total += w.end_s - w.start_s
    merged: list[list[float]] = []
    for w in sorted(picked, key=lambda w: w.start_s):
        if merged and abs(merged[-1][1] - w.start_s) < 1e-6:
            merged[-1][1] = w.end_s
        else:
            merged.append([w.start_s, w.end_s])
    return [(s, e) for s, e in merged], False


def read_speed() -> tuple[float, str]:
    """Швидкість Sidon (у скільки разів швидше за реальний час), заміряна під час install."""
    try:
        with open(long_path(models.benchmark_path()), encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data["sidon_realtime_x"]), str(data.get("device", "cpu"))
    except (OSError, ValueError, KeyError):
        manifest = models.read_manifest() or {}
        return 1.0, str(manifest.get("device", "cpu"))


def estimate(route: str, duration_s: float, probe_s: float, probe_whole: bool) -> dict:
    speed, device = read_speed()
    # Аналіз і перевірка (розпізнавання мови двічі, голос, DNSMOS) — приблизно чверть тривалості.
    overhead = 0.25 * duration_s
    if route == "master_only":
        seconds = 0.1 * duration_s + overhead
    else:
        restore = duration_s / speed
        probe = 0.0 if probe_whole else probe_s / speed
        seconds = restore + probe + overhead
    return {"seconds": round(seconds), "device": device, "sidon_realtime_x": round(speed, 2)}


def build_plan(info: MediaInfo, a: Analysis) -> Plan:
    route, reasons, warnings = choose_route(a, info.duration_s)
    probe, whole = choose_probe(a.windows, info.duration_s)
    probe_s = sum(e - s for s, e in probe)
    return Plan(
        version=PLAN_VERSION,
        tool=f"voice-normal {__version__}",
        input=asdict(info),
        targets={"lufs": TARGET_LUFS, "true_peak_dbtp": TARGET_TRUE_PEAK},
        analysis=a.to_dict(),
        route=route,
        reasons=reasons,
        warnings=warnings,
        probe=probe,
        probe_whole_file=whole,
        estimate=estimate(route, info.duration_s, probe_s, whole),
    )


# ---------- текстовий вигляд ----------


def clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt(value, pattern: str = "{:.1f}", missing: str | None = None) -> str:
    return (missing or t("plan.unknown")) if value is None else pattern.format(value)


def render(plan: Plan) -> str:
    i, a = plan.input, plan.analysis
    lines = [f"# {t('plan.title', name=Path(i['path']).name)}", ""]
    lines.append(
        t(
            "plan.file",
            duration=clock(i["duration_s"]),
            rate=i["sample_rate"] / 1000,
            channels=i["channels"],
            codec=i["codec"],
            video=t("plan.yes") if i["has_video"] else t("plan.no"),
        )
    )
    lines += ["", f"## {t('plan.section.analysis')}", ""]
    lines.append("- " + t("plan.loudness", lufs=a["integrated_lufs"], peak=a["true_peak_dbtp"],
                          target=plan.targets["lufs"], target_peak=plan.targets["true_peak_dbtp"]))
    lines.append("- " + t("plan.speech", percent=round(100 * a["speech_ratio"]), seconds=round(a["speech_seconds"])))
    lines.append("- " + t("plan.noise", floor=_fmt(a["noise_floor_db"]), snr=_fmt(a["snr_db"])))
    if a["clipped_samples"]:
        lines.append("- " + t("plan.clipping", percent=f"{100 * a['clipped_fraction']:.3f}"))
    else:
        lines.append("- " + t("plan.clipping.none"))
    lines.append("- " + t("plan.bandwidth", khz=a["bandwidth_hz"] / 1000))
    if a["rt60_s"] is None:
        lines.append("- " + t("plan.reverb.unknown"))
    else:
        lines.append("- " + t("plan.reverb", rt60=f"{a['rt60_s']:.2f}"))
    lines.append(
        "- " + t("plan.dnsmos", sig=_fmt(a["dnsmos_sig"], "{:.2f}"), bak=_fmt(a["dnsmos_bak"], "{:.2f}"),
                 ovrl=_fmt(a["dnsmos_ovrl"], "{:.2f}"))
    )
    if a["channel_strategy"] not in ("mono", "mean"):
        lines.append("- " + t("plan.channels.one", channel=a["channel_strategy"].removeprefix("channel"),
                              levels=", ".join(f"{x:.1f}" for x in a["channel_levels_db"])))

    lines += ["", f"## {t('plan.section.route', route=t('plan.route.' + plan.route))}", ""]
    lines += ["- " + t("plan.reason." + r.code, **r.values) for r in plan.reasons]
    if plan.warnings:
        lines += ["", f"## {t('plan.section.warnings')}", ""]
        lines += ["- " + t("plan.warning." + w.code, **w.values) for w in plan.warnings]

    lines += ["", f"## {t('plan.section.probe')}", ""]
    if plan.probe_whole_file:
        lines.append(t("plan.probe.whole"))
    else:
        total = sum(e - s for s, e in plan.probe)
        lines.append(t("plan.probe.parts", total=clock(total)))
        lines += [f"- {clock(s)}–{clock(e)}" for s, e in plan.probe]

    est = plan.estimate
    lines += ["", f"## {t('plan.section.estimate')}", ""]
    lines.append(t("plan.estimate", time=clock(est["seconds"]), device=est["device"].upper(), speed=est["sidon_realtime_x"]))
    return "\n".join(lines) + "\n"


def output_paths(source: Path, out_dir: Path) -> tuple[Path, Path]:
    return out_dir / f"{source.stem}.plan.json", out_dir / f"{source.stem}.plan.md"


def write(plan: Plan, source: Path, out_dir: Path) -> tuple[Path, Path]:
    from voice_normal.console import dumps, write_text

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_paths(source, out_dir)
    write_text(json_path, dumps(plan.to_dict()))
    write_text(md_path, render(plan))
    return json_path, md_path
