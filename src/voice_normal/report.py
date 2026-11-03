"""report.md: підсумок перевірки людською мовою (uk/ru/en); report.json — ті самі дані повністю."""

from __future__ import annotations

from pathlib import Path

from voice_normal.i18n import t
from voice_normal.plan import clock

MARK = {"pass": "[OK]", "warn": "[??]", "fail": "[!!]", "skip": "[--]"}
MAX_LISTED = 12


def _num(value, fmt: str = "{:.2f}") -> str:
    return "—" if value is None else fmt.format(value)


def measured(gate: dict) -> str:
    v, gid = gate["values"], gate["id"]
    if gate["status"] == "skip":
        return t(f"report.skip.{gate.get('reason') or 'other'}")
    if not v:
        return ""
    if gid == "quality":
        return t("report.m.quality", **v)
    if gid == "words":
        return t("report.m.words", **v)
    if gid == "voice":
        return t("report.m.voice", **v)
    if gid == "sync":
        return t("report.m.sync", **v)
    if gid == "loudness":
        return t("report.m.loudness", **v)
    if gid == "seams":
        return t("report.m.seams", seams=v["seams"], click=_num(v["worst_click_db"], "{:.1f}"),
                 jump=_num(v["worst_floor_jump_db"], "{:.1f}"))
    if gid == "echo":
        return t("report.m.echo", **v)
    if gid == "length":
        text = t("report.m.length", **v)
        if "video_duration_s" in v:
            text += "; " + t("report.m.length_video", **v)
        return text
    return ""


def limits(gate: dict) -> str:
    if gate["status"] == "skip" or not gate["limits"]:
        return ""
    return t(f"report.l.{gate['id']}", **gate["limits"])


def render(result: dict) -> str:
    name = Path(result["input"]["path"]).name
    lines = [f"# {t('report.title', name=name)}", ""]
    lines.append(f"**{t('report.verdict.' + result['verdict'])}**")
    lines.append("")
    repairs = len(result["rounds"]) - 1
    lines.append(t("report.route", route=t("plan.route." + result["route"]), repairs=repairs))
    if "restore_rejected" in result["notes"]:
        lines.append("")
        lines.append(t("report.note.restore_rejected"))
    if "probe_not_better" in result["notes"]:
        lines.append("")
        lines.append(t("report.note.probe_not_better", **{k: _num(result["probe"].get(k)) for k in ("ovrl_before", "ovrl_after")}))

    lines += ["", f"## {t('report.section.gates')}", ""]
    lines.append(f"| {t('report.col.gate')} | {t('report.col.status')} | {t('report.col.measured')} | {t('report.col.limit')} |")
    lines.append("|---|---|---|---|")
    for gate in result["gates"]:
        lines.append(f"| {t('gate.' + gate['id'])} | {t('report.status.' + gate['status'])} | {measured(gate)} | {limits(gate)} |")

    problems = _problems(result["gates"])
    if problems:
        lines += ["", f"## {t('report.section.problems')}", ""]
        lines += problems

    if repairs:
        lines += ["", f"## {t('report.section.rounds')}", ""]
        for r in result["rounds"]:
            failed = [t("gate." + g) for g, s in r["gates"].items() if s == "fail"]
            actions = ", ".join(t("repair." + a) for a in r["actions"])
            lines.append("- " + t("report.round", round=r["round"], actions=actions,
                                  failed=", ".join(failed) or t("report.none")))

    lines += ["", f"## {t('report.section.processing')}", ""]
    if result["chunks"]:
        lines.append("- " + t("report.chunks", count=len(result["chunks"]),
                              cut=sum(c["cut_in_silence"] for c in result["chunks"])))
    if result["timbre"].get("applied"):
        bands = ", ".join(
            t("report.band", freq=t("unit.khz", value=f"{int(f) / 1000:g}") if int(f) >= 1000 else t("unit.hz", value=f), db=d)
            for f, d in result["timbre"]["bands_db"].items()
        )
        lines.append("- " + t("report.timbre", bands=bands))
    m = result["mastering"]
    if "gain_db" in m:
        lines.append("- " + t("report.mastering", gain=m["gain_db"], limiter=m.get("limiter_db", 0.0),
                              percent=m.get("limited_percent", 0.0)))
    lines.append("- " + t("report.time", total=clock(result["seconds"].get("total", 0)),
                          verify=clock(result["seconds"].get("verify", 0))))

    out = result["outputs"] or {}
    lines += ["", f"## {t('report.section.files')}", ""]
    lines.append(f"- {t('report.file.audio')}: `{out.get('audio')}`")
    if out.get("video"):
        lines.append(f"- {t('report.file.video')}: `{out['video']}`")
    lines.append(f"- {t('report.file.json')}: `{out.get('summary')}`")

    lines += ["", f"## {t('report.section.limits')}", ""]
    lines += ["- " + t(f"report.limit.{k}") for k in ("listen", "words", "voice", "sync")]
    return "\n".join(lines) + "\n"


def _problems(gates: list[dict]) -> list[str]:
    out: list[str] = []
    for gate in gates:
        if gate["status"] not in ("fail", "warn"):
            continue
        name = t("gate." + gate["id"])
        if gate["id"] == "words":
            for d in [d for d in gate["details"] if d["changed"]][:MAX_LISTED]:
                out.append(f"- {name}, {clock(d['start_s'])}: «{d['before']}» → «{d['after']}»")
            if gate["status"] == "fail" and not any(d["changed"] for d in gate["details"]):
                out.append(f"- {name}: " + t("report.problem.wer", wer=gate["values"]["wer"]))
        elif gate["id"] == "seams":
            for d in gate["details"]:
                if d["status"] != "pass":
                    out.append(f"- {name}, {clock(d['at_s'])}: " + t(
                        "report.problem.seam", click=d["click_db"], jump=_num(d["floor_jump_db"], "{:+.1f}")))
        elif gate["where_s"]:
            spots = ", ".join(clock(s) for s in gate["where_s"][:MAX_LISTED])
            out.append(f"- {name}: {spots}")
        else:
            out.append(f"- {name}: {measured(gate)}")
    return out
