"""Точка входу CLI: doctor | install | plan | process.

Уся логіка живе тут, у Python; обгортки .cmd/.ps1/sh лише запускають `uv run`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voice_normal import __version__, console, doctor, i18n
from voice_normal.i18n import t

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_NOT_READY = 3
EXIT_INTERRUPTED = 130


def _peek_lang(argv: list[str]) -> str:
    # Мову треба знати ще до побудови парсера, бо від неї залежить текст --help.
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return i18n.default_lang()


def _common_options(with_defaults: bool) -> argparse.ArgumentParser:
    # Опції приймаються і до, і після назви команди. Підпарсер не повинен мати
    # власних значень за замовчуванням, інакше він затре `--lang ru doctor`.
    common = argparse.ArgumentParser(add_help=False)
    lang_default = i18n.default_lang() if with_defaults else argparse.SUPPRESS
    json_default = False if with_defaults else argparse.SUPPRESS
    common.add_argument("--lang", choices=i18n.LANGS, default=lang_default, help=t("cli.lang"))
    common.add_argument("--json", action="store_true", default=json_default, help=t("cli.json"))
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options(with_defaults=False)
    parser = argparse.ArgumentParser(
        prog="voice-normal", description=t("cli.description"), parents=[_common_options(with_defaults=True)]
    )
    parser.add_argument("--version", action="version", version=f"voice-normal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", parents=[common], help=t("cli.doctor"))
    install = sub.add_parser("install", parents=[common], help=t("cli.install"))
    install.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help=t("cli.device"))
    install.add_argument("--verify", action="store_true", help=t("cli.verify"))
    install.add_argument("--skip-packages", action="store_true", help=t("cli.skip_packages"))

    plan = sub.add_parser("plan", parents=[common], help=t("cli.plan"))
    plan.add_argument("input", type=Path, help=t("cli.input"))

    process = sub.add_parser("process", parents=[common], help=t("cli.process"))
    process.add_argument("input", type=Path, help=t("cli.input"))
    process.add_argument("--out", type=Path, default=Path("cleaned"), help=t("cli.out"))
    return parser


def _emit(payload: dict[str, object] | None, text: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = doctor.run_checks()
    summary, code = doctor.summarize(checks)
    _emit(doctor.as_json(checks, summary, code), doctor.format_human(checks, summary), args.json)
    return code


def cmd_install(args: argparse.Namespace) -> int:
    from voice_normal import download, install

    # Хід встановлення — у stderr, якщо stdout зайнятий JSON-результатом.
    stream = sys.stderr if args.json else sys.stdout

    def log(message: str) -> None:
        print(message, file=stream, flush=True)

    log(t("install.title"))
    try:
        result = install.run_install(args.device, args.verify, args.skip_packages, log)
    except download.ChecksumError as exc:
        result = install.InstallResult(ok=False, error=t("install.checksum_failed", error=exc))
    except download.DownloadError as exc:
        result = install.InstallResult(ok=False, error=t("install.download_failed", error=exc))
    except RuntimeError as exc:
        result = install.InstallResult(ok=False, error=str(exc))
    except KeyboardInterrupt:
        print(t("install.interrupted"), file=sys.stderr)
        return EXIT_INTERRUPTED
    text = t("install.done") if result.ok else (result.error or t("install.smoke.failed"))
    _emit(install.as_json(result), text, args.json)
    return EXIT_OK if result.ok else EXIT_FAIL


def _not_ready(args: argparse.Namespace, stage: int) -> int:
    if getattr(args, "input", None) is not None and not args.input.exists():
        message = t("cli.input_missing", path=args.input)
        _emit({"ok": False, "error": message}, message, args.json)
        return EXIT_USAGE
    message = t("cli.not_ready", command=args.command, stage=stage)
    _emit({"ok": False, "error": message}, message, args.json)
    return EXIT_NOT_READY


COMMANDS = {
    "doctor": cmd_doctor,
    "install": cmd_install,
    "plan": lambda args: _not_ready(args, 3),
    "process": lambda args: _not_ready(args, 4),
}


def main(argv: list[str] | None = None) -> int:
    console.setup_streams()
    argv = list(sys.argv[1:] if argv is None else argv)
    i18n.set_lang(_peek_lang(argv))
    args = build_parser().parse_args(argv)
    i18n.set_lang(args.lang)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        # Ctrl+C у Windows приходить як KeyboardInterrupt (SIGINT); SIGTERM там немає.
        print(t("cli.interrupted"), file=sys.stderr)
        return EXIT_INTERRUPTED
