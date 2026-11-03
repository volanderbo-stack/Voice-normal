"""Повідомлення інтерфейсу українською (за замовчуванням), російською та англійською."""

from __future__ import annotations

import os

LANGS = ("uk", "ru", "en")
DEFAULT_LANG = "uk"

MESSAGES: dict[str, dict[str, str]] = {
    "uk": {
        "doctor.title": "voice-normal doctor {version}",
        "doctor.python": "Python {version}",
        "doctor.python.bad": "Python {version} не підходить, потрібен {required}",
        "doctor.os": "ОС: {name}",
        "doctor.os.old": "ОС: {name} — потрібна Windows 10 або 11 (64-bit)",
        "doctor.tool.found": "{tool} {version} ({path})",
        "doctor.tool.missing": "{tool} не знайдено в PATH",
        "doctor.tool.hint_new_terminal": "після встановлення відкрийте новий термінал, щоб оновився PATH",
        "doctor.data_dir": "Папка даних: {path}",
        "doctor.data_dir.bad": "Папка даних недоступна для запису: {path} ({error})",
        "doctor.disk": "Вільно на диску: {free:.1f} ГБ",
        "doctor.disk.low": "Вільно на диску лише {free:.1f} ГБ, бажано щонайменше {want} ГБ",
        "doctor.disk.too_low": "Замало місця: {free:.1f} ГБ, потрібно щонайменше {need} ГБ",
        "doctor.longpaths.on": "Довгі шляхи (>260 символів): увімкнено в системі",
        "doctor.longpaths.off": "Довгі шляхи вимкнено в системі — voice-normal обходить це сам, але інші програми можуть не відкрити такі файли",
        "doctor.gpu.nvidia": "GPU: {name} — буде CUDA",
        "doctor.gpu.none": "GPU NVIDIA не знайдено — обробка йтиме на CPU (повільніше, але працює)",
        "doctor.ml.missing": "ML-пакети ще не встановлено",
        "doctor.ml.ok": "ML-пакети встановлено ({packages})",
        "doctor.models.missing": "Моделі ще не завантажено",
        "doctor.next_install": "voice-normal install",
        "doctor.summary.ok": "Усе готово до роботи.",
        "doctor.summary.pending": "Інструменти на місці. Далі: voice-normal install",
        "doctor.summary.fail": "Є проблеми ({count}) — виправте пункти з [!!] і запустіть doctor ще раз.",
        "cli.description": "Робить із запису мови чистий голос, як з подкастного мікрофона. Усе локально.",
        "cli.lang": "мова повідомлень і звітів: uk, ru або en",
        "cli.json": "вивести результат у JSON (для агентів і скриптів)",
        "cli.doctor": "перевірити, чи все готово до роботи",
        "cli.install": "встановити ML-пакети й завантажити моделі",
        "cli.plan": "проаналізувати запис і показати план обробки",
        "cli.process": "обробити запис",
        "cli.input": "вхідний аудіо- або відеофайл",
        "cli.out": "папка для результату",
        "cli.not_ready": "Команда «{command}» з'явиться на етапі {stage}. Зараз доступна лише doctor.",
        "cli.input_missing": "Файл не знайдено: {path}",
        "cli.interrupted": "Перервано користувачем.",
    },
    "ru": {
        "doctor.title": "voice-normal doctor {version}",
        "doctor.python": "Python {version}",
        "doctor.python.bad": "Python {version} не подходит, нужен {required}",
        "doctor.os": "ОС: {name}",
        "doctor.os.old": "ОС: {name} — нужна Windows 10 или 11 (64-bit)",
        "doctor.tool.found": "{tool} {version} ({path})",
        "doctor.tool.missing": "{tool} не найден в PATH",
        "doctor.tool.hint_new_terminal": "после установки откройте новый терминал, чтобы обновился PATH",
        "doctor.data_dir": "Папка данных: {path}",
        "doctor.data_dir.bad": "Папка данных недоступна для записи: {path} ({error})",
        "doctor.disk": "Свободно на диске: {free:.1f} ГБ",
        "doctor.disk.low": "Свободно на диске лишь {free:.1f} ГБ, желательно не меньше {want} ГБ",
        "doctor.disk.too_low": "Мало места: {free:.1f} ГБ, нужно не меньше {need} ГБ",
        "doctor.longpaths.on": "Длинные пути (>260 символов): включены в системе",
        "doctor.longpaths.off": "Длинные пути выключены в системе — voice-normal обходит это сам, но другие программы могут не открыть такие файлы",
        "doctor.gpu.nvidia": "GPU: {name} — будет CUDA",
        "doctor.gpu.none": "GPU NVIDIA не найден — обработка пойдёт на CPU (медленнее, но работает)",
        "doctor.ml.missing": "ML-пакеты ещё не установлены",
        "doctor.ml.ok": "ML-пакеты установлены ({packages})",
        "doctor.models.missing": "Модели ещё не загружены",
        "doctor.next_install": "voice-normal install",
        "doctor.summary.ok": "Всё готово к работе.",
        "doctor.summary.pending": "Инструменты на месте. Дальше: voice-normal install",
        "doctor.summary.fail": "Есть проблемы ({count}) — исправьте пункты с [!!] и запустите doctor ещё раз.",
        "cli.description": "Делает из записи речи чистый голос, как с подкастного микрофона. Всё локально.",
        "cli.lang": "язык сообщений и отчётов: uk, ru или en",
        "cli.json": "вывести результат в JSON (для агентов и скриптов)",
        "cli.doctor": "проверить, всё ли готово к работе",
        "cli.install": "установить ML-пакеты и загрузить модели",
        "cli.plan": "проанализировать запись и показать план обработки",
        "cli.process": "обработать запись",
        "cli.input": "входной аудио- или видеофайл",
        "cli.out": "папка для результата",
        "cli.not_ready": "Команда «{command}» появится на этапе {stage}. Сейчас доступна только doctor.",
        "cli.input_missing": "Файл не найден: {path}",
        "cli.interrupted": "Прервано пользователем.",
    },
    "en": {
        "doctor.title": "voice-normal doctor {version}",
        "doctor.python": "Python {version}",
        "doctor.python.bad": "Python {version} is not supported, need {required}",
        "doctor.os": "OS: {name}",
        "doctor.os.old": "OS: {name} — Windows 10 or 11 (64-bit) is required",
        "doctor.tool.found": "{tool} {version} ({path})",
        "doctor.tool.missing": "{tool} not found in PATH",
        "doctor.tool.hint_new_terminal": "open a new terminal after installing so PATH is refreshed",
        "doctor.data_dir": "Data folder: {path}",
        "doctor.data_dir.bad": "Data folder is not writable: {path} ({error})",
        "doctor.disk": "Free disk space: {free:.1f} GB",
        "doctor.disk.low": "Only {free:.1f} GB free, at least {want} GB recommended",
        "doctor.disk.too_low": "Not enough space: {free:.1f} GB, need at least {need} GB",
        "doctor.longpaths.on": "Long paths (>260 chars): enabled in the system",
        "doctor.longpaths.off": "Long paths are disabled in the system — voice-normal works around it, but other programs may fail to open such files",
        "doctor.gpu.nvidia": "GPU: {name} — CUDA will be used",
        "doctor.gpu.none": "No NVIDIA GPU — processing will run on CPU (slower, but works)",
        "doctor.ml.missing": "ML packages are not installed yet",
        "doctor.ml.ok": "ML packages installed ({packages})",
        "doctor.models.missing": "Models are not downloaded yet",
        "doctor.next_install": "voice-normal install",
        "doctor.summary.ok": "Everything is ready.",
        "doctor.summary.pending": "Tools are in place. Next: voice-normal install",
        "doctor.summary.fail": "{count} problem(s) — fix the [!!] items and run doctor again.",
        "cli.description": "Turns a speech recording into clean, podcast-mic quality voice. Fully local.",
        "cli.lang": "language of messages and reports: uk, ru or en",
        "cli.json": "print the result as JSON (for agents and scripts)",
        "cli.doctor": "check that everything is ready",
        "cli.install": "install ML packages and download models",
        "cli.plan": "analyse a recording and show the processing plan",
        "cli.process": "process a recording",
        "cli.input": "input audio or video file",
        "cli.out": "output folder",
        "cli.not_ready": "The '{command}' command arrives in stage {stage}. Only doctor is available now.",
        "cli.input_missing": "File not found: {path}",
        "cli.interrupted": "Interrupted by user.",
    },
}

_current = DEFAULT_LANG


def default_lang() -> str:
    lang = os.environ.get("VOICE_NORMAL_LANG", DEFAULT_LANG).lower()
    return lang if lang in LANGS else DEFAULT_LANG


def set_lang(lang: str) -> None:
    global _current
    _current = lang if lang in LANGS else DEFAULT_LANG


def t(key: str, **kwargs: object) -> str:
    text = MESSAGES[_current].get(key) or MESSAGES["en"][key]
    return text.format(**kwargs) if kwargs else text
