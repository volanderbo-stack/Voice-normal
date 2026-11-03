# voice-normal

Локальний інструмент для Windows 10/11, який робить із запису мови (камера, ноутбук,
запис екрана, петличка) чистий голос, як з подкастного мікрофона.
Працює як CLI і як Agent Skill для Claude Code / Codex. Усе обробляється на вашому
комп'ютері, нічого не відправляється в інтернет (крім одноразового завантаження моделей).

> Проєкт у розробці. Стан етапів — нижче.

| Етап | Що | Стан |
|---|---|---|
| 1 | Каркас, `doctor`, обгортки, CI | готово |
| 2 | `install`: пакети й моделі | готово |
| 3 | `plan`: аналіз і маршрут | готово |
| 4 | `process`: відновлення і зведення | — |
| 5 | Перевірка за гейтами, `report.md` | — |
| 6 | Skill і плагін Claude Code / Codex | — |
| 7 | Тести на зламаний вихід, документація | — |

## Встановлення на Windows

Потрібно: Windows 10 або 11 (64-bit) і щонайменше 6 ГБ вільного місця
(≈1 ГБ пакетів, ≈1.5 ГБ моделей, решта — тимчасові файли під час обробки).
GPU NVIDIA не обов'язковий: без нього все працює на CPU, лише повільніше.

1. Встановіть інструменти (у PowerShell або cmd):

   ```powershell
   winget install Git.Git
   winget install Gyan.FFmpeg
   winget install astral-sh.uv
   ```

   Після встановлення відкрийте **новий** термінал, щоб оновився PATH.

2. Завантажте проєкт:

   ```powershell
   git clone https://github.com/volanderbo-stack/Voice-normal.git
   cd Voice-normal
   ```

3. Перевірте, що все готово:

   ```powershell
   .\voice-normal doctor
   ```

   Перший запуск сам поставить потрібний Python (3.12) — це займе хвилину.
   Пункти з `[!!]` треба виправити, `[..]` — це те, що зробить `voice-normal install`.

4. Встановіть ML-пакети й моделі (≈1 ГБ пакетів і ≈1.5 ГБ моделей, один раз):

   ```powershell
   .\voice-normal install
   ```

   Пристрій обирається сам: є NVIDIA — CUDA, немає — CPU. Можна задати явно:
   `--device cpu` або `--device cuda`. Якщо інтернет обірвався, просто запустіть
   `install` ще раз — завантаження продовжиться з місця зупинки. Кожен файл
   перевіряється за sha256; `install --verify` перераховує їх усі ще раз.
   Наприкінці `install` запускає кожну модель на тестовому сигналі й показує,
   у скільки разів швидше (або повільніше) за реальний час працює відновлення.

5. Подивіться, що в записі і як його оброблятимуть:

   ```powershell
   .\voice-normal plan "D:\Відео\запис з камери.mp4" --out cleaned\
   ```

   `plan` нічого не змінює у файлі. Він рахує гучність (EBU R128), де є мова,
   рівень шуму й SNR, кліпінг, смугу частот, реверберацію та оцінку DNSMOS.
   Далі обирає маршрут: «відновлення голосу» або «лише зведення», якщо запис уже
   чистий. Знаходить найважчі 150 с для проби і оцінює час обробки на вашому
   комп'ютері. Результат — `cleaned\<ім'я>.plan.md` (для людини) і
   `cleaned\<ім'я>.plan.json` (для `process` і агентів).

Мова повідомлень: `--lang uk` (за замовчуванням), `--lang ru` або `--lang en`,
або змінна середовища `VOICE_NORMAL_LANG`. Для агентів і скриптів є `--json`.

Якщо PowerShell не дає запускати `.ps1` (політика виконання), користуйтеся
`.\voice-normal.cmd doctor` — він робить те саме.

Дані й моделі зберігаються в `%LOCALAPPDATA%\voice-normal`. Щоб тримати їх деінде,
задайте змінну `VOICE_NORMAL_HOME`.

## Ліцензія

Код — MIT, див. [LICENSE](LICENSE). Моделі завантажуються з їхніх джерел і мають
власні ліцензії:

| Модель | Для чого | Ліцензія | Джерело |
|---|---|---|---|
| Sidon v0.1 (Nakata, Saito та ін.) | відновлення голосу | MIT | [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) |
| w2v-BERT 2.0 (Meta) — конфіг препроцесора | вхід Sidon | MIT | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) |
| DNSMOS P.835 (Microsoft) | оцінка якості SIG/BAK/OVRL | ваги — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), код — MIT | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) |
| Whisper small (OpenAI), конвертація faster-whisper (SYSTRAN) | перевірка слів | MIT | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) |
| ECAPA-TDNN 512 (WeSpeaker) | перевірка голосу | Apache-2.0 | [Wespeaker/wespeaker-voxceleb-ecapa-tdnn512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512) |

DNSMOS: C. K. A. Reddy, V. Gopal, R. Cutler, "DNSMOS P.835: A non-intrusive perceptual
objective speech quality metric to evaluate noise suppressors", ICASSP 2022.
Модель використовується без змін.
