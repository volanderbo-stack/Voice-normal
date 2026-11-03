---
name: voice-normal
description: Turn a speech recording or a video's soundtrack (camera, laptop, screen recording, lavalier, phone) into clean, podcast-quality voice locally, with checks that words, voice and sync are unchanged. Use when the user asks to clean up, denoise, de-reverb, restore or level the voice in an audio or video file, or to make it sound like a podcast microphone — e.g. "почисти звук", "прибери шум", "зроби нормальний голос", "убери эхо", "make my voice sound clean", "fix the audio in this video".
---

# voice-normal

A local CLI that analyses a recording, restores the voice with Sidon, matches the
speaker's timbre, masters to −16 LUFS / −1 dBTP, then checks the result against eight
gates (quality, words, voice, sync, loudness, seams, echo, length) with up to three
repair rounds. Nothing leaves the computer. You run it and report what it says; you do
not edit audio yourself.

## 1. Find the CLI

The project root is the first of these that exists:

1. `${CLAUDE_PLUGIN_ROOT}` — when this skill came from the Claude Code plugin, this is
   already a real path. If it still reads literally as a dollar-brace placeholder, skip it.
2. The path written in `repo-root.txt` next to this `SKILL.md` (created by
   `scripts/install-skill.ps1` or `scripts/install-skill.sh`).

Then the command is:

- Windows (PowerShell or cmd): `& "<root>\voice-normal.cmd" <command> ...`
  Use the `.cmd` wrapper, not `.ps1`: it works whatever the PowerShell execution policy
  is. Never change the execution policy or any other system setting.
- macOS / Linux: `"<root>/voice-normal" <command> ...`

Quote every path: user files often have spaces and Cyrillic letters in their names.
Add `--lang uk`, `--lang ru` or `--lang en` to match the language the user writes in,
and `--json` whenever you need to read the result.

## 2. Check the setup

```powershell
& "<root>\voice-normal.cmd" doctor --json
```

- `ok: false` with a failing `ffmpeg`, `ffprobe` or `uv` check: show the user the
  `hints` from the JSON (winget commands) and stop. Do not install system tools yourself.
- The `ml_packages` or `models` check is `pending`: `install` is needed. It downloads
  about 1 GB of Python packages and 1.5 GB of models once. **Ask the user before running
  it**, then run `install` (add `--device cpu` if they have no NVIDIA GPU). It resumes
  after an interrupted download, so simply run it again if it fails on the network.

## 3. Plan, then confirm

```powershell
& "<root>\voice-normal.cmd" plan "<file>" --out "<dir>" --json
```

`plan` changes nothing. Tell the user, in a few lines: the route (`restore` = Sidon
restoration, `master_only` = the recording is already clean, only levelling), the main
reasons, any warnings (for example little speech, music), and the time estimate. If the
estimate is long (more than ~10 minutes) or there are warnings, ask before processing.
Default output folder: `cleaned` next to the input file, unless the user names one.

## 4. Process

```powershell
& "<root>\voice-normal.cmd" process "<file>" --out "<dir>" --json
```

It takes roughly 1¾ × the recording's length on a laptop CPU (less with CUDA). Run it in
the background when your environment allows and let the user know it is running.
Progress lines go to stderr; the final JSON goes to stdout.

Exit codes: `0` all gates passed (possibly with warnings), `1` a gate failed even after
repair, or the file could not be read, `2` bad arguments or a missing file, `3` `install`
has not been run, `130` interrupted.

## 5. Report honestly

Read `<dir>\<name>.report.json` (`verdict`, `gates`, `rounds`, `notes`) or the human
report `<dir>\<name>.report.md`, and tell the user:

- the verdict: `pass`, `warn` or `fail`, in plain words;
- the paths of the cleaned `.wav` and, for a video, the new video file;
- every gate with status `fail` or `warn`, with its time positions from the report;
- if `notes` contains `restore_rejected` or `probe_not_better`: restoration was dropped,
  the result is the original, only levelled — the noise is still there;
- that the gates do not replace listening: suggest playing the beginning, the middle and
  the end.

Never describe a failed or skipped gate as passed, and never re-run with other settings
just to make a gate pass. If the user wants to try again, explain what the report says
first.

## Limits worth knowing

- Sidon resynthesises the voice. It is very good on noisy speech but can change it; this
  is exactly what the words, voice and quality gates check.
- Music, singing or several people talking over each other are not what it is made for;
  `plan` warns when speech is a small share of the file.
- Output audio is mono, 48 kHz, 24-bit WAV of exactly the input length; video frames are
  copied without re-encoding.
