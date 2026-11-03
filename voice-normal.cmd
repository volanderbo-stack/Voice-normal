@echo off
rem voice-normal for cmd.exe. All logic lives in the Python package; this only starts it.
rem Keep this file ASCII-only: cmd.exe reads .cmd files in the OEM code page.
setlocal
set "PYTHONIOENCODING=utf-8"
where uv >nul 2>nul
if errorlevel 1 (
    echo [voice-normal] uv not found in PATH. Install it: winget install astral-sh.uv 1>&2
    exit /b 1
)
rem "%~dp0." avoids the trailing backslash that would escape the closing quote.
uv run --quiet --no-dev --project "%~dp0." voice-normal %*
exit /b %ERRORLEVEL%
