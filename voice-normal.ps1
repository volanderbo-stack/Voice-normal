# voice-normal for PowerShell. All logic lives in the Python package; this only starts it.
# PowerShell decodes native output with [Console]::OutputEncoding (often cp866), so
# Cyrillic from Python would turn into garbage. Switch both sides to UTF-8 for this
# call only and restore the previous state afterwards.

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    [Console]::Error.WriteLine('[voice-normal] uv not found in PATH. Install it: winget install astral-sh.uv')
    exit 1
}

$prevPyEncoding = $env:PYTHONIOENCODING
$prevConsoleEncoding = $null
try {
    $prevConsoleEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
} catch {
    # No real console (e.g. some CI hosts): nothing to switch.
}

$exitCode = 1
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & uv run --quiet --project $PSScriptRoot voice-normal @args
    $exitCode = $LASTEXITCODE
} finally {
    $env:PYTHONIOENCODING = $prevPyEncoding
    if ($null -ne $prevConsoleEncoding) {
        try { [Console]::OutputEncoding = $prevConsoleEncoding } catch { }
    }
}
exit $exitCode
