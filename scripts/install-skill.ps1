# Install the voice-normal Agent Skill for Claude Code and Codex on Windows.
#
#   .\scripts\install-skill.ps1                  # both agents, junction (falls back to a copy)
#   .\scripts\install-skill.ps1 -Agent claude    # only %USERPROFILE%\.claude\skills
#   .\scripts\install-skill.ps1 -Mode copy       # copy instead of linking
#   .\scripts\install-skill.ps1 -Uninstall
#
# A junction needs no administrator rights and no Developer Mode, and keeps the skill
# in sync with this checkout after "git pull". Nothing outside the two skill folders is
# touched: no registry, no execution policy, no environment variables.
#
# Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files in the ANSI
# code page.

[CmdletBinding()]
param(
    [ValidateSet('all', 'claude', 'codex')]
    [string]$Agent = 'all',
    [ValidateSet('junction', 'copy')]
    [string]$Mode = 'junction',
    [switch]$Uninstall,
    # For tests: a folder to use instead of the user profile.
    [string]$UserDir = $env:USERPROFILE
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $repo 'skills\voice-normal'
$marker = 'repo-root.txt'

$targets = @()
if ($Agent -in 'all', 'claude') { $targets += Join-Path $UserDir '.claude\skills\voice-normal' }
if ($Agent -in 'all', 'codex') { $targets += Join-Path $UserDir '.agents\skills\voice-normal' }

function Test-Link([string]$path) {
    $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    return $item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Get-LinkTarget([string]$path) {
    $item = Get-Item -LiteralPath $path -Force
    $target = @($item.Target)[0]
    if (-not $target) { return $null }
    return [IO.Path]::GetFullPath($target.TrimStart('\', '?').Replace('UNC\', '\\'))
}

function Test-Ours([string]$path) {
    # Ours: a link to this checkout, or a copy that carries our marker file.
    if (Test-Link $path) {
        $target = Get-LinkTarget $path
        return $target -and ($target.TrimEnd('\') -ieq $source.TrimEnd('\'))
    }
    return Test-Path -LiteralPath (Join-Path $path $marker)
}

function Remove-Skill([string]$path) {
    if (Test-Link $path) {
        # DirectoryInfo.Delete() on a junction removes only the link, never the target's files.
        (Get-Item -LiteralPath $path -Force).Delete()
    } else {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

if (-not $Uninstall) {
    # The skill finds the CLI through this file (the plugin uses CLAUDE_PLUGIN_ROOT instead).
    [IO.File]::WriteAllText((Join-Path $source $marker), $repo + "`n")
}

$failed = $false
foreach ($target in $targets) {
    $exists = Test-Path -LiteralPath $target
    if ($exists -and -not (Test-Ours $target)) {
        Write-Host "[!!] $target already exists and is not a voice-normal install. Remove it yourself, then run this again."
        $failed = $true
        continue
    }
    if ($exists) { Remove-Skill $target }
    if ($Uninstall) {
        if ($exists) { Write-Host "[OK] removed $target" } else { Write-Host "[--] not installed: $target" }
        continue
    }

    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $how = $Mode
    if ($Mode -eq 'junction') {
        try {
            New-Item -ItemType Junction -Path $target -Target $source | Out-Null
        } catch {
            # Junctions fail on network drives and some synced folders; a copy always works.
            Write-Host "[??] junction failed ($($_.Exception.Message.Trim())), copying instead"
            $how = 'copy'
        }
    }
    if ($how -eq 'copy') {
        Copy-Item -LiteralPath $source -Destination $target -Recurse
        [IO.File]::WriteAllText((Join-Path $target $marker), $repo + "`n")
    }
    Write-Host "[OK] $target ($how -> $source)"
}

# The marker in the checkout is shared by both junctions: drop it only when both go.
if ($Uninstall -and $Agent -eq 'all' -and (Test-Path -LiteralPath (Join-Path $source $marker))) {
    Remove-Item -LiteralPath (Join-Path $source $marker) -Force
}
if ($failed) { exit 1 }
if (-not $Uninstall) {
    Write-Host ''
    Write-Host 'Restart Claude Code or Codex so it picks up the skill. Then ask, for example:'
    Write-Host '  "clean up the voice in D:\Video\camera.mp4"'
}
exit 0
