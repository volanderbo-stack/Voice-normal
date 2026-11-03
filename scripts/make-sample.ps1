# Make a demo "camera" recording to try voice-normal on: built-in Windows speech
# synthesis (System.Speech) plus room noise and a little echo, optionally inside a video.
#
#   .\scripts\make-sample.ps1                       # .\samples\camera-sample.wav
#   .\scripts\make-sample.ps1 -Out D:\test -Video   # also camera-sample.mp4 with a test picture
#
# Everything is generated locally; nothing is downloaded. Needs ffmpeg in PATH.
# Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files in the ANSI code page.

[CmdletBinding()]
param(
    [string]$Out = (Join-Path (Get-Location) 'samples'),
    [switch]$Video,
    [string]$Text = ('Hello. This is a demo recording for voice normal. It was made on a laptop, ' +
        'in a room with bare walls, and the fan is running in the background. ' +
        'The words are easy to follow, but the sound is thin, distant and a little noisy. ' +
        'After processing, the voice should sound close and clean, like a podcast microphone, ' +
        'while every word and the timing stay exactly the same.')
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host '[!!] ffmpeg not found. Install it: winget install Gyan.FFmpeg'
    exit 1
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$clean = Join-Path $Out 'camera-sample.clean.wav'
$noisy = Join-Path $Out 'camera-sample.wav'

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1
if (-not $voice) {
    Write-Host '[!!] No English System.Speech voice installed (Settings > Time & language > Speech).'
    exit 1
}
$synth.SelectVoice($voice.VoiceInfo.Name)
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 48000, `
    ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($clean, $format)
$synth.Speak($Text)
$synth.Dispose()

# Room: early reflections and a muffled top (aecho + lowpass), fan noise (brown + pink), quieter level.
$filter = '[0:a]aecho=0.8:0.7:19|37|61|97:0.32|0.24|0.17|0.11,lowpass=f=6500,volume=0.5[v];' +
    '[1:a]volume=0.6[n];[v][n]amix=inputs=2:normalize=0:duration=first[out]'
& ffmpeg -nostdin -v error -y -i $clean `
    -f lavfi -i 'anoisesrc=color=brown:amplitude=0.02:sample_rate=48000:seed=3' `
    -filter_complex $filter -map '[out]' -ar 48000 -ac 1 $noisy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item -LiteralPath $clean
Write-Host "[OK] $noisy"

if ($Video) {
    $mp4 = Join-Path $Out 'camera-sample.mp4'
    & ffmpeg -nostdin -v error -y -f lavfi -i 'testsrc2=size=640x360:rate=25' -i $noisy `
        -shortest -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k $mp4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[OK] $mp4"
}
exit 0
