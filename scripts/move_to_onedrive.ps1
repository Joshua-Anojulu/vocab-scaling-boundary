# Move the project into OneDrive while keeping heavy, regenerable data OFF the sync.
#
# WHY JUNCTIONS: OneDrive does not traverse NTFS junction points, so a junctioned
# directory is skipped by the sync engine entirely. That is what keeps 14 GB of
# regenerable token arrays and a multi-GB venv out of the cloud while the 9,400 lines
# that matter do sync.
#
# WHAT STAYS PUT: data/ and .venv/ remain physically at the Projects path.
#   - data/  is 14 GB of derived arrays, already gitignored, fully regenerable.
#   - .venv/ contains absolute paths baked into pyvenv.cfg and the Scripts shims;
#     relocating it would break the interpreter.
#
# PRECONDITION: no process may hold a handle in the tree. Tokenization writes
# results/tokenize_log.txt continuously, and Windows refuses to move an open file.

$ErrorActionPreference = 'Stop'

$Src  = 'C:\Users\josha\Projects\vocab-scaling-boundary'
$Dst  = 'C:\Users\josha\OneDrive\Documents\vocab-scaling-boundary'
$Keep = @('data', '.venv')

Write-Host "=== preflight ===" -ForegroundColor Cyan

if (-not (Test-Path $Src)) { throw "source missing: $Src" }
if (Test-Path $Dst)        { throw "destination already exists: $Dst" }

# Refuse to run while anything holds the tree.
#
# A command-line filter is NOT sufficient and was actively misleading during development:
# `.venv\Scripts\python.exe` is a shim that execs the real uv-managed interpreter, so the
# process doing the work runs as a CHILD whose command line contains no project path.
# Filtering on '*vocab-scaling*' therefore matched only the idle shim (1 MB, 0 s CPU)
# while a 1.5 GB child burned 60 s of CPU per 40 s of wall clock.
#
# The decisive test is a file-system one: if any handle is open anywhere in the tree,
# Windows refuses the rename below. That cannot be fooled by process-naming.
$probe = Join-Path $Src '.movelock'
try {
    New-Item -ItemType File -Path $probe -Force | Out-Null
    Rename-Item -LiteralPath $probe -NewName '.movelock.ok' -ErrorAction Stop
    Remove-Item -LiteralPath (Join-Path $Src '.movelock.ok') -Force
} catch {
    throw "cannot write/rename inside the tree - something holds it: $_"
}

# Informational only, now that the filter's limits are understood.
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*vocab-scaling*' -or $_.ExecutablePath -like '*vocab-scaling*' }
if ($busy) {
    foreach ($p in $busy) {
        $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.ProcessId)"
        throw "python still active in the tree (pid $($p.ProcessId), $($kids.Count) child(ren)) - wait for exit"
    }
}

Push-Location $Src
$dirty = git status --porcelain | Where-Object { $_ -notmatch '^\?\? \.omc' }
if ($dirty) { Write-Warning "uncommitted changes present:`n$($dirty -join "`n")" }
$ahead = git status -sb | Select-String '\[ahead'
if ($ahead) { Write-Warning "local commits not pushed: $ahead" }
Pop-Location

Write-Host "=== moving code and history ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $Dst -Force | Out-Null

Get-ChildItem -Path $Src -Force | Where-Object { $Keep -notcontains $_.Name } | ForEach-Object {
    Write-Host "  -> $($_.Name)"
    Move-Item -LiteralPath $_.FullName -Destination $Dst -Force
}

Write-Host "=== junctioning heavy directories back ===" -ForegroundColor Cyan
foreach ($name in $Keep) {
    $target = Join-Path $Src $name
    $link   = Join-Path $Dst $name
    if (-not (Test-Path $target)) { Write-Warning "  $name absent at source; skipping"; continue }
    cmd /c mklink /J "`"$link`"" "`"$target`"" | Out-Null
    if (-not (Test-Path $link)) { throw "junction failed: $link" }
    $item = Get-Item $link -Force
    if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$link is not a reparse point - OneDrive would sync it"
    }
    Write-Host "  $name -> $target  (reparse point OK)"
}

Write-Host "=== verification ===" -ForegroundColor Cyan
Push-Location $Dst
git rev-parse --short HEAD
git status --porcelain | Where-Object { $_ -notmatch '^\?\? \.omc' } | ForEach-Object { Write-Host "  dirty: $_" }
$n = (Get-ChildItem 'data\tokens' -Filter *.npy -ErrorAction SilentlyContinue).Count
Write-Host "  token arrays visible through junction: $n"
$t = (Get-ChildItem 'tokenizers' -Filter 'bpe_v*.json' -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -notmatch 'report' }).Count
Write-Host "  tokenizers visible: $t"
Pop-Location

Write-Host ""
Write-Host "MOVED. New root: $Dst" -ForegroundColor Green
Write-Host "Heavy data remains at: $Src (junctioned, not synced)"
Write-Host "Run the suite from the new root to confirm the venv still resolves."
