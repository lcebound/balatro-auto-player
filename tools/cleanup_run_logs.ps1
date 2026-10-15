[CmdletBinding()]
param([string]$Manifest = 'analysis/log_cleanup_20261009.json')

$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$runRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'runs')).Path
$manifestPath = Join-Path $workspacePath $Manifest
$record = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($record.status -ne 'prepared' -or $record.runs_root -ne $runRoot) { throw 'Unexpected cleanup manifest' }
$activeRunners = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -match 'runner\.py|bot\.py|balatro_ide|airtest.*run' })
if ($activeRunners.Count) { throw 'A game runner still uses these files' }
$checkedFiles = [Collections.Generic.List[string]]::new()
foreach ($target in $record.targets) {
    if ($target.batch -notmatch '^(?:(autonomous|batch|test)-[0-9-]+|test-[0-9]{8}-setup)$') { throw 'Unexpected batch name' }
    $expectedPath = Join-Path $runRoot $target.batch
    $resolvedPath = (Resolve-Path -LiteralPath $target.path).Path
    if ($resolvedPath -ne $expectedPath -or (Split-Path -Parent $resolvedPath) -ne $runRoot) {
        throw 'Cleanup target escaped the workspace runs directory'
    }
    if ((Get-Item -LiteralPath $resolvedPath).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Linked cleanup directory is not allowed'
    }
    $sourceLog = Join-Path $resolvedPath 'events.jsonl'
    if ($target.events_sha256) {
        if ((Get-FileHash -LiteralPath $sourceLog -Algorithm SHA256).Hash.ToLowerInvariant() -ne $target.events_sha256) {
            throw 'Log changed after its history was exported'
        }
    }
    if ($target.history_preserved) {
        $summaryPath = Join-Path $workspacePath ('results\' + $target.batch + '\summary.json')
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($summary.raw_source.events_sha256 -ne $target.events_sha256) { throw 'Exported history does not match the source log' }
    }
    $files = @(Get-ChildItem -LiteralPath $resolvedPath -File | Where-Object {
        $_.Name -match '^\d+-.*\.png$' -or $_.Name -in @('events.jsonl', 'stdout.log', 'stderr.log')
    })
    if ($files.Count -ne $target.files) { throw 'Generated files changed after the manifest was prepared' }
    foreach ($entry in $files) {
        if ($entry.DirectoryName -ne $resolvedPath -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Unsafe generated-file target'
        }
        $checkedFiles.Add($entry.FullName)
    }
}
$deleted = 0
foreach ($filePath in $checkedFiles) {
    Remove-Item -LiteralPath $filePath -Force
    $deleted++
}
$record.status = 'generated_logs_and_screenshots_deleted'
$record | Add-Member -NotePropertyName deleted_files -NotePropertyValue $deleted -Force
$record | Add-Member -NotePropertyName deleted_at -NotePropertyValue ([DateTimeOffset]::Now.ToString('o')) -Force
[IO.File]::WriteAllText($manifestPath, ($record | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
Write-Output "Deleted $deleted generated screenshots/logs. Exported results, fixtures and state files are retained."
