# Run manually after reviewing results/. This script has not been executed by Codex.
[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$runRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'runs')).Path
$archiveRoot = Join-Path $workspacePath 'results\autonomous-20261002-000231'
$manifestPath = Join-Path $archiveRoot 'cleanup.json'
$cleanupRecord = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not (Test-Path -LiteralPath (Join-Path $archiveRoot 'summary.json'))) { throw 'Exported summary is missing' }
$activeRunners = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" | Where-Object { $_.CommandLine -match 'runner\.py|bot\.py|balatro_ide|airtest.*run' })
if ($activeRunners.Count -gt 0) { throw 'An active game runner still uses the logs' }
$sourceLog = Join-Path $runRoot 'autonomous-20261002-000231\events.jsonl'
if (Test-Path -LiteralPath $sourceLog) {
    $sourceHash = (Get-FileHash -LiteralPath $sourceLog -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $cleanupRecord.source_log_sha256) { throw 'Raw log changed after export; export it again before cleanup' }
}
$checkedFiles = @()
foreach ($batchName in @('autonomous-20261002-000231', 'autonomous-20261001-235758')) {
    $expectedPath = Join-Path $runRoot $batchName
    if (-not (Test-Path -LiteralPath $expectedPath)) { continue }
    $resolvedPath = (Resolve-Path -LiteralPath $expectedPath).Path
    if ($resolvedPath -ne $expectedPath -or -not $resolvedPath.StartsWith($runRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe cleanup path' }
    $directory = Get-Item -LiteralPath $resolvedPath
    if (-not $directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected linked directory' }
    foreach ($entry in Get-ChildItem -LiteralPath $resolvedPath -File) {
        if ($entry.Name -match '^\d+-.*\.png$' -or $entry.Name -in @('events.jsonl', 'stdout.log', 'stderr.log')) {
            if ($entry.DirectoryName -ne $resolvedPath -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsafe file target' }
            $checkedFiles += $entry.FullName
        }
    }
}
$deletedCount = 0
foreach ($filePath in $checkedFiles) {
    if ($PSCmdlet.ShouldProcess($filePath, 'Delete exported raw screenshot/log')) {
        Remove-Item -LiteralPath $filePath -Force
        $deletedCount++
    }
}
if (-not $WhatIfPreference -and $deletedCount -eq $checkedFiles.Count) {
    $cleanupRecord.status = 'raw_logs_deleted_state_retained'
    $cleanupRecord.deleted_at = [DateTimeOffset]::Now.ToString('o')
    $cleanupRecord | Add-Member -NotePropertyName deleted_files -NotePropertyValue $deletedCount -Force
    [IO.File]::WriteAllText($manifestPath, ($cleanupRecord | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
}
Write-Output "Deleted $deletedCount generated files. Results and state files are retained."
