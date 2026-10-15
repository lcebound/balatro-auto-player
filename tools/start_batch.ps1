[CmdletBinding()]
param([int]$Games = 10)
$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$activeRunners = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -match 'runner\.py|balatro_ide|airtest.*run' })
if ($activeRunners.Count) { throw 'An active Balatro runner already exists' }
$stamp = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8))
$batchName = 'autonomous-' + $stamp.ToString('yyyyMMdd-HHmmss')
$runPath = Join-Path $workspacePath ('runs/' + $batchName)
New-Item -ItemType Directory -Path $runPath | Out-Null
$runnerProcess = Start-Process -FilePath (Join-Path $workspacePath '.venv/Scripts/python.exe') `
    -ArgumentList @('-u', 'runner.py', 'batch', '--games', $Games.ToString(), '--run-dir', ('runs/' + $batchName)) `
    -WorkingDirectory $workspacePath -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runPath 'stdout.log') `
    -RedirectStandardError (Join-Path $runPath 'stderr.log') -PassThru
$pointer = @{run_dir=$runPath;started_at=$stamp.ToString('o');target_games=$Games;pid=$runnerProcess.Id}
$awakeProcess = Start-Process -FilePath (Join-Path $workspacePath '.venv/Scripts/pythonw.exe') `
    -ArgumentList @('tools/keep_batch_awake.py', '--pid', $runnerProcess.Id.ToString(), '--run-dir', ('runs/' + $batchName)) `
    -WorkingDirectory $workspacePath -WindowStyle Hidden -PassThru
$pointer.awake_pid = $awakeProcess.Id
[IO.File]::WriteAllText((Join-Path $workspacePath 'runs/active-batch.json'), ($pointer | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$pointer | ConvertTo-Json
