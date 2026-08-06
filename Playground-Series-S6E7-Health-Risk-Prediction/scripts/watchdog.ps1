# Watchdog for the model-zoo training run. Runs on a schedule (see
# scripts/install_watchdog.ps1, registered as Windows Task Scheduler task
# "KaggleS6E7-TrainWatchdog"). Checks whether a `python -m src.train`
# process is currently alive; if not (crash, reboot, OOM-kill, laptop sleep
# that killed the process, manual stop -- anything), relaunches it. Training
# itself resumes from artifacts/manifest.csv, so a relaunch never redoes an
# already-completed model -- this script only needs to answer "is it running
# right now", nothing more.

$RepoDir = "C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\health risk prediction"
$LogDir = Join-Path $RepoDir "artifacts\logs"
$WatchdogLog = Join-Path $LogDir "watchdog.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $WatchdogLog -Value "[$ts] $msg"
}

try {
    # process image name varies (python.exe via the Windows Store app-execution
    # alias, python3.11.exe when resolved directly, pythonw.exe, etc.) so match
    # by command line, not by exact image name.
    $running = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "-m\s+src\.train" }

    if ($running) {
        $pidList = ($running | ForEach-Object { $_.ProcessId }) -join ","
        Log "OK: training running (pid(s) $pidList)"
    } else {
        Log "training NOT running -- relaunching (resumes from manifest.csv)"
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $trainLog = Join-Path $LogDir "train_run_$stamp.log"

        # cmd.exe wrapper gives us simple ">>" append redirection without
        # having to manually pump PowerShell process output streams.
        $cmdArgs = "/c cd /d `"$RepoDir`" && python -u -m src.train >> `"$trainLog`" 2>&1"
        Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgs -WindowStyle Hidden

        Log "relaunched via cmd.exe wrapper, log: $trainLog"
    }
} catch {
    Log "ERROR in watchdog: $_"
}
