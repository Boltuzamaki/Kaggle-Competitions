# Registers the training watchdog as a Windows Task Scheduler task that
# fires every 10 minutes, indefinitely, and relaunches `python -m src.train`
# whenever it finds it isn't currently running. This is the local
# (Task-Scheduler-based) equivalent of a cron job -- it runs on this
# machine, under this user's session, with access to the repo's local data
# and artifacts, which a cloud-side cron cannot do.
#
# Uses the ScheduledTasks module (Register-ScheduledTask) rather than the
# legacy schtasks.exe string-based CLI, since this repo's path contains
# spaces ("health risk prediction") and schtasks.exe's single-string /TR
# argument runs into nested-quoting problems with that; the cmdlet takes
# -Execute/-Argument as separate parameters and avoids the issue entirely.
#
# Usage (run once from a PowerShell prompt):
#   powershell -ExecutionPolicy Bypass -File scripts\install_watchdog.ps1
#
# To remove it later:
#   powershell -ExecutionPolicy Bypass -File scripts\uninstall_watchdog.ps1

$TaskName = "KaggleS6E7-TrainWatchdog"
$RepoDir = "C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\health risk prediction"
$WatchdogScript = Join-Path $RepoDir "scripts\watchdog.ps1"

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$argString = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$WatchdogScript`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argString -WorkingDirectory $RepoDir

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Checks every 10 min whether 'python -m src.train' (health-risk model zoo) is running; relaunches it if not. Training resumes from artifacts/manifest.csv, so this never redoes completed models." `
    -Force | Out-Null

Write-Host ""
Write-Host "Installed scheduled task '$TaskName' -- runs every 10 minutes."
Write-Host "It only takes action (relaunches training) when it finds no"
Write-Host "'python -m src.train' process alive; otherwise it's a no-op."
Write-Host ""
Write-Host "Watchdog activity log: $RepoDir\artifacts\logs\watchdog.log"
Write-Host "Check task status:     Get-ScheduledTaskInfo -TaskName `"$TaskName`""
Write-Host "Remove it:              powershell -File scripts\uninstall_watchdog.ps1"
