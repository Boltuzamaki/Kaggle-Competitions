$TaskName = "KaggleS6E7-TrainWatchdog"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
Write-Host "Removed scheduled task '$TaskName'. Any currently-running training process is untouched -- this only stops future auto-relaunches."
