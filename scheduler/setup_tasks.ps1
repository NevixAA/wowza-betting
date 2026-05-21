# Wowza Task Scheduler Setup
# Run as Administrator: Right-click PowerShell -> "Run as administrator"
# Then: powershell -ExecutionPolicy Bypass -File "C:\Users\Nevo\OneDrive - Apollo\archive\נבו אישי\אישי\Wowza\mixed\v9\scheduler\setup_tasks.ps1"

$Python = "C:\Users\Nevo\OneDrive - Apollo\archive\נבו אישי\אישי\Wowza\mixed\.venv\Scripts\python.exe"
$V9Dir  = "C:\Users\Nevo\OneDrive - Apollo\archive\נבו אישי\אישי\Wowza\mixed\v9"

# Task 1: Predict every 4 hours (06:00, 10:00, 14:00, 18:00, 22:00)
$a1 = New-ScheduledTaskAction -Execute $Python -Argument "pipeline.py --mode predict" -WorkingDirectory $V9Dir
$t1 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 4) -Once -At "06:00"
$s1 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "WowzaPredict" -TaskPath "\Wowza\" -Action $a1 -Trigger $t1 -Settings $s1 -RunLevel Highest -Force
Write-Host "OK: WowzaPredict - every 4 hours from 06:00" -ForegroundColor Green

# Task 2: Update results every 2 hours (07:00, 09:00, 11:00, ...)
$a2 = New-ScheduledTaskAction -Execute $Python -Argument "update_results.py" -WorkingDirectory $V9Dir
$t2 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 2) -Once -At "07:00"
$s2 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "WowzaUpdateResults" -TaskPath "\Wowza\" -Action $a2 -Trigger $t2 -Settings $s2 -RunLevel Highest -Force
Write-Host "OK: WowzaUpdateResults - every 2 hours from 07:00" -ForegroundColor Green

# Task 3: Telegram notifier - 5 min after each predict
$a3 = New-ScheduledTaskAction -Execute $Python -Argument "telegram_bot\notifier.py" -WorkingDirectory $V9Dir
$t3 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 4) -Once -At "06:05"
$s3 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "WowzaTelegram" -TaskPath "\Wowza\" -Action $a3 -Trigger $t3 -Settings $s3 -RunLevel Highest -Force
Write-Host "OK: WowzaTelegram - every 4 hours from 06:05" -ForegroundColor Green

Write-Host ""
Write-Host "Done. To check: Task Scheduler -> Task Scheduler Library -> Wowza" -ForegroundColor Cyan
Write-Host "To remove all: Unregister-ScheduledTask -TaskPath '\Wowza\' -Confirm:`$false"
