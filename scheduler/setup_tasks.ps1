# Wowza Task Scheduler Setup
# Run as Administrator

$Cmd = "C:\Windows\System32\cmd.exe"

# Task 1: Predict every 4 hours from 06:00
$a1 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\predict.bat"
$t1 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 4) -Once -At "06:00"
$s1 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaPredict" -TaskPath "\Wowza\" -Action $a1 -Trigger $t1 -Settings $s1 -RunLevel Highest -Force
Write-Host "OK: WowzaPredict - every 4 hours from 06:00" -ForegroundColor Green

# Task 2: Update results every 2 hours from 07:00
$a2 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\update_results.bat"
$t2 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 2) -Once -At "07:00"
$s2 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaUpdateResults" -TaskPath "\Wowza\" -Action $a2 -Trigger $t2 -Settings $s2 -RunLevel Highest -Force
Write-Host "OK: WowzaUpdateResults - every 2 hours from 07:00" -ForegroundColor Green

# Task 3: Telegram notifier - 5 min after each predict
$a3 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\telegram.bat"
$t3 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 4) -Once -At "06:05"
$s3 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaTelegram" -TaskPath "\Wowza\" -Action $a3 -Trigger $t3 -Settings $s3 -RunLevel Highest -Force
Write-Host "OK: WowzaTelegram - every 4 hours from 06:05" -ForegroundColor Green

# Task 4: Git push - 10 min after each predict
$a4 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\git_push.bat"
$t4 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 4) -Once -At "06:10"
$s4 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaGitPush" -TaskPath "\Wowza\" -Action $a4 -Trigger $t4 -Settings $s4 -RunLevel Highest -Force
Write-Host "OK: WowzaGitPush - every 4 hours from 06:10" -ForegroundColor Green

# Task 5: Live scanner loop - starts at boot, runs continuously (self-throttles)
$a5 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\live_loop.bat"
$t5 = New-ScheduledTaskTrigger -AtStartup
$s5 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 24) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaLive" -TaskPath "\Wowza\" -Action $a5 -Trigger $t5 -Settings $s5 -RunLevel Highest -Force
Write-Host "OK: WowzaLive - starts at boot, runs 30s/scan when live" -ForegroundColor Green

# Task 7: World Cup drift tracker - every 2 hours from 08:00
$a5 = New-ScheduledTaskAction -Execute $Cmd -Argument "/c C:\WowzaBot\worldcup.bat"
$t5 = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 2) -Once -At "08:00"
$s5 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun
Register-ScheduledTask -TaskName "WowzaWorldCup" -TaskPath "\Wowza\" -Action $a5 -Trigger $t5 -Settings $s5 -RunLevel Highest -Force
Write-Host "OK: WowzaWorldCup - every 2 hours from 08:00" -ForegroundColor Green

Write-Host ""
Write-Host "Done. Tasks are in Task Scheduler -> Task Scheduler Library -> Wowza" -ForegroundColor Cyan
