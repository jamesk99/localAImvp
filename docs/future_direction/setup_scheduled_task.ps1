# PowerShell script to set up Windows Task Scheduler for auto-ingestion
# Run this script as Administrator to create a scheduled task

param(
    [int]$IntervalDays = 2,
    [string]$TaskName = "RAG_AutoIngest"
)

Write-Host "=" * 60
Write-Host "Setting up Scheduled Task for Auto-Ingestion"
Write-Host "=" * 60

# Get the project directory
$ProjectDir = Split-Path -Parent $PSScriptRoot
$ScriptPath = Join-Path $ProjectDir "src\auto_ingest.py"
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

# Check if virtual environment exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Virtual environment not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Please create a virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Check if script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Script not found at: $ScriptPath" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Task Name: $TaskName"
Write-Host "  Interval: Every $IntervalDays days"
Write-Host "  Python: $VenvPython"
Write-Host "  Script: $ScriptPath"
Write-Host ""

# Create the scheduled task action
$Action = New-ScheduledTaskAction `
    -Execute $VenvPython `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $ProjectDir

# Create the trigger (runs every N days)
$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -DaysInterval $IntervalDays `
    -At "2:00AM"

# Create task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false

# Task principal (run whether user is logged on or not)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Limited

# Register the task
try {
    # Remove existing task if it exists
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($ExistingTask) {
        Write-Host "Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    
    # Register new task
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Automatically checks for new documents in data/raw and ingests them into the RAG system every $IntervalDays days."
    
    Write-Host ""
    Write-Host "SUCCESS: Scheduled task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  - Runs every $IntervalDays days at 2:00 AM"
    Write-Host "  - Checks data/raw for new documents"
    Write-Host "  - Logs saved to: logs/auto_ingest.log"
    Write-Host ""
    Write-Host "To manage the task:" -ForegroundColor Yellow
    Write-Host "  - View: Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  - Run now: Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  - Disable: Disable-ScheduledTask -TaskName '$TaskName'"
    Write-Host "  - Remove: Unregister-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "You can also manage it via Task Scheduler GUI (taskschd.msc)"
    Write-Host "=" * 60
    
} catch {
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
    exit 1
}
