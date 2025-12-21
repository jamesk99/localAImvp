# Auto-Ingestion Guide

This guide explains how to set up automatic document ingestion that runs periodically without requiring a background process.

## Overview

The system uses the tracking database (`data/tracking.db`) to remember which documents have been ingested. When you add new documents to `data/raw`, they will be automatically processed on the next scheduled run.

## Two Ways to Run Ingestion

### 1. Manual Ingestion (Recommended for Testing)

Run this anytime to manually check and ingest new documents:

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Run manual ingestion
python src/ingest.py
```

This will:
- Check which documents are already in the database
- Only process NEW documents
- Skip already-ingested files

### 2. Scheduled Auto-Ingestion

Set up a Windows Task Scheduler job to run ingestion automatically every N days.

#### Setup (One-Time)

1. **Open PowerShell as Administrator**

2. **Run the setup script:**
   ```powershell
   # Default: runs every 2 days at 2:00 AM
   .\setup_scheduled_task.ps1
   
   # Custom interval (e.g., every 3 days)
   .\setup_scheduled_task.ps1 -IntervalDays 3
   ```

3. **Verify the task was created:**
   ```powershell
   Get-ScheduledTask -TaskName "RAG_AutoIngest"
   ```

#### Managing the Scheduled Task

**View task details:**
```powershell
Get-ScheduledTask -TaskName "RAG_AutoIngest" | Format-List *
```

**Run the task immediately (for testing):**
```powershell
Start-ScheduledTask -TaskName "RAG_AutoIngest"
```

**Check task history:**
```powershell
Get-ScheduledTaskInfo -TaskName "RAG_AutoIngest"
```

**Disable the task (without deleting):**
```powershell
Disable-ScheduledTask -TaskName "RAG_AutoIngest"
```

**Enable the task:**
```powershell
Enable-ScheduledTask -TaskName "RAG_AutoIngest"
```

**Remove the task:**
```powershell
Unregister-ScheduledTask -TaskName "RAG_AutoIngest" -Confirm:$false
```

**View logs:**
```powershell
Get-Content logs\auto_ingest.log -Tail 50
```

## How It Works

1. **Document Tracking**: Every ingested document is recorded in `data/tracking.db` with:
   - File path
   - File hash (to detect modifications)
   - Ingestion timestamp
   - Number of chunks created

2. **Scheduled Check**: The scheduled task runs `src/auto_ingest.py` which:
   - Scans `data/raw` for `.txt`, `.pdf`, and `.md` files
   - Checks each file against the tracking database
   - Only processes files that are NEW or MODIFIED
   - Logs all activity to `logs/auto_ingest.log`

3. **No Duplicates**: The tracking database ensures documents are never processed twice unless they've been modified.

## Workflow Example

```
Day 1:
  - Add document1.pdf to data/raw
  - Run: python src/ingest.py
  - document1.pdf is ingested ✓

Day 2:
  - Scheduled task runs at 2:00 AM
  - No new documents found
  - Nothing happens ✓

Day 3:
  - Add document2.txt and document3.md to data/raw
  - Scheduled task runs at 2:00 AM (Day 3)
  - Both new documents are ingested ✓
  - document1.pdf is skipped (already ingested) ✓

Day 5:
  - Scheduled task runs at 2:00 AM
  - No new documents
  - Nothing happens ✓
```

## Verifying Everything Works

### Test the scheduled script manually:
```powershell
.venv\Scripts\activate
python src/auto_ingest.py
```

Expected output if no new documents:
```
============================================================
🔍 SCHEDULED INGESTION CHECK - 2025-11-02 02:00:00
============================================================
Checking directory: data/raw
✅ No new documents found. Database is up to date.
============================================================
```

### Test with a new document:
```powershell
# Add a test file
echo "Test document" > data/raw/test.txt

# Run the scheduled script
python src/auto_ingest.py
```

Expected output:
```
============================================================
🔍 SCHEDULED INGESTION CHECK - 2025-11-02 02:00:00
============================================================
Checking directory: data/raw
📥 Found 1 new document(s):
   - test.txt

🚀 Starting ingestion...
[... ingestion process ...]
✅ Scheduled ingestion completed successfully
============================================================
```

## Troubleshooting

### Task doesn't run
- Check if task is enabled: `Get-ScheduledTask -TaskName "RAG_AutoIngest"`
- View task history in Task Scheduler GUI (`taskschd.msc`)
- Check logs: `logs/auto_ingest.log`

### Documents not being ingested
- Verify files are in `data/raw`
- Check file extensions (must be `.txt`, `.pdf`, or `.md`)
- Run manually to see error messages: `python src/auto_ingest.py`

### Want to re-ingest a document
The tracking database prevents re-ingestion. To force re-ingest:
1. Delete the entry from tracking database, OR
2. Use the reset option: `python src/ingest.py --reset` (re-ingests ALL documents)

## Configuration

### Change Schedule Interval
Re-run the setup script with a different interval:
```powershell
.\setup_scheduled_task.ps1 -IntervalDays 7  # Weekly
```

### Change Schedule Time
Edit the task in Task Scheduler GUI or modify `setup_scheduled_task.ps1`

## Benefits of This Approach

✅ **No background process** - Task Scheduler handles scheduling  
✅ **Lightweight** - Only runs when scheduled  
✅ **Reliable** - Windows Task Scheduler is robust  
✅ **Logged** - All activity saved to `logs/auto_ingest.log`  
✅ **Manual override** - Can still run `python src/ingest.py` anytime  
✅ **Smart tracking** - Never processes the same document twice  
✅ **Flexible** - Easy to change schedule or disable  

## Files Created

- `src/auto_ingest.py` - Scheduled ingestion script
- `setup_scheduled_task.ps1` - Task Scheduler setup script
- `logs/auto_ingest.log` - Ingestion activity log
- `data/tracking.db` - Document tracking database (already existed)
