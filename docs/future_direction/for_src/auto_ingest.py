"""
Scheduled document ingestion script.
Checks data/raw directory for new files and ingests them.
Designed to be run periodically via Task Scheduler or cron.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from config import RAW_DOCS_DIR, TRACKING_DB_PATH
from document_tracker import DocumentTracker
from ingest import ingest_documents

# Configure logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'auto_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_for_new_documents():
    """
    Check for new documents and return count of unprocessed files.
    """
    tracker = DocumentTracker(TRACKING_DB_PATH)
    raw_path = Path(RAW_DOCS_DIR)
    
    # Get all document files
    file_patterns = ["*.txt", "*.pdf", "*.md"]
    files = []
    for pattern in file_patterns:
        files.extend(raw_path.glob(pattern))
    
    # Count new files
    new_files = []
    for file_path in files:
        if not tracker.is_document_ingested(file_path):
            new_files.append(file_path)
    
    return new_files


def run_scheduled_ingest():
    """
    Main function to run scheduled ingestion.
    Checks for new documents and ingests them if found.
    """
    logger.info("="*60)
    logger.info(f"🔍 SCHEDULED INGESTION CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    logger.info(f"Checking directory: {RAW_DOCS_DIR}")
    
    # Check for new documents
    new_files = check_for_new_documents()
    
    if not new_files:
        logger.info("✅ No new documents found. Database is up to date.")
        logger.info("="*60)
        return
    
    logger.info(f"📥 Found {len(new_files)} new document(s):")
    for file_path in new_files:
        logger.info(f"   - {file_path.name}")
    
    logger.info("\n🚀 Starting ingestion...")
    
    try:
        # Run ingestion
        ingest_documents()
        logger.info("\n✅ Scheduled ingestion completed successfully")
    except Exception as e:
        logger.error(f"\n❌ Scheduled ingestion failed: {e}", exc_info=True)
        raise
    
    logger.info("="*60)


if __name__ == "__main__":
    run_scheduled_ingest()
