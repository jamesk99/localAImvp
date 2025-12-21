# Supported Document Formats

The RAG system now supports multiple document formats for ingestion. Simply place your files in the `data/raw/` directory and run the ingestion script.

## Supported Formats

### Text-Based Documents
- **`.txt`** - Plain text files
- **`.md`** - Markdown files

### PDF Documents
- **`.pdf`** - PDF documents (text extraction)

### Microsoft Office
- **`.docx`** - Microsoft Word documents
  - Extracts paragraphs and tables
- **`.xlsx`, `.xls`** - Microsoft Excel spreadsheets
  - Processes all sheets
  - Converts to structured text format

### Data Formats
- **`.csv`** - Comma-separated values
  - Auto-detects delimiter
  - Preserves structure with pipe separators
- **`.json`** - JSON files
  - Formatted for readability

### Web Formats
- **`.html`, `.htm`** - HTML files
  - Strips scripts and styles
  - Extracts clean text content

## Installation

To use all document formats, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

If you only need specific formats, you can install dependencies selectively:

```bash
# For Word documents
pip install python-docx

# For Excel and CSV
pip install pandas openpyxl

# For HTML
pip install beautifulsoup4 lxml
```

## Usage

1. Place your documents in `data/raw/` directory
2. Run the ingestion script:
   ```bash
   python src/ingest.py
   ```
3. The system will automatically detect and process all supported file types

## How It Works

The system uses a modular loader architecture (`document_loaders.py`) that:
- Automatically detects file types by extension
- Uses specialized loaders for each format
- Gracefully handles missing optional dependencies
- Provides clear error messages

## Adding New Formats

To add support for a new file format:

1. Create a new loader class in `document_loaders.py`:
   ```python
   class MyFormatLoader(DocumentLoader):
       @staticmethod
       def load(file_path: Path) -> Optional[str]:
           # Your loading logic here
           return extracted_text
   ```

2. Register it in the `LOADER_REGISTRY`:
   ```python
   LOADER_REGISTRY = {
       # ... existing loaders
       '.myext': MyFormatLoader,
   }
   ```

3. Install any required dependencies in `requirements.txt`

## Notes

- The system tracks ingested documents to avoid duplicates
- Empty files are automatically skipped
- If a loader dependency is missing, you'll see a helpful installation message
- All formats are chunked and embedded using the same pipeline
