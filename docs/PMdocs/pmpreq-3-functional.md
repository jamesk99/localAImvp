# PMP Project Requirements - Part 3: Functional Requirements

## 1. User Stories and Use Cases

### 1.1 Primary User Stories

**US-001: Document Ingestion**
- **As a** user
- **I want to** add documents to the system
- **So that** I can query them using natural language

**Acceptance Criteria:**
- System accepts PDF, TXT, and MD files
- Documents are parsed and chunked automatically
- Progress is displayed during ingestion
- Duplicate documents are automatically detected and skipped
- Ingestion completes successfully for valid documents

---

**US-002: Natural Language Query**
- **As a** user
- **I want to** ask questions about my documents
- **So that** I can quickly find relevant information

**Acceptance Criteria:**
- System accepts natural language questions
- Relevant context is retrieved from documents
- Response is generated based on retrieved context
- Sources are cited with relevance scores
- Response time is under 30 seconds

---

**US-003: Source Verification**
- **As a** user
- **I want to** see which documents were used to generate the answer
- **So that** I can verify the accuracy and trace back to original content

**Acceptance Criteria:**
- Source documents are listed with each response
- Relevance scores are displayed for each source
- Text previews show the actual content used
- Metadata includes filename and file type

---

**US-004: Incremental Updates**
- **As a** user
- **I want to** add new documents without re-processing existing ones
- **So that** I can efficiently expand my knowledge base

**Acceptance Criteria:**
- System detects already-ingested documents
- Only new or modified documents are processed
- Processing statistics show new vs. existing documents
- Database remains consistent across updates

---

**US-005: System Monitoring**
- **As a** user
- **I want to** check the status of my document database
- **So that** I can verify what has been ingested and troubleshoot issues

**Acceptance Criteria:**
- Can view total documents and chunks
- Can list all ingested documents with timestamps
- Can check for new documents in the raw directory
- Can verify database integrity

---

### 1.2 Use Case Diagrams

**UC-001: Ingest Documents**
```
Actor: User
Precondition: Documents exist in data/raw/
Main Flow:
1. User runs python src/ingest.py
2. System scans data/raw/ directory
3. System checks each file against tracking database
4. System skips already-ingested files
5. System parses new documents
6. System splits text into chunks
7. System generates embeddings via Ollama
8. System stores vectors in ChromaDB
9. System updates tracking database
10. System displays completion statistics

Postcondition: New documents indexed and queryable
Alternative Flows:
- 5a. PDF parsing fails → Skip file, log error, continue
- 7a. Ollama unavailable → Display error, abort
- Empty document → Skip with warning
```

---

**UC-002: Query Documents (Interactive Mode)**
```
Actor: User
Precondition: Documents have been ingested
Main Flow:
1. User runs python src/query.py
2. System loads vector database
3. System initializes LLM connection
4. System displays prompt for question
5. User enters question
6. System embeds question
7. System retrieves top-K relevant chunks
8. System filters by similarity threshold
9. System sends context to LLM
10. LLM generates response
11. System displays answer and sources
12. Return to step 4 (loop until user quits)

Postcondition: User has received answers to questions
Alternative Flows:
- 3a. Vector database not found → Display error, exit
- 10a. LLM fails (OOM) → Retry with fallback model
- 5a. User types 'quit' → Exit gracefully
```

---

**UC-003: Query Documents (Single Query Mode)**
```
Actor: User
Precondition: Documents have been ingested
Main Flow:
1. User runs python src/query.py "question text"
2. System loads vector database
3. System initializes LLM connection
4. System processes the provided question
5. System embeds question
6. System retrieves top-K relevant chunks
7. System filters by similarity threshold
8. System sends context to LLM
9. LLM generates response
10. System displays answer and sources
11. System exits

Postcondition: User has received answer to specific question
Alternative Flows: Same as UC-002
```

---

**UC-004: Database Management**
```
Actor: User
Precondition: System has been used at least once
Main Flow:
1. User runs python src/db_manager.py --all
2. System reads tracking database
3. System displays statistics
4. System lists all ingested documents
5. System checks for new documents in raw directory
6. System verifies database integrity
7. System displays results

Postcondition: User understands current system state
Alternative Flows:
- User can specify individual flags (--stats, --list, etc.)
- User can remove documents from tracking with --remove
```

---

## 2. Functional Requirements by Component

### 2.1 Document Ingestion (src/ingest.py)

**FR-ING-001: File Discovery**
- System SHALL scan data/raw/ directory for supported file types
- System SHALL support *.pdf, *.txt, and *.md file patterns
- System SHALL ignore hidden files and system files
- System SHALL report count of files found

**FR-ING-002: Duplicate Detection**
- System SHALL calculate SHA-256 hash for each file
- System SHALL compare hash against tracking database
- System SHALL skip files with matching hash and size
- System SHALL report skipped vs. new files separately

**FR-ING-003: Document Parsing**
- System SHALL extract text from PDF files using PyPDF2
- System SHALL read text files with UTF-8 encoding
- System SHALL read Markdown files as plain text
- System SHALL handle multi-page PDFs
- System SHALL skip empty or unreadable files with warnings

**FR-ING-004: Text Chunking**
- System SHALL split documents using SentenceSplitter
- System SHALL create chunks of 1024 tokens (configurable)
- System SHALL overlap chunks by 128 tokens (configurable)
- System SHALL preserve sentence boundaries where possible
- System SHALL track actual chunk count per document

**FR-ING-005: Embedding Generation**
- System SHALL connect to Ollama at configured base URL
- System SHALL use nomic-embed-text model for embeddings
- System SHALL generate 768-dimensional vectors
- System SHALL handle Ollama connection errors gracefully
- System SHALL display progress during embedding

**FR-ING-006: Vector Storage**
- System SHALL store embeddings in ChromaDB
- System SHALL use persistent storage in data/vectordb/
- System SHALL store metadata with each chunk (filename, file_path, file_type)
- System SHALL use collection name from config
- System SHALL maintain data persistence across sessions

**FR-ING-007: Tracking Database Updates**
- System SHALL record each ingested document in tracking.db
- System SHALL store file_path, file_hash, file_size, ingested_at, num_chunks
- System SHALL use INSERT OR REPLACE for idempotency
- System SHALL display updated statistics after completion

**FR-ING-008: Progress Reporting**
- System SHALL display header with pipeline description
- System SHALL show which files are new vs. skipped
- System SHALL display progress bars for chunk processing
- System SHALL report final statistics (docs processed, chunks created)
- System SHALL indicate next steps to user

---

### 2.2 Query System (src/query.py)

**FR-QRY-001: System Initialization**
- System SHALL load existing ChromaDB vector store
- System SHALL verify collection exists before proceeding
- System SHALL initialize Ollama embedding model
- System SHALL initialize Ollama LLM with fallback
- System SHALL display initialization status

**FR-QRY-002: Embedding Model Configuration**
- System SHALL use same embedding model as ingestion (nomic-embed-text)
- System SHALL connect to Ollama at configured base URL
- System SHALL handle embedding model unavailability

**FR-QRY-003: LLM Configuration**
- System SHALL attempt to use primary LLM (llama3:latest)
- System SHALL fall back to secondary LLM if primary fails
- System SHALL set temperature to 0.1 for deterministic responses
- System SHALL set timeout to 180 seconds
- System SHALL report which LLM is being used

**FR-QRY-004: Query Processing**
- System SHALL accept natural language text as input
- System SHALL embed query using same model as documents
- System SHALL retrieve top-K chunks (K=5, configurable)
- System SHALL filter results by similarity threshold (0.5, configurable)
- System SHALL handle empty query input gracefully

**FR-QRY-005: Context Assembly**
- System SHALL combine retrieved chunks into context
- System SHALL preserve chunk ordering by relevance
- System SHALL include chunk text and metadata
- System SHALL format context for LLM consumption

**FR-QRY-006: Prompt Engineering**
- System SHALL use custom prompt template
- System SHALL instruct LLM to provide structured responses
- System SHALL require direct answer, supporting details, and implications
- System SHALL instruct LLM to indicate when context is insufficient
- System SHALL pass both context and user question to LLM

**FR-QRY-007: Response Generation**
- System SHALL send assembled prompt to LLM
- System SHALL stream or wait for complete response
- System SHALL handle LLM errors (timeout, OOM)
- System SHALL retry with fallback LLM on OOM errors
- System SHALL return generated text

**FR-QRY-008: Response Formatting**
- System SHALL display answer text clearly
- System SHALL list source chunks with relevance scores
- System SHALL show filename for each source
- System SHALL preview first 300 characters of each source chunk
- System SHALL format scores to 3 decimal places

**FR-QRY-009: Interactive Mode**
- System SHALL display welcome message and instructions
- System SHALL prompt user for questions in a loop
- System SHALL process each question independently
- System SHALL allow user to quit with 'quit', 'exit', or 'q'
- System SHALL handle keyboard interrupts gracefully

**FR-QRY-010: Single Query Mode**
- System SHALL accept question as command-line argument
- System SHALL process single question and exit
- System SHALL format output identically to interactive mode
- System SHALL support multi-word questions with proper parsing

---

### 2.3 Document Tracking (src/document_tracker.py)

**FR-TRK-001: Database Initialization**
- System SHALL create tracking.db if it doesn't exist
- System SHALL create documents table with proper schema
- System SHALL create indexes on file_hash and ingested_at
- System SHALL use absolute paths for database location
- System SHALL ensure parent directory exists

**FR-TRK-002: File Hash Calculation**
- System SHALL compute SHA-256 hash of file contents
- System SHALL read files in 4096-byte chunks for efficiency
- System SHALL handle large files without loading fully into memory
- System SHALL return hexadecimal hash string

**FR-TRK-003: Ingestion Status Check**
- System SHALL query database for existing file_path
- System SHALL compare stored hash with current file hash
- System SHALL compare stored size with current file size
- System SHALL return True only if both hash and size match
- System SHALL return False if file not found or changed

**FR-TRK-004: Document Recording**
- System SHALL accept file_path and num_chunks as parameters
- System SHALL calculate current hash and size
- System SHALL generate ISO timestamp for ingested_at
- System SHALL use INSERT OR REPLACE for updates
- System SHALL commit transaction immediately

**FR-TRK-005: Statistics Retrieval**
- System SHALL count total documents in database
- System SHALL sum total chunks across all documents
- System SHALL find earliest ingestion timestamp
- System SHALL find latest ingestion timestamp
- System SHALL return dictionary with statistics

**FR-TRK-006: Document Listing**
- System SHALL retrieve all documents ordered by ingestion date (newest first)
- System SHALL return file_path, ingested_at, num_chunks, status
- System SHALL handle empty database gracefully
- System SHALL format results as list of dictionaries

**FR-TRK-007: Document Removal**
- System SHALL accept file_path for removal
- System SHALL delete record from database
- System SHALL commit transaction
- System SHALL NOT delete actual file or vector embeddings

**FR-TRK-008: Integrity Verification**
- System SHALL execute SQLite PRAGMA integrity_check
- System SHALL return True if result is 'ok'
- System SHALL return False on any error or failure
- System SHALL handle database connection errors

---

### 2.4 Database Management (src/db_manager.py)

**FR-DBM-001: Statistics Display**
- System SHALL display total documents count
- System SHALL display total chunks count
- System SHALL display first ingestion timestamp
- System SHALL display last ingestion timestamp
- System SHALL display database file location

**FR-DBM-002: Document Listing**
- System SHALL list all ingested documents
- System SHALL display filename, full path, ingestion time, chunk count, status
- System SHALL format timestamps in readable format (YYYY-MM-DD HH:MM:SS)
- System SHALL handle empty database with appropriate message

**FR-DBM-003: New Document Detection**
- System SHALL scan data/raw/ directory for all supported files
- System SHALL check each file against tracking database
- System SHALL identify files not yet ingested
- System SHALL display count of total files vs. new files
- System SHALL list new files with size information
- System SHALL suggest running ingest.py if new files found

**FR-DBM-004: Integrity Verification**
- System SHALL run database integrity check
- System SHALL display OK or FAILED status
- System SHALL display database file path and size
- System SHALL suggest remediation if integrity fails
- System SHALL handle missing database file

**FR-DBM-005: Document Removal**
- System SHALL accept filename as parameter
- System SHALL search for matching documents
- System SHALL handle multiple matches by requesting full path
- System SHALL confirm removal with user feedback
- System SHALL warn that file and vectors are NOT deleted

**FR-DBM-006: Command-Line Interface**
- System SHALL support --stats flag for statistics only
- System SHALL support --list flag for document listing only
- System SHALL support --check flag for new document check
- System SHALL support --verify flag for integrity check
- System SHALL support --remove filename flag for removal
- System SHALL support --all flag for comprehensive report
- System SHALL display help if no flags provided

---

### 2.5 Configuration Management (src/config.py)

**FR-CFG-001: Centralized Configuration**
- System SHALL define all configurable parameters in single file
- System SHALL provide default values for all settings
- System SHALL document purpose of each configuration parameter
- System SHALL be importable by all other modules

**FR-CFG-002: Path Management**
- System SHALL define DATA_DIR relative to project root
- System SHALL define RAW_DOCS_DIR, VECTOR_DB_DIR, TRACKING_DB_PATH
- System SHALL create directories if they don't exist
- System SHALL use os.path.join for cross-platform compatibility

**FR-CFG-003: Model Configuration**
- System SHALL define OLLAMA_BASE_URL for service connection
- System SHALL define LLM_MODEL and LLM_FALLBACK
- System SHALL define EMBED_MODEL
- System SHALL allow easy model switching

**FR-CFG-004: RAG Parameter Configuration**
- System SHALL define CHUNK_SIZE and CHUNK_OVERLAP
- System SHALL define TOP_K retrieval count
- System SHALL define SIMILARITY_THRESHOLD
- System SHALL document impact of changing each parameter

**FR-CFG-005: Collection Management**
- System SHALL define COLLECTION_NAME for ChromaDB
- System SHALL support changing collection name for different projects

---

## 3. Non-Functional Requirements

### 3.1 Usability Requirements

**NFR-USE-001: Installation Simplicity**
- System SHALL require no more than 5 commands to set up
- System SHALL provide clear error messages for missing dependencies
- System SHALL include comprehensive README with setup instructions

**NFR-USE-002: Command-Line Clarity**
- System SHALL display clear prompts and instructions
- System SHALL use emoji or visual markers for different message types
- System SHALL show progress indicators for long-running operations
- System SHALL format output for readability

**NFR-USE-003: Error Messages**
- System SHALL provide actionable error messages
- System SHALL suggest remediation steps for common errors
- System SHALL include relevant context (file names, paths, settings)
- System SHALL avoid cryptic technical jargon where possible

**NFR-USE-004: Documentation**
- System SHALL include README with quick start guide
- System SHALL include technical documentation explaining architecture
- System SHALL include usage examples for all commands
- System SHALL document configuration options and their effects

---

### 3.2 Reliability Requirements

**NFR-REL-001: Data Persistence**
- System SHALL persist vector embeddings across restarts
- System SHALL persist tracking database across restarts
- System SHALL maintain data consistency even after crashes
- System SHALL not corrupt data on abnormal termination

**NFR-REL-002: Error Recovery**
- System SHALL continue processing remaining files after individual file failures
- System SHALL recover from Ollama temporary disconnections
- System SHALL provide fallback LLM when primary fails
- System SHALL maintain database consistency on partial failures

**NFR-REL-003: Idempotency**
- System SHALL produce same results when run multiple times
- System SHALL safely skip already-processed documents
- System SHALL use INSERT OR REPLACE for database operations
- System SHALL not create duplicate embeddings

---

### 3.3 Performance Requirements

**NFR-PERF-001: Ingestion Speed**
- System SHALL process at least 2 documents per minute
- System SHALL display progress for operations taking >5 seconds
- System SHALL not block on individual file processing errors

**NFR-PERF-002: Query Latency**
- System SHALL respond to queries in under 30 seconds (p95)
- System SHALL retrieve vectors in under 500ms
- System SHALL provide feedback during generation

**NFR-PERF-003: Memory Efficiency**
- System SHALL operate within 16GB RAM constraint
- System SHALL not load entire documents into memory
- System SHALL use streaming/chunked reading for large files
- System SHALL handle memory errors gracefully

---

### 3.4 Maintainability Requirements

**NFR-MAIN-001: Code Organization**
- System SHALL use modular architecture with single-responsibility modules
- System SHALL separate configuration from implementation
- System SHALL use descriptive function and variable names
- System SHALL include docstrings for all public functions

**NFR-MAIN-002: Dependency Management**
- System SHALL pin exact versions in requirements.txt
- System SHALL document purpose of each dependency
- System SHALL minimize number of dependencies
- System SHALL use only well-maintained packages

**NFR-MAIN-003: Configurability**
- System SHALL externalize all tunable parameters to config.py
- System SHALL avoid hardcoded values in implementation
- System SHALL document effect of configuration changes
- System SHALL provide sensible defaults

---

### 3.5 Portability Requirements

**NFR-PORT-001: Platform Support**
- System SHALL run on Windows 10/11 without modification
- System SHALL use cross-platform path handling
- System SHALL use Python's built-in os module for file operations
- System SHALL document platform-specific requirements

**NFR-PORT-002: Python Version**
- System SHALL support Python 3.10+
- System SHALL avoid Python 3.12+ only features
- System SHALL use standard library features where possible

---

## 4. Data Requirements

### 4.1 Input Data Requirements

**DR-IN-001: Document Formats**
- System SHALL accept PDF files with text layer
- System SHALL accept plain text files (.txt)
- System SHALL accept Markdown files (.md)
- System SHALL reject unsupported formats with clear message

**DR-IN-002: Document Content**
- System SHALL handle documents from 1KB to 100MB
- System SHALL support multi-page PDFs
- System SHALL handle UTF-8 encoded text
- System SHALL skip documents with no extractable text

**DR-IN-003: File System Organization**
- System SHALL read documents from data/raw/ only
- System SHALL preserve original files unmodified
- System SHALL support nested directories (future enhancement)

---

### 4.2 Output Data Requirements

**DR-OUT-001: Query Responses**
- System SHALL return plain text answers
- System SHALL include source citations
- System SHALL include relevance scores (0.0 to 1.0)
- System SHALL format output for console display

**DR-OUT-002: Status Reports**
- System SHALL report ingestion statistics
- System SHALL report database statistics
- System SHALL report configuration settings in use

---

### 4.3 Data Retention Requirements

**DR-RET-001: Vector Storage**
- System SHALL retain embeddings indefinitely
- System SHALL not auto-delete old embeddings
- System SHALL allow manual cleanup via database reset

**DR-RET-002: Tracking Database**
- System SHALL retain all ingestion records
- System SHALL maintain audit trail of all ingestions
- System SHALL support manual removal of specific records

---

## 5. Interface Requirements

### 5.1 User Interface Requirements

**IF-UI-001: Command-Line Interface**
- System SHALL provide CLI for all operations
- System SHALL accept command-line arguments for single-query mode
- System SHALL provide interactive mode for multi-query sessions
- System SHALL display help text when invoked incorrectly

**IF-UI-002: Visual Feedback**
- System SHALL use consistent formatting for messages
- System SHALL use emoji or symbols to indicate status (✅ ❌ ⚠️ 🔍)
- System SHALL draw visual separators for sections
- System SHALL use colors if terminal supports them (future)

---

### 5.2 External Interface Requirements

**IF-EXT-001: Ollama API**
- System SHALL communicate with Ollama via HTTP REST API
- System SHALL send embedding requests to /api/embeddings endpoint
- System SHALL send generation requests to /api/generate endpoint
- System SHALL handle 404, 500, and timeout errors

**IF-EXT-002: ChromaDB Interface**
- System SHALL use ChromaDB Python client library
- System SHALL use persistent client with disk storage
- System SHALL use get_or_create pattern for collections
- System SHALL include metadata with all vectors

**IF-EXT-003: SQLite Interface**
- System SHALL use Python sqlite3 standard library
- System SHALL use context managers for connections
- System SHALL use parameterized queries to prevent injection
- System SHALL handle locked database gracefully

---

## 6. Validation and Testing Requirements

### 6.1 Unit Testing Requirements

**VT-UNIT-001: Component Testing**
- DocumentTracker SHALL have tests for hash calculation
- DocumentTracker SHALL have tests for duplicate detection
- Configuration SHALL be validated for required parameters
- File parsing SHALL be tested with sample documents

---

### 6.2 Integration Testing Requirements

**VT-INT-001: End-to-End Testing**
- System SHALL be tested with complete ingestion-query cycle
- System SHALL be tested with test document set
- System SHALL verify query results against known answers
- System SHALL measure and log performance metrics

**VT-INT-002: Error Scenario Testing**
- System SHALL be tested with Ollama offline
- System SHALL be tested with corrupted PDFs
- System SHALL be tested with empty documents
- System SHALL be tested with missing database files

---

### 6.3 Acceptance Testing Requirements

**VT-ACC-001: Success Criteria Validation**
- System SHALL correctly answer 7/10 test questions (70% accuracy)
- System SHALL ingest 20-50 documents successfully
- System SHALL respond to queries in under 30 seconds
- System SHALL demonstrate incremental update capability

---

**Document Status:** APPROVED  
**Next Document:** PMP Project Requirements - Part 4: Quality & Risk Management
