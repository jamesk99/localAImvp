# RAG System v2 Info

## Overview

This document summarizes the v2 updates to the Phase 0 RAG system and web interface.

Goals:
- **Use the Python RAG pipeline** (Chroma + llama-index) from the browser UI.
- **Expose a minimal HTTP API** on the LAN (Flask).
- **Protect the UI with basic username/password auth**.
- **Add lightweight but scalable logging** for requests and RAG behavior.

---

## New Flask RAG API (`src/app.py`)

A small Flask app now wraps the existing RAG pipeline defined in `src/query.py`.

- **Initialization (on startup)**
  - Imports existing RAG helpers:
    - `initialize_rag_system()`
    - `create_query_engine()`
    - `format_response()`
  - Runs once at module import:
    - `index = initialize_rag_system()`
    - `query_engine = create_query_engine(index)`

- **Routes**
  - `GET /`
    - Serves `src/index.html` via `send_from_directory`.
    - Same origin as the API, so **no CORS needed**.
  - `POST /api/query`
    - Expects JSON: `{ "question": "..." }`.
    - Calls `query_engine.query(question)`.
    - Uses `format_response(response)` to build the JSON payload:
      - `answer` (string)
      - `sources` (list of chunks with score/metadata)

- **Running**
  - Entrypoint:
    - `python src/app.py`
  - Host/port can be controlled via env vars:
    - `RAG_APP_HOST` (default: `0.0.0.0`)
    - `RAG_APP_PORT` (default: `8000`)

---

## Frontend Wiring (`src/index.html`)

The chat UI is now wired to the Flask RAG API instead of directly to Ollama.

- **Old behavior**
  - Called Ollama at `http://<host>:11434/api/generate` directly from the browser.
  - Bypassed `query.py` and the vector database.

- **New behavior**
  - Defines:
    - `const API_URL = "/api/query";`
  - `queryRAG(message)` now:
    - Sends `POST /api/query` with JSON `{ question: message }`.
    - Expects JSON response and returns `data.answer` to the UI.
  - All UI behavior (message bubbles, "Thinking...", etc.) is unchanged.

Effect: the browser now talks to **Flask**; Flask talks to **RAG + Ollama** on the backend.

---

## HTTP Basic Auth (Username/Password Popup)

The Flask app is protected with simple HTTP Basic Authentication.

- **Configuration**
  - Credentials come from environment variables:
    - `RAG_USER` (default: `raguser`)
    - `RAG_PASS` (default: `changeme`)
  - If env vars are not set, defaults are used (not recommended for real use).

- **Implementation**
  - Helper functions in `app.py`:
    - `check_auth(username, password)`
    - `authenticate()` – returns `401` with `WWW-Authenticate` header.
    - `requires_auth` decorator – wraps protected routes.
  - Applied to routes:
    - `@requires_auth` on `GET /` and `POST /api/query`.

- **UX**
  - First visit to `http://<server>:8000/` triggers the browser's **Basic Auth popup**.
  - Once authenticated, the browser reuses the credentials for subsequent API calls.

This is intentionally minimal but adequate for LAN-only v2.

---

## Logging System (Centralized and Scalable)

A lightweight, centralized logging setup was added to `src/app.py` using Python's `logging` module.

### Configuration

- Environment variables:
  - `RAG_LOG_LEVEL`
    - Default: `INFO`.
    - Typical values: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
  - `RAG_LOG_FILE`
    - If set, logs are also written to a rotating file at this path.
    - Uses `RotatingFileHandler` with:
      - `maxBytes = 5 MB`
      - `backupCount = 3`.
  - `RAG_LOG_QUESTIONS`
    - Default: `true`.
    - If set to `false`, the system logs question length but **not** the text preview.

- Logger name:
  - `rag_app`

- Format:
  - `%(asctime)s | %(levelname)s | %(name)s | %(message)s`

### Events Logged

- On RAG system initialization:
  - `"RAG system initialized"`

- On `GET /` (serve index):
  - `event=serve_index user=<username> ip=<client_ip>`

- On `POST /api/query`:
  - **Before query**:
    - `event=query_received`
    - `request_id=<uuid4 hex>`
    - `user=<username>`
    - `ip=<client_ip>`
    - `q_len=<length of question>`
    - `q_preview=<first 200 chars or empty if RAG_LOG_QUESTIONS=false>`
  - **On success**:
    - `event=query_success`
    - `request_id=<same id>`
    - `duration_ms=<elapsed time>`
    - `source_count=<len(result["sources"])>`
    - `top_score=<score of first source if available>`
  - **On error**:
    - `event=query_error`
    - `request_id=<same id>`
    - Full stack trace (via `logger.exception`).

This gives a clear, structured trail for each query without heavy infra.

---

## Running the v2 RAG System (Quick Reference)

1. **Install dependencies** (from project root):

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure vector DB is built**:

   ```bash
   python src/ingest.py
   ```

3. **Start Ollama** on the RAG machine (bound to localhost or as configured in `config.py`).

4. **Set credentials and logging options** (PowerShell example):

   ```powershell
   $env:RAG_USER = "youruser"
   $env:RAG_PASS = "yourstrongpassword"
   $env:RAG_LOG_LEVEL = "INFO"          # or DEBUG
   # Optional file logging
   # $env:RAG_LOG_FILE = "C:\\path\\to\\rag_app.log"
   # Optional: disable logging question text
   # $env:RAG_LOG_QUESTIONS = "false"
   ```

5. **Run the Flask app**:

   ```powershell
   python src/app.py
   ```

   Optionally set host/port:

   ```powershell
   $env:RAG_APP_HOST = "192.168.1.23"
   $env:RAG_APP_PORT = "8000"
   python src/app.py
   ```

6. **Access the UI from LAN devices**:

   - Navigate to: `http://<RAG_APP_HOST>:<RAG_APP_PORT>/`
   - Enter Basic Auth credentials when prompted.
   - Chat via the UI; all answers are served by the Python RAG pipeline.

---

## Future v3+ Ideas (Not Implemented Yet)

- JSON-formatted logs for ingestion into centralized log systems (ELK, Loki, etc.).
- Exposing source chunks in the frontend UI (e.g., showing filenames and scores per answer).
- Additional endpoints for health checks and metrics (e.g., `/health`, `/metrics`).
- More granular auth (per-user accounts, roles, rate limiting) if needed beyond LAN.
