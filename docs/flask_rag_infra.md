# Flask-Based RAG Infrastructure & Configuration

This document explains the **v2 RAG architecture** using Flask, how the components fit together, what ports are exposed, and how to configure and run the system. It is written so you can share it with the team.

---

## 1. High-Level Architecture

### Components

- **Browser client (LAN)**
  - Serves the chat UI from `src/index.html`.
  - Talks only to the Flask app via HTTP.

- **Flask RAG API (Python)** – `src/app.py`
  - Exposes HTTP endpoints:
    - `GET /` → serves the chat UI (`index.html`).
    - `POST /api/query` → RAG query endpoint used by the UI.
  - Handles:
    - HTTP Basic Auth (username/password popup).
    - Logging (centralized, env-configured).
    - Calling the RAG pipeline in `query.py`.

- **RAG pipeline (Python)** – `src/query.py`
  - Uses `chromadb` and `llama_index`:
    - `initialize_rag_system()` → loads vector DB and sets up embeddings/LLM.
    - `create_query_engine(index)` → configures retriever and response synthesizer.
    - `format_response(response)` → normalizes the answer and source chunks.

- **Ollama LLM server**
  - Runs locally on the RAG machine.
  - Default URL used by Python code: `http://localhost:11434`.
  - **Not** exposed to LAN clients; only Flask/RAG code talks to it.

### Request Flow

1. User opens browser on LAN and goes to:
   - `http://<RAG_APP_HOST>:<RAG_APP_PORT>/`  
     (e.g. `http://192.168.1.23:8000/`)

2. Browser receives:
   - Basic Auth popup (username/password).
   - After successful auth, `index.html` chat UI.

3. When user sends a question:
   - Frontend calls `POST /api/query` with JSON:
     ```json
     { "question": "..." }
     ```

4. Flask RAG API:
   - Validates auth and request body.
   - Logs the request.
   - Calls `query_engine.query(question)`.
   - Formats the answer and sources via `format_response`.
   - Logs timing and source info.
   - Returns JSON back to the browser.

5. RAG pipeline:
   - Uses the vector database in `data/vectordb` (built by `src/ingest.py`).
   - Retrieves relevant chunks with `chromadb` + `llama_index`.
   - Calls Ollama locally (e.g. `localhost:11434`) to generate the answer.

6. Browser updates the chat UI with the answer text.

### Architecture Diagram & Flows

The key design goal is that **both the terminal and the browser paths use the same RAG core**, so fidelity of service is the same regardless of how a user asks a question.

#### Diagram

```text
           +-----------------+                      +----------------------+
           |  Terminal User  |                      |   Browser User (LAN) |
           +--------+--------+                      +-----------+----------+
                    |                                           |
                    |  (CLI: python src/query.py)               |  (HTTP: POST /api/query)
                    v                                           v
           +------------------------+                 +------------------------+
           |  query.py (CLI entry) |                 |   Flask RAG API        |
           |  - main()             |                 |   (src/app.py)         |
           +-----------+-----------+                 +-----------+------------+
                       \_____________________________/           |
                                         |                        |
                                         v                        |
                            +-----------------------------+       |
                            |  RAG Core (shared)          |       |
                            |  - initialize_rag_system()  |       |
                            |  - create_query_engine()    |       |
                            |  - query_engine.query(...)  |       |
                            +---------------+-------------+       |
                                            |                     |
                                            v                     v
                                 +---------------------+  +---------------------+
                                 |  Ollama @          |  |  HTTP JSON Response |
                                 |  localhost:11434   |  |  to Browser UI      |
                                 +----------+----------+  +----------+----------+
                                            |                        |
                                            v                        v
                                 +---------------------+  +---------------------+
                                 |  Answer + Sources   |  |  Answer + Sources   |
                                 |  (printed in CLI)   |  |  (rendered in UI)   |
                                 +---------------------+  +---------------------+
```

#### Fidelity of Service

- Both paths (terminal and browser) call the **same vector store**, **same embeddings**, **same LLM configuration**, and the **same query engine** implementation.
- As long as the question text and configuration (models, thresholds, vector DB) are the same, **you should expect equivalent answers and retrieved sources**, regardless of whether the user comes in through the CLI or the browser.
- Differences will mainly be in:
  - Presentation (CLI text vs. rich UI).
  - Logging context (CLI vs. HTTP metadata like IP/user).

---

## 2. Network & Port Model

### Intentional Exposure

- **Exposed to LAN:** Flask app port only
  - `RAG_APP_PORT` (default `8000`).
  - Accessed as `http://<RAG_APP_HOST>:<RAG_APP_PORT>/`.

- **Not exposed to LAN:** Ollama port
  - Ollama listens on `localhost:11434`.
  - Only the Python RAG code can talk to it.
  - Windows Defender inbound rules for port `11434` are **not** required and should be removed/disabled.

### Firewall Summary

- **Open inbound:** Flask port (e.g. TCP 8000 or 8100) from trusted LAN.
- **Closed/blocked inbound:** port 11434 (Ollama), and any other internal ports.

This keeps the large language model server private while providing a controlled RAG API over the LAN.

---

## 3. Flask RAG API Details (`src/app.py`)

### Routes

- `GET /`
  - Auth-protected.
  - Logs `event=serve_index` with user and client IP.
  - Serves `index.html` from the `src` directory.

- `POST /api/query`
  - Auth-protected.
  - Request body: `{ "question": "<user question>" }`.
  - Response format (from `format_response`):
    ```json
    {
      "answer": "<RAG-generated answer>",
      "sources": [
        {
          "chunk_id": 1,
          "text": "Snippet of source text ...",
          "score": 0.92,
          "metadata": { "filename": "...", ... }
        },
        ...
      ]
    }
    ```

### Basic Auth

- Implemented in `app.py` via:
  - `check_auth(username, password)`
  - `authenticate()`
  - `requires_auth` decorator, applied to both routes.

- Credentials are read from environment variables:
  - `RAG_USER` – username (default `raguser`).
  - `RAG_PASS` – password (default `changeme`).

- Behavior:
  - Browser hits `/` or `/api/query`.
  - If no/invalid credentials, Flask responds `401` with a `WWW-Authenticate` header.
  - Browser shows the built-in username/password popup.
  - On success, credentials are reused for subsequent API calls.

These defaults are for development only; in practice, `RAG_USER` and `RAG_PASS` should be set explicitly.

#### How `RAG_USER` / `RAG_PASS` map to what users type

- On the **server**, Flask reads:
  - `RAG_USER` → becomes the allowed username.
  - `RAG_PASS` → becomes the allowed password.
- In the **browser**, when the popup appears, the person must type **exactly those same values**.
- In other words: the env vars define “the one credential pair that is accepted,” and the browser popup is just how the user sends that pair to the server.

---

## 4. Frontend Wiring (`src/index.html`)

The chat UI has been updated to talk to the Flask API instead of directly to Ollama.

- Configuration in JS:
  ```js
  const API_URL = "/api/query";
  ```

- Query function:
  ```js
  async function queryRAG(message) {
      const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: message })
      });
      const data = await response.json();
      return data.answer || "No answer available.";
  }
  ```

The rest of the UI (message list, thinking indicator, copy/regenerate buttons) remains unchanged.

---

## 5. Logging System

The Flask app uses Python's `logging` module with a centralized configuration.

### Logger

- Named logger: `rag_app`.
- Format:
  ```
  %(asctime)s | %(levelname)s | %(name)s | %(message)s
  ```

### Configuration via Environment Variables

- `RAG_LOG_LEVEL`
  - Default: `INFO`.
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

- `RAG_LOG_FILE`
  - If set, logs are written to this file using a `RotatingFileHandler`:
    - `maxBytes = 5 MB`
    - `backupCount = 3`
  - Example value: `C:\\logs\\rag_app.log`.

- `RAG_LOG_QUESTIONS`
  - Default: `true`.
  - If set to `false`, logs will not include the text preview of user questions:
    - Still logs `q_len` (length).
    - `q_preview` is blank.

### Logged Events

- On startup:
  - `"RAG system initialized"` – after the vector index and query engine are created.

- On `GET /`:
  - `event=serve_index user=<username> ip=<client_ip>`

- On `POST /api/query`:
  - **Request received**:
    - `event=query_received`
    - `request_id=<uuid>`
    - `user=<username>`
    - `ip=<client_ip>`
    - `q_len=<question length>`
    - `q_preview=<first 200 chars or empty>`

  - **Success**:
    - `event=query_success`
    - `request_id=<same id>`
    - `duration_ms=<elapsed time>`
    - `source_count=<number of source chunks>`
    - `top_score=<score of first source if available>`

  - **Error**:
    - `event=query_error`
    - `request_id=<same id>`
    - Full stack trace (logged via `logger.exception`).

This provides traceability and basic observability now, and can be upgraded to JSON or shipped to a centralized log system later without changing application logic.

---

## 6. Configuration Summary (Environment Variables)

### Auth & API

- `RAG_USER` – Basic Auth username.
- `RAG_PASS` – Basic Auth password.
- `RAG_APP_HOST` – Flask bind host (default `0.0.0.0`).
- `RAG_APP_PORT` – Flask bind port (default `8000`).

### Logging

- `RAG_LOG_LEVEL` – log level (`INFO` by default).
- `RAG_LOG_FILE` – optional path for rotating log file.
- `RAG_LOG_QUESTIONS` – `true`/`false` to control question text logging.

### Ollama & RAG

- `OLLAMA_BASE_URL` – base URL for Ollama (default `http://localhost:11434`).
- `OLLAMA_HOST` – how Ollama itself binds; should be `127.0.0.1:11434` or unset for v2.
- Other RAG settings are in `src/config.py` (chunk sizes, similarity thresholds, collection names, etc.).

---

## 7. Runbook (How to Start the System)

1. **Install dependencies** (from project root):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the vector database**:
   ```bash
   python src/ingest.py
   ```

3. **Start Ollama** on the RAG machine
   - Ensure it listens on `localhost:11434` (or consistent with `OLLAMA_BASE_URL`).

4. **Set environment variables** (example using PowerShell on Windows):
   ```powershell
   # Auth
   $env:RAG_USER = "youruser"
   $env:RAG_PASS = "yourstrongpassword"

   # Logging
   $env:RAG_LOG_LEVEL = "INFO"           # or DEBUG during development
   # Optional
   # $env:RAG_LOG_FILE = "C:\\logs\\rag_app.log"
   # $env:RAG_LOG_QUESTIONS = "false"   # if you don't want question text in logs

   # Network
   $env:RAG_APP_HOST = "192.168.1.23"    # or 0.0.0.0
   $env:RAG_APP_PORT = "8000"           # or your chosen port
   ```

5. **Run the Flask app**:
   ```powershell
   python src/app.py
   ```

6. **Access from the LAN**:
   - Open a browser on another machine:
     - `http://192.168.1.23:8000/` (adjust to match your host/port).
   - Enter the Basic Auth credentials.
   - Use the chat UI; responses come from the Python RAG pipeline.

---

## 8. Future Enhancements (Not Implemented Yet)

- Switch to **JSON-structured logging** for easier integration with log aggregators.
- Add a `/health` endpoint for basic health checks (DB, Ollama, etc.).
- Surface `sources` in the UI (e.g. clickable citations with filenames and scores).
- Replace Basic Auth with a more advanced auth system (per-user accounts, roles, SSO) if needed.
- Introduce metrics (latency histograms, query counts, error rates) via Prometheus or similar.

---

## 9. Why Flask Was Chosen for v2

This project could have used several different approaches (Node.js backend, FastAPI, a reverse proxy-only setup, etc.). Flask was chosen for **v2** for the following reasons:

- **Same language as the RAG core**
  - The RAG pipeline (`query.py`, `config.py`, `ingest.py`) is already written in Python.
  - Flask runs in the same Python process and can directly `import query` and reuse:
    - `initialize_rag_system()`
    - `create_query_engine()`
    - `format_response()`
  - This avoids cross-language integration, extra processes, or RPC layers at this stage.

- **Very small surface area / minimal boilerplate**
  - v2 needs only two endpoints: `GET /` and `POST /api/query`.
  - Flask handles this with a few decorators and minimal configuration.
  - The goal is to stand up something reliable quickly without committing to a heavy framework.

- **Easy to keep Ollama private**
  - Flask runs on the same machine as Ollama and the vector DB.
  - The browser only talks to Flask, and Flask talks to `localhost:11434`.
  - This design keeps the LLM server off the network while still allowing LAN access to the RAG UI.

- **No CORS complexity for the browser**
  - Flask serves both the HTML/JS (`GET /`) and the RAG API (`/api/query`) from the same origin.
  - Same-origin traffic means no need for CORS headers, OPTIONS handling, or proxy workarounds in v2.

- **Simple, but upgradeable**
  - If the system grows, Flask does not block future changes:
    - A reverse proxy (nginx, Caddy, etc.) can be added in front for TLS, routing, and rate limiting.
    - The app can be containerized and scaled horizontally behind a load balancer.
    - If needed, the API layer could later be rewritten in FastAPI or another framework while keeping the Python RAG core.

- **Why not Node.js for v2?**
  - A Node server would either:
    - Duplicate RAG logic in JavaScript/TypeScript, or
    - Call out to the Python RAG process over HTTP/CLI, adding an extra hop.
  - For this phase, that adds complexity without clear benefit, since most of the intelligence already lives in Python.

In short, Flask gives a **thin, Python-native HTTP layer** that sits directly on top of the existing RAG components, keeps Ollama unexposed, avoids CORS issues, and is easy for the team to understand and evolve in future versions.

### Running the app

Defaults (no env set):
host = 0.0.0.0 → listens on all interfaces (LAN-ready).
port = 8000.
debug = False, use_reloader = False → more production-like: no debugger, no auto-reload.
Development mode:
Set FLASK_DEBUG=1 (or true, yes, on) to enable Flask debug and reloader.
Bind to a specific LAN IP / different port (PowerShell example):

```powershell
# Auth
$env:RAG_USER = "youruser"
$env:RAG_PASS = "yourstrongpassword"

# Network (pick your LAN IP + port)
$env:RAG_APP_HOST = "192.168.1.23"
$env:RAG_APP_PORT = "8000"

# Optional debug (only during dev)
# $env.FLASK_DEBUG = "1"

python src/app.py
```

From another machine on the LAN:

Visit http://192.168.1.23:8000/
Browser shows Basic Auth prompt (uses RAG_USER/RAG_PASS).
After login, use the chat; answers + cited sources should appear.
