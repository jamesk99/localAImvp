# RAG Environment & Ports Guide

This doc explains **when you actually need to set environment variables** for the Flask-based RAG app, what the defaults are, and how to override them (with Windows / PowerShell examples).

---

## 1. Do I always have to set ports or other details?

**Short answer: No.**

For a simple local/dev run on the RAG box:

- You can usually just run:
  ```powershell
  cd notebook\e6-phase0-dadproject\src
  python app.py
  ```
- If you do **nothing else**, the app will use sensible defaults:
  - `RAG_APP_HOST` → `0.0.0.0` (bind to all interfaces, LAN-ready)
  - `RAG_APP_PORT` → `8000`
  - `FLASK_DEBUG` → `False` (because it is off unless explicitly set)
  - `RAG_USER` → `"raguser"`
  - `RAG_PASS` → `"changeme"`

You **should** override at least the auth variables (`RAG_USER`, `RAG_PASS`) for real usage, but **you don’t have to set ports every time** unless you want something different from the defaults.

Use env vars when you need to:

- Change which IP / port Flask listens on.
- Tighten or loosen logging.
- Change credentials.
- Change which Ollama models or base URL are used.

---

## 2. Overview of the main environment variables

The important env vars are grouped into:

- **Auth & API / network**
- **Logging**
- **Ollama & RAG models**

### 2.1 Auth & API / Network

These are consumed mainly in `src/app.py`.

#### `RAG_USER` (recommended to set)

- **Purpose**: HTTP Basic Auth username for both `/` and `/api/query`.
- **Default**: `"raguser"`.
- **When to set**: Almost always, to avoid leaving the default in place.

Example (PowerShell):
```powershell
$env:RAG_USER = "youruser"
```

#### `RAG_PASS` (recommended to set)

- **Purpose**: HTTP Basic Auth password.
- **Default**: `"changeme"`.
- **When to set**: Always for anything beyond pure local testing.

Example:
```powershell
$env:RAG_PASS = "yourstrongpassword"
```

#### `RAG_APP_HOST`

- **Purpose**: What IP address Flask binds to.
- **Default**: `"0.0.0.0"` (all interfaces). This is **already LAN-friendly**.
- **Typical values**:
  - `0.0.0.0` – listen on all interfaces (default; fine for internal LAN server).
  - `127.0.0.1` – only local machine (no LAN access).
  - `192.168.x.y` – a specific LAN IP.

Example (bind to a specific LAN IP):
```powershell
$env:RAG_APP_HOST = "192.168.1.23"
```

#### `RAG_APP_PORT`

- **Purpose**: TCP port used by the Flask app.
- **Default**: `"8000"`.
- **When to change**:
  - If port 8000 is already in use.
  - If you want a different convention (e.g., 8100).

Example:
```powershell
$env:RAG_APP_PORT = "8100"
```

#### `FLASK_DEBUG`

- **Purpose**: Controls Flask debug mode and auto-reload.
- **Default**: _off_ (debug = `False`), which is more "production-ish".
- **Accepted truthy values**: `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive).

Behavior in `app.py`:

```python
host = os.environ.get("RAG_APP_HOST", "0.0.0.0")
port = int(os.environ.get("RAG_APP_PORT", "8000"))
debug_env = os.environ.get("FLASK_DEBUG", "").lower()
debug = debug_env in ("1", "true", "yes", "on")
app.run(host=host, port=port, debug=debug, use_reloader=debug)
```

- If `FLASK_DEBUG` is **not** set → `debug=False`, `use_reloader=False`.
- If `FLASK_DEBUG=1` → classic Flask debug server with reloader.

Example (development session only):
```powershell
$env:FLASK_DEBUG = "1"
```

> In short: **You do not need to set `FLASK_DEBUG` for normal LAN usage.** Only set it when actively developing.

---

### 2.2 Logging

These tune the `logging` behavior in `src/app.py`.

#### `RAG_LOG_LEVEL`

- **Purpose**: Sets the verbosity of logs for the `rag_app` logger.
- **Default**: `"INFO"`.
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

Example (more verbose while debugging):
```powershell
$env:RAG_LOG_LEVEL = "DEBUG"
```

#### `RAG_LOG_FILE`

- **Purpose**: Path to the rotating log file (now set, always on by default).
- **Default**: `logs/rag_app.log` (relative to project root). **File logging is now always on by default.** Was not set in previous versions (logs go to stderr/console only).
- **Handler behavior**:
  - Rotating file up to **5 MB** per file.
  - Keeps **10 backups** (~50 MB total history).
  - Log directory is created automatically if it doesn't exist.

Example (custom location):
```powershell
$env:RAG_LOG_FILE = "C:\\logs\\rag_app.log"
```

**Note**: You no longer need to set this variable unless you want logs in a different location. The app will always write to `logs/rag_app.log` by default.

#### `RAG_LOG_QUESTIONS`

- **Purpose**: Controls whether the log includes a preview of the user question text.
- **Default**: `"true"`.
- **If set to `"false"`**:
  - Logs still record question length (`q_len`).
  - `q_preview` is blanked out.

Example (privacy-focused logging):
```powershell
$env:RAG_LOG_QUESTIONS = "false"
```

---

### 2.3 Ollama & RAG Models (from `src/config.py`)

These live in `config.py` and are read by the RAG pipeline (`query.py`, `ingest.py`). Defaults are fine unless you intentionally want to change models or base URLs.

#### `OLLAMA_BASE_URL`

- **Purpose**: Where the Python code expects Ollama to be reachable.
- **Default**: `"http://localhost:11434"`.
- Usually, you leave this alone and ensure Ollama is running on the same box.

Example (only if you changed Ollama’s bind address):
```powershell
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
```

#### `LLM_MODEL`

- **Purpose**: Primary LLM model name to use via Ollama.
- **Default**: `"llama3:latest"` (per current `config.py`).

Example:
```powershell
$env:LLM_MODEL = "llama3:8b"
```

#### `LLM_FALLBACK`

- **Purpose**: Fallback LLM if the primary fails (e.g., OOM).
- **Default**: `"deepseek-r1:latest"`.

Example (changing fallback):
```powershell
$env:LLM_FALLBACK = "llama3:latest"
```

#### `EMBED_MODEL`

- **Purpose**: Embedding model used for vector storage.
- **Default**: `"nomic-embed-text"`.

Example:
```powershell
$env:EMBED_MODEL = "nomic-embed-text"
```

> Note: Changing embeddings or LLMs after data has been ingested may require re-ingesting documents to keep things consistent.

---

## 3. Recommended setups

### 3.1 Quick local/dev run (single machine)

When you’re just testing on the same box:

1. Ensure Ollama is running and the vector DB is built:
   ```powershell
   cd notebook\e6-phase0-dadproject\src
   python ingest.py
   ```

2. Optionally set safer credentials (recommended):
   ```powershell
   $env:RAG_USER = "devuser"
   $env:RAG_PASS = "devpassword"
   ```

3. (Optional) More verbose logs + debug:
   ```powershell
   $env:RAG_LOG_LEVEL = "DEBUG"
   $env:FLASK_DEBUG  = "1"   # for development only
   ```

4. Run the app:
   ```powershell
   python app.py
   ```

5. Visit from the same machine:
   - `http://localhost:8000/`


### 3.2 LAN-accessible "almost prod" setup

When you want other machines on the LAN to use the RAG UI:

1. On the RAG server, choose IP + port (or keep defaults):
   ```powershell
   # Auth
   $env:RAG_USER = "lanuser"
   $env:RAG_PASS = "lan-strong-password"

   # Network (optional if you like the defaults)
   $env:RAG_APP_HOST = "192.168.1.23"   # or leave unset to use 0.0.0.0
   $env:RAG_APP_PORT = "8000"           # or change if needed

   # Logging (optional)
   $env:RAG_LOG_LEVEL = "INFO"
   # $env:RAG_LOG_FILE = "C:\\logs\\rag_app.log"
   # $env:RAG_LOG_QUESTIONS = "false"   # if you don’t want text previews logged
   ```

2. **Do not** set `FLASK_DEBUG` in this mode (keeps debug server off).

3. Start the app:
   ```powershell
   cd notebook\e6-phase0-dadproject\src
   python app.py
   ```

4. From another LAN machine, open:
   - `http://192.168.1.23:8000/` (adjust IP/port as needed)
   - Browser will prompt for the Basic Auth credentials you set above.

---

## 4. TL;DR

- You **don’t** have to set ports/host every time; defaults (`0.0.0.0:8000`) are already good for LAN in many cases.
- You **should** override `RAG_USER` and `RAG_PASS` for anything beyond quick local testing.
- Use `FLASK_DEBUG` only when developing; leave it unset for a more production-like run.
- Logging and model env vars are **optional tuning knobs** you use when you need more visibility or want different models.
