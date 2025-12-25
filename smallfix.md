# Small Fixes / Immediate Upgrades (Ollama + Ryzen AI / 128GB)

This repo uses **LlamaIndex** with:
- `llama_index.llms.ollama.Ollama` for chat/completions
- `llama_index.embeddings.ollama.OllamaEmbedding` for embeddings
- ChromaDB for the vector store

The main knobs are currently controlled in `src/config.py` via environment variables:
- `OLLAMA_BASE_URL`
- `LLM_MODEL`
- `LLM_FALLBACK`
- `EMBED_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD`

Below is a **ranked** list of the best “small changes” to get immediate benefit on a stronger machine.

---

## 1) Upgrade the models (highest impact, lowest effort)
**Where:** `src/config.py` (env vars) and Ollama model pulls.

**What to do:** Set a stronger `LLM_MODEL` and a sensible `LLM_FALLBACK`.

### Recommended pattern
- **Primary** = best model you can run *comfortably*.
- **Fallback** = smaller model that is *always reliable*.

Right now your defaults are:
- `LLM_MODEL = llama3:latest`
- `LLM_FALLBACK = deepseek-r1:latest`

That fallback is likely **larger/heavier** than the primary, which defeats the purpose.

### Suggested replacements (pick based on your hardware)
Because you only specified CPU/RAM (128GB) and not GPU/VRAM, these are safe starting points for a strong CPU box:
- **Primary**: `qwen2.5:14b-instruct` or `llama3.1:8b-instruct` (fast-ish, high quality)
- **Fallback**: `llama3.2:3b-instruct` or `qwen2.5:7b-instruct` (reliable)

If your Ryzen AI machine has a strong GPU/VRAM or you’re comfortable with slower CPU inference:
- **Primary**: `qwen2.5:32b-instruct`
- **Fallback**: `qwen2.5:14b-instruct`

### How to apply (no code changes)
Set env vars when starting the app:
- `LLM_MODEL`
- `LLM_FALLBACK`
- `EMBED_MODEL`

Example (PowerShell):
```powershell
$env:LLM_MODEL = "qwen2.5:14b-instruct"
$env:LLM_FALLBACK = "llama3.2:3b-instruct"
$env:EMBED_MODEL = "nomic-embed-text"
python .\src\app.py
```

---

## 2) Fix the fallback logic to match reality (small config change)
**Where:** `src/config.py`

**What to do:** Make sure your fallback is actually **smaller** than primary.

**Why:** In `src/query.py`, the fallback is used when the primary errors, including common memory/500-type failures.
If fallback is bigger, the retry will be slower and more likely to fail.

---

## 3) Increase effective answer quality by tuning retrieval (cheap and high ROI)
**Where:** `src/config.py` and retrieval code in `src/query.py`.

Current defaults:
- `CHUNK_SIZE = 1024`
- `CHUNK_OVERLAP = 128`
- `TOP_K = 5`
- `SIMILARITY_THRESHOLD = 0.3`

### Recommended quick tweaks
- **If answers feel incomplete**:
  - Increase `TOP_K` to `8` or `10`
- **If you see irrelevant sources**:
  - Increase `SIMILARITY_THRESHOLD` to `0.4` or `0.5`
- **If your docs are very long / technical**:
  - Keep `CHUNK_SIZE` large (1024 is fine), but consider `CHUNK_OVERLAP = 150-200`

Note: after changing chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`) you should re-run ingestion:
- `python .\src\ingest.py`

---

## 4) Raise the model context window (only if your chosen Ollama model supports it)
**Where:** `src/query.py` when constructing `Ollama(...)`.

Right now you set:
- `temperature=0.1`
- `request_timeout=180.0`

**What to add:** an Ollama option for context, typically `num_ctx`.

In LlamaIndex’s Ollama integration, you can usually pass additional generation options through `Ollama(...)` (exact parameter name depends on your installed `llama-index-llms-ollama` version).

Practical suggestion:
- Try `num_ctx = 8192` (or `4096` if you want to be conservative)

If you do this, also consider increasing timeouts (see next item).

---

## 5) Increase timeouts for bigger models (prevents “random 500” failures)
**Where:** `src/query.py`

Current values:
- primary: `request_timeout=180.0`
- fallback: `request_timeout=120.0`

On larger models (or CPU inference), these can be too low.

**Suggested starting point on CPU-heavy inference:**
- primary: `request_timeout=600.0`
- fallback: `request_timeout=300.0`

---

## 6) Add streaming later (bigger UX win, small code change)
**Where:** Flask endpoint `POST /api/query` in `src/app.py`.

Right now, the endpoint blocks until `query_engine.query(question)` finishes.

**Upgrade idea:** stream tokens to the browser (Server-Sent Events) so you see output immediately.
This is one of the best UX improvements once you move to bigger/slower models.

This requires modest code changes:
- a streaming route in Flask
- using Ollama streaming output (or LlamaIndex streaming if supported in your version)

---

## 7) Fix `benchmark.py` config mismatch (small maintenance fix)
**Where:** `src/benchmark.py` imports:
```py
from config import (
    LLM_MODEL, EMBED_MODEL, OLLAMA_BASE_URL,
    USE_ROCM, USE_NPU, GPU_LAYERS, LLM_CONTEXT_WINDOW
)
```

But `src/config.py` currently **does not define** `USE_ROCM`, `USE_NPU`, `GPU_LAYERS`, or `LLM_CONTEXT_WINDOW`.
So `benchmark.py` will fail to run as-is.

**Small fix options:**
- Add those env vars to `src/config.py`, or
- Remove those imports/prints from `benchmark.py`.

---

## Suggested “first hour” upgrade order
1. Set `LLM_MODEL`/`LLM_FALLBACK` to a strong+small pairing.
2. Increase `TOP_K` to 8 and bump `SIMILARITY_THRESHOLD` slightly if needed.
3. Increase `request_timeout` to avoid timeouts on larger models.
4. Only then experiment with `num_ctx` / larger context windows.

---

## What I need from you to fully tailor model picks
Reply with:
- Whether you have a **dedicated GPU** (and VRAM amount), or if this is **CPU-only** inference.
- The exact Ollama models you already have installed (`ollama list`).

With that, I can recommend a precise primary/fallback pair and context window that matches your hardware.

### Current

What I found in src/ (backend + tuning points)
Ollama is the backend, but you’re not calling /api/chat directly.
You’re using LlamaIndex’s Ollama adapters:
src/query.py: from llama_index.llms.ollama import Ollama
src/ingest.py: from llama_index.embeddings.ollama import OllamaEmbedding
Config is centralized in src/config.py via env vars:
OLLAMA_BASE_URL (default h ttp://localhost:11434)
LLM_MODEL, LLM_FALLBACK, EMBED_MODEL
RAG knobs: CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIMILARITY_THRESHOLD

### Future

Should you call the Ollama backend directly?
Not by default. For this repo, staying on LlamaIndex + Ollama is usually the best choice because your app is primarily a RAG pipeline (Chroma retrieval + synthesis), and LlamaIndex already handles a lot of the orchestration cleanly.

That said, there are specific cases where calling Ollama directly is worth it.

Keep LlamaIndex (recommended for this MVP) if…
You want RAG to “just work” with minimal code.
You care about:
retrieval + postprocessing (SimilarityPostprocessor)
response synthesis (get_response_synthesizer)
prompt templating in one place
You’re OK with LlamaIndex’s abstraction of generation options (it can hide some Ollama knobs depending on version).
In your code, src/query.py is already structured around LlamaIndex’s query engine (RetrieverQueryEngine), so switching to raw Ollama would mean rewriting the response path (build prompt manually, attach sources manually, etc.).

Call Ollama directly if you need one of these “power-user” features

1) True token streaming to the UI (best reason)
Your Flask endpoint POST /api/query currently blocks until completion. If you want fast perceived latency on bigger models, direct /api/chat streaming (SSE/NDJSON) is often easier and more controllable than going through multiple abstraction layers.

2) Fine-grained control over Ollama options
If you want to reliably tune things like:

num_ctx
num_predict
num_thread
repeat_penalty
mirostat
stop sequences and you find LlamaIndex isn’t passing them through cleanly in your installed version, direct calls remove uncertainty.
3) Multi-request optimizations
If you want to do patterns like:

parallel retrieval + summarization calls
speculative decoding patterns
caching prompt+KV behavior in a custom way
Direct calls are simpler.
The hybrid approach I’d recommend (best of both)
Keep LlamaIndex for retrieval (it’s doing your chunking/retrieval/sources well already).
Optionally replace only the final “LLM call” with a direct Ollama request when you want streaming or special options.
Conceptually:

Retrieve nodes with LlamaIndex (top_k + similarity cutoff)
Build a prompt with context_str + question
Call Ollama /api/chat directly with streaming enabled
Return sources + streamed tokens
This is a smaller refactor than removing LlamaIndex entirely.

Practical recommendation for you right now
Don’t switch yet.
First do the immediate wins in smallfix.md:
better model pairing (LLM_MODEL + smaller LLM_FALLBACK)
adjust TOP_K / SIMILARITY_THRESHOLD
increase request_timeout
Only move to direct Ollama if:
you want streaming output, or
you hit a wall passing num_ctx/advanced options through LlamaIndex.