# RAG System Improvements

## Changes Made

### 1. **Primary LLM: DeepSeek-R1 with Fallback**
- **Primary LLM**: `deepseek-r1:latest` (more capable reasoning model)
- **Fallback LLM**: `llama3:latest` (automatic fallback if DeepSeek unavailable)
- **Auto-detection**: System tests connection and automatically switches to fallback if needed

### 2. **Improved RAG Parameters**
- **Chunk Size**: Increased from 512 → 1024 tokens (more context per chunk)
- **Chunk Overlap**: Increased from 50 → 128 tokens (better continuity between chunks)
- **Similarity Threshold**: Increased from 0.3 → 0.5 (filters out less relevant results)
- **Temperature**: Set to 0.1 (more focused, deterministic responses)

### 3. **Custom Prompt Template**
Added structured prompt that instructs the LLM to:
- Provide clear, comprehensive answers
- Structure responses with:
  1. Direct answer
  2. Supporting details from context
  3. Relevant implications
- Explicitly state when context is insufficient
- Include specific examples from the documents

### 4. **Better Response Formatting**
- **Source Preview**: Increased from 200 → 300 characters
- **Better Context Display**: Shows more of each retrieved chunk
- **Improved Readability**: Clearer separation of answer and sources

## Expected Improvements

1. **Better Answer Quality**: DeepSeek-R1 provides superior reasoning and comprehension
2. **More Context**: Larger chunks mean the LLM sees more relevant information
3. **Better Relevance**: Higher similarity threshold filters out noise
4. **Structured Responses**: Custom prompt ensures consistent, well-organized answers
5. **Reliability**: Automatic fallback ensures system always works

## Usage

Simply run the query script as before:

```bash
python src/query.py
```

The system will automatically:
1. Try to use DeepSeek-R1
2. Fall back to Llama3 if DeepSeek is unavailable
3. Apply all improvements transparently

## Notes

- **First Run**: If DeepSeek-R1 is not installed, run: `ollama pull deepseek-r1:latest`
- **Fallback**: System gracefully handles missing models
- **Re-indexing**: If you want to use the new chunk sizes, re-run `ingest.py` to rebuild the vector database
