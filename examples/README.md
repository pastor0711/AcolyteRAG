# SillyTavern Integration

Run AcolyteRAG as a proxy server between SillyTavern and your LLM. SillyTavern sends its chat history, the server retrieves relevant memories, injects them into the prompt, and forwards everything to the LLM. SillyTavern doesn't need any extensions or changes — it just thinks it's talking to a normal API.

```
SillyTavern → proxy server (retrieval + injection) → LLM → response back
```

## Setup

1. Install dependencies:
```bash
pip install acolyterag flask requests
```

2. Pick the proxy that matches your LLM setup:

| File | Use when |
|---|---|
| `proxy_local.py` | Ollama, llama.cpp, LM Studio, etc. |
| `proxy_openrouter.py` | Using OpenRouter API |

3. Edit the config at the top of the file:
   - **Ollama:** defaults to `http://localhost:11434` — just run it
   - **Other local LLMs:** change `LLM_BASE` to your endpoint

4. Run the proxy:
```bash
python proxy_local.py
# or
python proxy_openrouter.py
```

5. In SillyTavern, change your API endpoint to:
```
http://localhost:5111/v1
```

SillyTavern will auto-detect available models via `/v1/models`.

That's it. Every generation will now include retrieved memories automatically.

## Tuning

All retrieval settings are in the proxy file — adjust `max_retrieved`, `exclude_last_n`, or pass custom scoring weights. No SillyTavern changes needed.
