"""
AcolyteRAG proxy for local LLMs (Ollama, llama.cpp, LM Studio, etc.)

SillyTavern points its API endpoint to this server.
It retrieves relevant memories and injects them before forwarding to the LLM.

Usage:
    1. Edit LLM_BASE below to match your local LLM's endpoint
    2. python proxy_local.py
    3. In SillyTavern, set API URL to http://localhost:5111/v1
"""

from flask import Flask, request, Response, jsonify
from acolyterag import retrieve_related_messages
import requests

# --- Config ---
LLM_BASE = "http://localhost:11434"  # Ollama base URL
PROXY_PORT = 5111
MAX_RETRIEVED = 4
EXCLUDE_LAST_N = 6
# --------------

LLM_CHAT = f"{LLM_BASE}/v1/chat/completions"

app = Flask(__name__)


# === MODELS ===

@app.route("/v1/models")
@app.route("/models")
def list_models():
    try:
        # Standard OpenAI format (LM Studio, llama.cpp)
        resp = requests.get(f"{LLM_BASE}/v1/models", timeout=10)
        if resp.status_code == 200:
            return jsonify(resp.json())
    except Exception:
        pass

    try:
        # Fallback to Ollama's /api/tags
        resp = requests.get(f"{LLM_BASE}/api/tags", timeout=10)
        data = resp.json()
        
        result = {"object": "list", "data": []}
        for m in data.get("models", []):
            result["data"].append({
                "id": m["name"],
                "object": "model",
                "created": 0,
                "owned_by": m.get("details", {}).get("family", "local"),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"object": "list", "data": [], "error": str(e)}), 502


# === CHAT ===

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])

    # Retrieve relevant memories from chat history
    memories = retrieve_related_messages(
        messages,
        query_text=messages[-1]["content"] if messages else "",
        max_retrieved=MAX_RETRIEVED,
        exclude_last_n=EXCLUDE_LAST_N,
    )

    if memories:
        memory_block = "\n".join(m["content"] for m in memories)
        memory_msg = {
            "role": "system",
            "content": f"Relevant context from earlier in the conversation:\n{memory_block}",
        }
        insert_at = next(
            (i for i, m in enumerate(messages) if m["role"] != "system"),
            len(messages),
        )
        messages = messages[:insert_at] + [memory_msg] + messages[insert_at:]
        data["messages"] = messages

    # Forward to local LLM
    streaming = data.get("stream", False)
    resp = requests.post(LLM_CHAT, json=data, stream=streaming)

    # Build response, stripping chunked transfer-encoding headers
    # (we de-chunk the body, so Content-Length replaces it)
    safe_headers = {}
    for k, v in resp.headers.items():
        kl = k.lower()
        if kl in ("transfer-encoding", "content-encoding"):
            continue
        safe_headers[k] = v

    if streaming:
        return Response(
            resp.iter_content(chunk_size=1024),
            status=resp.status_code,
            headers=safe_headers,
        )
    return (resp.content, resp.status_code, safe_headers)


if __name__ == "__main__":
    print(f"AcolyteRAG proxy running on http://localhost:{PROXY_PORT}")
    print(f"Forwarding to: {LLM_BASE}")
    print(f"Set SillyTavern API URL to: http://localhost:{PROXY_PORT}/v1")
    app.run(port=PROXY_PORT)
