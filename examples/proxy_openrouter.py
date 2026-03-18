"""
AcolyteRAG proxy for OpenRouter + SillyTavern.
Minimal OpenAI-compatible proxy with memory injection.
"""

from flask import Flask, request, Response, jsonify
from acolyterag import retrieve_related_messages
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OPENROUTER_API = "https://openrouter.ai/api/v1"
PROXY_PORT = 5111
MAX_RETRIEVED = 4
EXCLUDE_LAST_N = 2  # only exclude the last 2 messages (current turn)

# Headers we forward from OpenRouter (skip transfer-encoding, content-encoding, etc.)
FORWARD_HEADERS = {"content-type", "date", "server", "x-request-id", "openrouter-processing-ms"}

app = Flask(__name__)


def filter_headers(resp_headers):
    """Only forward safe headers from upstream."""
    return {k: v for k, v in resp_headers.items() if k.lower() in FORWARD_HEADERS}


@app.before_request
def log_request():
    log.info(f">>> {request.method} {request.path} from {request.remote_addr} ua={request.headers.get('User-Agent','?')[:60]}")


# === MODELS ===

@app.route("/v1/models")
@app.route("/models")
def list_models():
    auth = request.headers.get("Authorization", "")
    headers = {"Authorization": auth} if auth else {}
    try:
        resp = requests.get(f"{OPENROUTER_API}/models", headers=headers, timeout=30)
        data = resp.json()
        # Reformat to clean OpenAI-compatible format
        result = {"object": "list", "data": []}
        for m in data.get("data", []):
            result["data"].append({
                "id": m["id"],
                "object": "model",
                "created": m.get("created", 0),
                "owned_by": m.get("id", "").split("/")[0] if "/" in m.get("id", "") else "openrouter",
            })
        log.info(f"Returning {len(result['data'])} models")
        return jsonify(result)
    except Exception as e:
        log.error(f"Models error: {e}")
        return jsonify({"error": str(e)}), 502


# === CHAT COMPLETIONS ===

@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json or {}
    messages = data.get("messages", [])
    streaming = data.get("stream", False)
    model = data.get("model", "openrouter/auto")

    log.info(f"Chat: model={model} stream={streaming} msgs={len(messages)}")

    # RAG injection
    try:
        memories = retrieve_related_messages(
            messages,
            query_text=messages[-1]["content"] if messages else "",
            max_retrieved=MAX_RETRIEVED,
            exclude_last_n=EXCLUDE_LAST_N,
        )
    except Exception as e:
        log.warning(f"RAG error: {e}")
        memories = []

    if memories:
        memory_block = "\n".join(m["content"] for m in memories)
        memory_msg = {"role": "system", "content": f"Relevant context from earlier:\n{memory_block}"}
        insert_at = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
        messages = messages[:insert_at] + [memory_msg] + messages[insert_at:]
        data["messages"] = messages

    # Forward to OpenRouter
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    try:
        resp = requests.post(
            f"{OPENROUTER_API}/chat/completions",
            json=data, headers=headers, stream=streaming, timeout=300,
        )
    except Exception as e:
        log.error(f"Upstream error: {e}")
        return jsonify({"error": {"message": str(e)}}), 502

    if streaming:
        return Response(
            resp.iter_content(chunk_size=1024),
            status=resp.status_code,
            content_type="text/event-stream",
        )

    # Non-streaming: return raw bytes with clean headers (no chunked transfer-encoding)
    return Response(resp.content, status=resp.status_code,
                    content_type=resp.headers.get("Content-Type", "application/json"))


if __name__ == "__main__":
    print(f"Proxy: http://localhost:{PROXY_PORT}")
    print(f"ST API URL: http://localhost:{PROXY_PORT}/v1")
    app.run(host="0.0.0.0", port=PROXY_PORT, threaded=True)
