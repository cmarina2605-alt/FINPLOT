# llm.py (uses prompts.py for prompt building)
# ------------------------------------------------------------
# Purpose:
# - Take a question + context (retrieved chunks) and ask a local LLM
#   running on Ollama for a concise, cited answer.
# - Prompt text & style are centralized in prompts.py.
# ------------------------------------------------------------

import os
import requests
from typing import Dict, Any, List, Optional

import prompts  # centralized system prompt + builder

# ---- Config ----
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# Safety caps to avoid sending giant prompts
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))  # model context window

# (optional) allow forcing CPU or limiting GPU via env (if you added this earlier)
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))  # 0 = CPU-only; omit if not needed


# ------------------------------------------------------------
# Prompt builder (now delegates to prompts.py)
# ------------------------------------------------------------
def build_prompt(question: str, context: str) -> str:
    """Use prompts.render_prompt to format SYSTEM + CONTEXT + QUESTION."""
    return prompts.render_prompt(
        question=question,
        context=context,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

def ensure_model_downloaded(model: str):
    """Checks if the model exists locally in Ollama, pulls it if missing."""
    # Ollama endpoint para ver modelos locales
    resp = requests.get("http://localhost:11434/api/models")
    if resp.ok:
        local_models = [m["name"] for m in resp.json()]
        if model not in local_models:
            print(f"[INFO] Model '{model}' not found locally, pulling...")
            pull_resp = requests.post("http://localhost:11434/api/pull", json={"model": model})
            if pull_resp.ok:
                print(f"[OK] Model '{model}' downloaded successfully")
            else:
                print(f"[ERROR] Failed to pull model '{model}': {pull_resp.text}")
    else:
        print(f"[ERROR] Failed to list Ollama models: {resp.text}")
# ------------------------------------------------------------
# Low-level Ollama call (non-streaming for simplicity)
# ------------------------------------------------------------
import subprocess
import requests

def generate_with_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Llama al endpoint de Ollama y devuelve la respuesta.
    Si el modelo no está descargado, lo descarga automáticamente.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_gpu": NUM_GPU,
        },
    }

    try:
        print(f"[DEBUG] Calling Ollama with model={model}...")
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)

        if resp.status_code == 404:
            # Modelo no encontrado, intentamos descargarlo
            print(f"[WARNING] Model '{model}' not found. Pulling it...")
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"[INFO] Model '{model}' downloaded. Retrying request...")
            # Reintento
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)

        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to pull model '{model}': {e}")
        return f"ERROR pulling model '{model}': {e}"

    except Exception as e:
        print(f"[ERROR] Failed to call Ollama: {e}")
        return f"ERROR calling Ollama: {e}"


# ------------------------------------------------------------
# Public function: question + context -> final answer text
# ------------------------------------------------------------
def answer(question: str, context: str, model: str = MODEL_NAME) -> str:
    prompt = build_prompt(question, context)
    return generate_with_ollama(prompt, model=model)



# ------------------------------------------------------------
# Convenience: do the whole RAG round-trip in one call
# ------------------------------------------------------------
def answer_from_rag(
    question: str,
    k: int = 4,
    where: Optional[Dict[str, Any]] = None,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    1) Retrieves top-k chunks from Chroma (via rag.py)
    2) Builds the context block with citations
    3) Asks the local LLM via Ollama
    4) Returns answer + citations + raw context
    """
    import rag  # lazy import to avoid circular deps

    search = rag.query(question, k=k, where=where)
    context = rag.build_context(search)
    text = answer(question, context, model=model)

    citations: List[Dict[str, Any]] = [
        {
            "source": r["source"],
            "page": r.get("page"),  # puede ser None
            "chunk": r.get("chunk"),
        }
        for r in search.get("results", [])
    ]
    return {"answer": text, "citations": citations, "context": context}


# ------------------------------------------------------------
# Minimal CLI for quick testing
#   python llm.py "What’s the equity limit for a moderate-risk investor?"
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        q = " ".join(sys.argv[1:])
        out = answer_from_rag(q, k=4)
        print("\n=== ANSWER ===\n")
        print(out["answer"])
        print("\n=== CITATIONS ===")
        for c in out["citations"]:
            print(f"- {c['source']} p.{c['page']} c.{c['chunk']}")
    else:
        print("Usage:\n  python llm.py \"your question here\"")