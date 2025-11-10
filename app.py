# app.py — FastAPI wrapper for FinPilot (RAG + Ollama LLM)
# ---------------------------------------------------------
# Endpoints:
#   GET  /                 → redirects to the beautiful static frontend
#   GET  /health           → service status
#   POST /ingest           → ingest folder or single PDF (legacy)
#   POST /ingest-file      → upload PDF from frontend (drag & drop)
#   POST /ask              → ask questions using RAG (supports model override)
#
# Run: double-click start_all.bat
# Frontend: http://127.0.0.1:8000  (auto-opens)
# ---------------------------------------------------------

import os
import tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag
import llm


# ---------- FastAPI app ----------
app = FastAPI(
    title="FinPilot API",
    version="1.0.0",
    description="RAG-powered financial document assistant using Ollama",
    docs_url="/docs",
    redoc_url=None,
)

# CORS – wide open for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request models ----------
class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to folder or single PDF file")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Question to the assistant")
    k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")
    source: Optional[str] = Field(None, description="Filter by exact PDF filename")
    model: Optional[str] = Field(None, description="Override OLLAMA_MODEL for this request")


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to the static frontend (ChatGPT-style UI)"""
    return HTMLResponse('<script>window.location="/static/index.html"</script>')


@app.get("/health", response_class=JSONResponse)
def health() -> Dict[str, Any]:
    """Return service status and current configuration"""
    return {
        "status": "ok",
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "finpilot_documents"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:2b"),
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """Legacy endpoint – ingest a folder or a single PDF from a file path"""
    path = req.path.strip()
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        if os.path.isdir(path):
            results = rag.ingest_folder(path)
            return {"type": "folder", "path": path, "results": results}
        else:
            result = rag.ingest_pdf(path)
            return {"type": "file", "path": path, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    """
    New endpoint used by the frontend.
    Receives a PDF via drag & drop, saves it temporarily, and ingests it.
    """
    # Create a temporary file that persists after closing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag.ingest_pdf(tmp_path)
        os.unlink(tmp_path)  # delete temp file
        return {"result": f"PDF '{file.filename}' loaded successfully", "details": result}
    except Exception as e:
        os.unlink(tmp_path)  # clean up on error
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    """Answer a question using RAG – model can be overridden per request"""
    original_model = os.getenv("OLLAMA_MODEL")

    # Temporarily override model if a different one is requested
    if req.model:
        os.environ["OLLAMA_MODEL"] = req.model

    where_filter = {"source": req.source} if req.source else None

    try:
        response = llm.answer_from_rag(
            question=req.question,
            k=req.k,
            where=where_filter,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        # Always restore the original model (safe with --reload and concurrent requests)
        if req.model and original_model:
            os.environ["OLLAMA_MODEL"] = original_model


# ---------- Serve static frontend (no templates needed) ----------
app.mount("/static", StaticFiles(directory="static", html=True), name="static")