# rag.py
# ------------------------------------------------------------
# This script handles all the RAG (Retrieval-Augmented Generation) logic.
# It reads PDFs, splits them into text chunks, creates embeddings, and
# stores them in a local vector database (ChromaDB) for later retrieval.
#
# When we ask a question, it searches the most relevant chunks and
# returns them with metadata (document name, page, etc.).
#
# Requirements (install once):
# pip install chromadb sentence-transformers pymupdf
# ------------------------------------------------------------

import os
import re
import uuid
from typing import List, Dict, Any, Optional, Union

import fitz  # PyMuPDF → used to extract text from PDFs
import chromadb  # our vector database
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# Define where to store the database and which model to use for embeddings
PERSIST_DIR = "chroma_db"  # Folder for Chroma's local database
COLLECTION_NAME = "finpilot_documents"  # Name of our collection (like a table)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
CHUNK_WORDS = 400  # Split PDFs into ~400-word chunks


# ------------------------------------------------------------
# TEXT CLEANING AND CHUNKING HELPERS
# ------------------------------------------------------------

def clean_text(s: str) -> str:
    """Cleans text by removing extra spaces and line breaks."""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def words(text: str) -> List[str]:
    """Splits text into a list of words."""
    return [w for w in re.split(r"\s+", text) if w]

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> List[str]:
    """
    Splits long text into smaller parts (~400 words).
    Each chunk will later be embedded and stored.
    """
    ws = words(text)
    chunks = []
    for i in range(0, len(ws), chunk_words):
        piece = " ".join(ws[i:i + chunk_words]).strip()
        if piece:
            chunks.append(piece)
    return chunks

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Reads a PDF and extracts text page by page.
    Returns a list like [{'page': 1, 'text': '...'}, ...].
    """
    out = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            t = clean_text(doc.load_page(i).get_text("text"))
            if t:
                out.append({"page": i + 1, "text": t})
    return out


# ------------------------------------------------------------
# DATABASE INITIALIZATION
# ------------------------------------------------------------
def get_collection():
    """
    Connects to or creates the local Chroma database.
    Defines how embeddings are generated and stored.
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)  # Ensure folder exists
    client = chromadb.PersistentClient(path=PERSIST_DIR)  # Local persistent DB
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)  # Embedding model

    # Create or load a collection (like a table)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for search
    )
    return col


# ------------------------------------------------------------
# INGESTION: ADD DOCUMENTS TO THE DATABASE
# ------------------------------------------------------------
def ingest_pdf(source: Union[str, bytes], filename: str = None):
    """
    Ingest PDF from path or bytes (frontend upload).
    Always returns a clean dict with status and chunks count.
    """
    try:
        # Open PDF from path or bytes
        if isinstance(source, bytes):
            if filename is None:
                filename = "uploaded.pdf"
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            if not os.path.exists(source):
                raise FileNotFoundError(f"PDF not found: {source}")
            doc = fitz.open(source)
            filename = os.path.basename(source) if filename is None else filename

        # Extract text
        text = ""
        for page in doc:
            text += page.get_text()

        if not text.strip():
            return {"status": "warning", "chunks": 0, "message": "No text found in PDF (scanned image?)"}

        # Split into chunks
        chunks = [text[i:i+400] for i in range(0, len(text), 400)]

        # Embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeds = model.encode(chunks).tolist()

        # Save to Chroma
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection("finpilot_documents")

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeds,
            documents=chunks,
            metadatas=metadatas
        )

        # ALWAYS return a clean dict
        return {
            "status": "ok",
            "chunks": len(chunks),
            "source": filename,
            "message": f"PDF '{filename}' processed ({len(chunks)} chunks)"
        }

    except Exception as e:
        # Never let raw PDF content leak
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }


def ingest_folder(folder: str) -> List[Dict[str, Any]]:
    """Ingests all PDFs inside a folder (used for bulk upload)."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    results = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".pdf"):
            results.append(ingest_pdf(os.path.join(folder, fname), source_name=fname))
    return results


# ------------------------------------------------------------
# QUERYING THE DATABASE
# ------------------------------------------------------------
def query(question: str, k: int = 4, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Searches for the most relevant chunks (k=4 by default)
    that semantically match the user's question.
    """
    if not question.strip():
        raise ValueError("Empty question.")
    col = get_collection()

    # Perform semantic search
    res = col.query(
        query_texts=[question],
        n_results=max(1, k),
        where=where,  # Optional filter by document name
        include=["documents", "metadatas", "distances"],
    )

    # Organize results with text, metadata, and similarity score
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    items = []
    for d, m, dist in zip(docs, metas, dists):
        items.append({
            "text": d,
            "source": m.get("source"),
            "page": m.get("page"),
            "chunk": m.get("chunk"),
            "distance": float(dist),
        })
    return {"question": question, "results": items}

def build_context(results: Dict[str, Any]) -> str:
    """
    Combines the retrieved text chunks into a single context block.
    This is what we'll send to the LLM (Gemma) to answer the question.
    """
    lines = []
    for r in results.get("results", []):
        tag = f"[{r['source']} p.{r['page']} c.{r['chunk']}]"
        lines.append(f"{tag} {r['text']}")
    return "\n\n".join(lines)


# ------------------------------------------------------------
# COMMAND-LINE INTERFACE (for quick testing)
# ------------------------------------------------------------
# Allows running this file directly from the terminal:
#   python rag.py ingest ./data   -> adds PDFs to the DB
#   python rag.py ask "question"  -> searches for relevant text
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        target = sys.argv[2]
        if os.path.isdir(target):
            print(ingest_folder(target))
        else:
            print(ingest_pdf(target))

    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        q = " ".join(sys.argv[2:])
        res = query(q, k=4)
        print(f"Q: {q}\n")
        for i, r in enumerate(res["results"], 1):
            print(f"{i}) {r['source']} (p.{r['page']} c.{r['chunk']}), dist={r['distance']:.4f}")
            print(f"   {r['text'][:250]}...\n")
        print("--- Context ---\n")
        print(build_context(res))

    else:
        print("Usage:\n  python rag.py ingest <pdf_file_or_folder>\n  python rag.py ask \"your question\"")


