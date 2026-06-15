# backend/rag/ingest.py

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "backend/rag/chroma_store")
DOCS_PATH = os.getenv("DOCS_PATH", "docs")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONFIGURED_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

COLLECTION_NAME = "datacore_pdf_chunks"
FALLBACK_EMBEDDING_MODEL = "hash-token-384"
EMBEDDING_DIM = 384

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_embedding_model_name(model_name: str) -> str:
    candidate = str(model_name or "").strip()

    if not candidate:
        return DEFAULT_EMBEDDING_MODEL

    if candidate.startswith("sentence-transformers/"):
        return candidate

    candidate_path = Path(candidate)
    if candidate_path.is_absolute() or candidate_path.exists():
        return candidate

    return DEFAULT_EMBEDDING_MODEL


EMBEDDING_MODEL = resolve_embedding_model_name(CONFIGURED_EMBEDDING_MODEL)


def build_chroma_client(chroma_path: Path):
    try:
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except Exception:
        return chromadb.PersistentClient(path=str(chroma_path))


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)


def _hash_embed_text(text: str, dimension: int = EMBEDDING_DIM) -> List[float]:
    vector = [0.0] * dimension
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())

    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector

    return [round(value / norm, 8) for value in vector]


def current_embedding_label() -> str:
    try:
        get_embedding_model()
        return EMBEDDING_MODEL
    except Exception:
        return FALLBACK_EMBEDDING_MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    except Exception:
        return [_hash_embed_text(text) for text in texts]


def get_collection():
    chroma_path = resolve_path(CHROMA_DB_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = build_chroma_client(chroma_path)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    words = text.split()

    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def make_chunk_id(
    filename: str,
    page_number: int,
    chunk_index: int,
    chunk_text: str,
) -> str:
    raw = f"{filename}:page-{page_number}:chunk-{chunk_index}:{chunk_text[:100]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{filename}:p{page_number}:c{chunk_index}:{digest}"


def extract_pdf_chunks(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    filename = pdf_path.name

    extracted_chunks: List[Dict[str, Any]] = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        page_text = clean_text(page_text)

        if not page_text:
            continue

        chunks = chunk_text(page_text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(
                filename=filename,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_text=chunk,
            )

            extracted_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "source_file": filename,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "source": f"{filename} page {page_number} chunk {chunk_index}",
                    },
                }
            )

    return extracted_chunks


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    pdf_file = resolve_path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")

    chunks = extract_pdf_chunks(pdf_file)

    if not chunks:
        return {
            "filename": pdf_file.name,
            "chunks_ingested": 0,
            "message": "No readable text found in PDF.",
            "source": pdf_file.name,
        }

    collection = get_collection()

    try:
        collection.delete(where={"filename": pdf_file.name})
    except Exception:
        pass

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {
        "filename": pdf_file.name,
        "chunks_ingested": len(chunks),
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "embedding_model": current_embedding_label(),
        "vector_store": "ChromaDB",
        "collection": COLLECTION_NAME,
        "source": pdf_file.name,
    }


def ingest_docs_folder(docs_path: str = DOCS_PATH) -> Dict[str, Any]:
    docs_dir = resolve_path(docs_path)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("*.pdf"))

    results = []

    for pdf_file in pdf_files:
        result = ingest_pdf(str(pdf_file))
        results.append(result)

    return {
        "docs_path": str(docs_dir),
        "pdf_count": len(pdf_files),
        "results": results,
    }


if __name__ == "__main__":
    result = ingest_docs_folder()
    print(result)
