# backend/rag/vector_store.py

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "backend/rag/chroma_store")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONFIGURED_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

COLLECTION_NAME = "datacore_pdf_chunks"
FALLBACK_EMBEDDING_MODEL = "hash-token-384"
EMBEDDING_DIM = 384


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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def get_chroma_client():
    chroma_path = resolve_path(CHROMA_DB_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    return build_chroma_client(chroma_path)


def get_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def clean_text(text: str, max_chars: int = 5000) -> str:
    text = " ".join(str(text or "").split())

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."

    return text


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []

    try:
        model = get_embedding_model()
        embeddings = model.encode(list(texts), show_progress_bar=False)
        return embeddings.tolist()
    except Exception:
        return [_hash_embed_text(text) for text in texts]


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    clean_metadata: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = str(value)

    return clean_metadata


def build_source_label(metadata: Dict[str, Any]) -> str:
    filename = (
        metadata.get("filename")
        or metadata.get("source_file")
        or metadata.get("source")
        or "unknown_pdf"
    )

    page = metadata.get("page")
    chunk_index = metadata.get("chunk_index")

    parts = [str(filename)]

    if page is not None:
        parts.append(f"page {page}")

    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")

    return " ".join(parts)


def distance_to_score(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None

    try:
        return round(1 - float(distance), 4)
    except Exception:
        return None


def add_chunks(
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Stores chunk documents and embeddings in ChromaDB.

    Used by backend/rag/ingest.py after PDF chunking.
    """

    if not ids:
        return {
            "stored": 0,
            "message": "No chunks provided.",
            "collection": COLLECTION_NAME,
        }

    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("ids, documents, and metadatas must have the same length.")

    documents = [clean_text(doc, max_chars=5000) for doc in documents]
    metadatas = [normalize_metadata(meta) for meta in metadatas]

    embeddings = embed_texts(documents)
    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {
        "stored": len(ids),
        "embedding_model": current_embedding_label(),
        "vector_store": "ChromaDB",
        "collection": COLLECTION_NAME,
    }


def delete_chunks_by_filename(filename: str) -> Dict[str, Any]:
    """
    Deletes existing chunks for one PDF before re-ingesting it.
    """

    filename = str(filename or "").strip()

    if not filename:
        raise ValueError("filename cannot be empty.")

    collection = get_collection()

    try:
        collection.delete(where={"filename": filename})
        deleted = True
    except Exception:
        deleted = False

    return {
        "filename": filename,
        "deleted": deleted,
        "collection": COLLECTION_NAME,
    }


def query_chunks(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Retrieves top-k relevant chunks from ChromaDB.

    Returns source filename, page number, chunk index, chunk id, score, and text.
    """

    query = str(query or "").strip()

    if not query:
        raise ValueError("query cannot be empty.")

    top_k = max(1, min(int(top_k), 5))

    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "query": query,
            "top_k": top_k,
            "source": "PDF vector store",
            "results": [],
            "message": "No PDF chunks found. Run backend/rag/ingest.py first.",
        }

    query_embedding = embed_texts([query])[0]

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    ids = response.get("ids", [[]])[0]

    results: List[Dict[str, Any]] = []

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        chunk_id = ids[index] if index < len(ids) else metadata.get("chunk_id")

        filename = (
            metadata.get("filename")
            or metadata.get("source_file")
            or "unknown_pdf"
        )

        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        source_label = build_source_label(metadata)

        results.append(
            {
                "text": clean_text(document),
                "filename": filename,
                "page": page,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "score": distance_to_score(distance),
                "source": source_label,
            }
        )

    return {
        "query": query,
        "top_k": top_k,
        "source": "PDF vector store",
        "results": results,
    }


def count_chunks() -> Dict[str, Any]:
    collection = get_collection()

    return {
        "collection": COLLECTION_NAME,
        "count": collection.count(),
        "vector_store": "ChromaDB",
        "path": str(resolve_path(CHROMA_DB_PATH)),
    }


def list_source_files() -> Dict[str, Any]:
    """
    Lists unique PDF files currently stored in the vector database.
    Useful for debugging ingestion.
    """

    collection = get_collection()

    if collection.count() == 0:
        return {
            "collection": COLLECTION_NAME,
            "files": [],
            "count": 0,
        }

    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    files = []

    for metadata in metadatas:
        filename = (
            metadata.get("filename")
            or metadata.get("source_file")
            or metadata.get("source")
        )

        if filename and filename not in files:
            files.append(filename)

    return {
        "collection": COLLECTION_NAME,
        "files": sorted(files),
        "count": len(files),
    }


def reset_vector_store(confirm: bool = False) -> Dict[str, Any]:
    """
    Deletes and recreates the ChromaDB collection.

    Use only during development.
    """

    if not confirm:
        raise ValueError("Pass confirm=True to reset the vector store.")

    client = get_chroma_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    get_collection.cache_clear() if hasattr(get_collection, "cache_clear") else None

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return {
        "message": "Vector store reset successfully.",
        "collection": collection.name,
        "path": str(resolve_path(CHROMA_DB_PATH)),
    }


class VectorStore:
    """
    Assessment-friendly wrapper used by rag_tool.py and ingest.py.
    """

    def get_collection(self):
        return get_collection()

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return add_chunks(ids=ids, documents=documents, metadatas=metadatas)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return add_chunks(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        response = query_chunks(query=query, top_k=limit)
        hits: List[Dict[str, Any]] = []

        for result in response.get("results", []):
            score = result.get("score")
            hits.append(
                {
                    "id": str(result.get("chunk_id") or result.get("source") or "unknown_chunk"),
                    "content": result.get("text", ""),
                    "source": result.get("source", "PDF vector store"),
                    "filename": result.get("filename"),
                    "page": result.get("page"),
                    "chunk_index": result.get("chunk_index"),
                    "score": score,
                    "distance": None if score is None else round(1 - float(score), 4),
                }
            )

        return hits

    def delete_by_filename(self, filename: str) -> Dict[str, Any]:
        return delete_chunks_by_filename(filename)

    def count(self) -> Dict[str, Any]:
        return count_chunks()

    def list_source_files(self) -> Dict[str, Any]:
        return list_source_files()

    def reset(self, confirm: bool = False) -> Dict[str, Any]:
        return reset_vector_store(confirm=confirm)


if __name__ == "__main__":
    print("Vector store status:")
    print(count_chunks())
    print(list_source_files())
