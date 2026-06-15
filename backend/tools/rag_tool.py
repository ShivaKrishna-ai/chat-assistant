# =========================================================
# SECTION 01: IMPORTS
# Purpose:
# - Document-only retrieval helpers for PDF-backed answers.
# =========================================================

import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pypdf import PdfReader

from ..rag.vector_store import VectorStore


load_dotenv()

# =========================================================
# SECTION 02: ENVIRONMENT AND PATH SETUP
# Purpose:
# - Resolve docs/ and project-relative paths.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DOCS_PATH = os.getenv("DOCS_PATH", "docs")

# =========================================================
# SECTION 03: TEXT CLEANING HELPERS
# Purpose:
# - Normalize queries and page text for fallback matching.
# =========================================================


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _score_text(query_tokens: set[str], text: str) -> int:
    haystack = _tokenize(text)
    return sum(token in haystack for token in query_tokens)


# =========================================================
# SECTION 04: TARGET DOCUMENT DETECTION
# Purpose:
# - Route report-oriented questions toward the correct PDF.
# =========================================================

def _extract_target_filename(query: str) -> str | None:
    lowered = query.lower()
    if "campaign performance report" in lowered:
        return "campaign_performance_2025.pdf"
    if "quarterly report" in lowered:
        return "quarterly_report_q1_2025.pdf"
    if "stellar run" in lowered and ("trending" in lowered or "campaign" in lowered):
        return "campaign_performance_2025.pdf"
    if any(marker in lowered for marker in ["key kpi", "key kpis", "q1 2025 kpi", "q1 kpi"]):
        return "quarterly_report_q1_2025.pdf"
    return None


# =========================================================
# SECTION 07: RERANKING
# Purpose:
# - Prefer the most relevant section/page after retrieval.
# =========================================================

def _rerank_hits(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lowered = query.lower()
    genre_trends_request = "genre trend" in lowered or "genre trends" in lowered
    audience_growth_request = "audience growth" in lowered or "growth summary" in lowered
    operational_request = "operational recommendations" in lowered
    top_titles_request = "top performing titles" in lowered or "top titles" in lowered
    top_campaign_titles_request = "top campaign titles by spend" in lowered
    specific_section_request = any(
        [
            genre_trends_request,
            audience_growth_request,
            operational_request,
            top_titles_request,
            top_campaign_titles_request,
        ]
    )

    def bonus(hit: Dict[str, Any]) -> int:
        content = _clean_text(hit.get("content", "")).lower()
        page = int(hit.get("page", 0) or 0)
        score = 0

        if "executive summary" in lowered and "executive summary" in content:
            score += 40
        if genre_trends_request and "genre trends" in content:
            score += 60
        if audience_growth_request and "audience growth summary" in content:
            score += 60
        if operational_request and "operational recommendations" in content:
            score += 60
        if top_titles_request and "top performing titles by watch hours" in content:
            score += 60
        if top_campaign_titles_request and "top campaign titles by spend" in content:
            score += 70
        if not specific_section_request and any(marker in lowered for marker in ["overview", "what does", "what is", "about"]) and "executive summary" in content:
            score += 25
        if any(marker in lowered for marker in ["key kpi", "key kpis", "metrics"]) and "key kpis" in content:
            score += 50
        if not audience_growth_request and not operational_request and "quarterly report" in lowered and page == 1:
            score += 20
        if "campaign performance report" in lowered and page == 1:
            score += 20
        if top_campaign_titles_request and hit.get("filename") == "campaign_performance_2025.pdf" and page == 1:
            score += 25
        if "stellar run" in lowered and ("trending" in lowered or "campaign" in lowered):
            if hit.get("filename") == "campaign_performance_2025.pdf":
                score += 35
            if "why stellar run is trending" in content or "executive summary" in content:
                score += 15
        if "audience growth" in lowered and "audience growth summary" in content:
            score += 30

        return score

    return sorted(
        hits,
        key=lambda item: (bonus(item), -float(item.get("distance", 9999))),
        reverse=True,
    )


# =========================================================
# SECTION 06: PDF FALLBACK SEARCH
# Purpose:
# - Scan raw PDFs when Chroma does not return usable hits.
# =========================================================

def _search_pdf_fallback(query: str, top_k: int) -> List[Dict[str, Any]]:
    docs_dir = _resolve_path(DOCS_PATH)
    if not docs_dir.exists():
        return []

    query_tokens = _tokenize(query)
    target_filename = _extract_target_filename(query)
    hits: List[Dict[str, Any]] = []

    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        if target_filename and pdf_path.name != target_filename:
            continue

        reader = PdfReader(str(pdf_path))

        for page_index, page in enumerate(reader.pages):
            try:
                text = _clean_text(page.extract_text() or "")
            except Exception:
                text = ""

            if not text:
                continue

            score = _score_text(query_tokens, text)
            if score <= 0:
                continue

            preview = text[:5000]
            hits.append(
                {
                    "id": f"{pdf_path.name}-page-{page_index + 1}",
                    "content": preview,
                    "source": f"{pdf_path.name} page {page_index + 1}",
                    "filename": pdf_path.name,
                    "page": page_index + 1,
                    "chunk_index": 0,
                    "score": score,
                }
            )

    hits.sort(key=lambda item: item.get("score", 0), reverse=True)
    return hits[:top_k]


# =========================================================
# SECTION 08: CONFIDENCE CHECK
# Purpose:
# - Reject weak matches so the agent does not summarize noise.
# =========================================================

def _passes_confidence_check(hits: List[Dict[str, Any]]) -> bool:
    if not hits:
        return False

    top_hit = hits[0]
    if "score" in top_hit:
        score = float(top_hit.get("score", 0) or 0)
        if score <= 1:
            return score >= 0.12
        return score >= 2

    return bool(_clean_text(top_hit.get("content", "")))


# =========================================================
# SECTION 09: PUBLIC search_documents()
# Purpose:
# - Main RAG entrypoint used by agent.py.
# =========================================================

def search_documents(query: str, top_k: int = 4, limit: int | None = None) -> List[Dict[str, Any]]:
    query = _clean_text(query)
    max_results = limit if limit is not None else top_k
    max_results = max(1, min(5, int(max_results)))
    target_filename = _extract_target_filename(query)

    if not query:
        return []

    try:
        chroma_hits = VectorStore().search(query=query, limit=max_results)
    except Exception:
        chroma_hits = []

    if target_filename and chroma_hits:
        chroma_hits = [hit for hit in chroma_hits if hit.get("filename") == target_filename]

    if chroma_hits:
        ranked_hits = _rerank_hits(query, chroma_hits)
        return ranked_hits if _passes_confidence_check(ranked_hits) else []

    fallback_hits = _search_pdf_fallback(query, max_results)
    ranked_fallback_hits = _rerank_hits(query, fallback_hits)
    return ranked_fallback_hits if _passes_confidence_check(ranked_fallback_hits) else []
