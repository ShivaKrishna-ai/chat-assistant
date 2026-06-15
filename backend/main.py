# backend/main.py

import asyncio
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    AppSettings,
    ChatRequest,
    ChatResponse,
    GenreTrendRow,
    GenreTrendArgs,
    HistoryResponse,
    MovieAnalyticsRow,
    PDFIngestResponse,
    QueryMovieDataArgs,
    RegionalStatsArgs,
    RegionAnalyticsRow,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

settings = AppSettings()


try:
    from backend.agent import get_agent_runtime_status, run_agent
except Exception:
    try:
        from agent import get_agent_runtime_status, run_agent
    except Exception:
        get_agent_runtime_status = None
        run_agent = None

try:
    from backend.tools.sql_tool import (
        get_genre_trends as get_genre_trends_tool,
        get_regional_stats as get_regional_stats_tool,
        query_movie_data as query_movie_data_tool,
    )
except Exception:
    from tools.sql_tool import (
        get_genre_trends as get_genre_trends_tool,
        get_regional_stats as get_regional_stats_tool,
        query_movie_data as query_movie_data_tool,
    )


app = FastAPI(
    title="DataCore Telugu Analytics Assistant",
    description="Internal analytics assistant using FastAPI, SQLite, RAG, and tool-based source attribution.",
    version="1.0.0",
)


allowed_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CHAT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


async def run_chat_request(request: ChatRequest) -> ChatResponse:
    if run_agent is None:
        raise HTTPException(
            status_code=501,
            detail="Agent is not configured. Create backend/agent.py with run_agent().",
        )

    session_history = CHAT_HISTORY.setdefault(request.session_id, [])

    try:
        result = await asyncio.to_thread(
            run_agent,
            message=request.message,
            session_id=request.session_id,
            chat_history=session_history[-10:],
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed while processing the message: {str(exc)}",
        )

    if inspect.isawaitable(result):
        result = await result

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Agent must return a dictionary with answer, sources, and tool_calls.",
        )

    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources", [])
    tool_calls = result.get("tool_calls", [])

    if not answer:
        raise HTTPException(
            status_code=500,
            detail="Agent returned an empty answer.",
        )

    if not isinstance(sources, list):
        sources = [str(sources)]

    if not isinstance(tool_calls, list):
        tool_calls = [str(tool_calls)]

    runtime_status = get_agent_runtime_status() if get_agent_runtime_status else {}
    mode = str(runtime_status.get("mode") or "unknown")
    notice_value = runtime_status.get("notice")
    notice = str(notice_value) if notice_value else None

    session_history.append(
        {
            "user": request.message,
            "assistant": answer,
            "sources": sources,
            "tool_calls": tool_calls,
            "mode": mode,
            "notice": notice,
        }
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        tool_calls=tool_calls,
        session_id=request.session_id,
        mode=mode,
        notice=notice,
    )


def build_sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text_for_streaming(text: str, words_per_chunk: int = 8) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]

    chunks: List[str] = []
    for index in range(0, len(words), words_per_chunk):
        segment = " ".join(words[index : index + words_per_chunk]).strip()
        if segment:
            suffix = " " if index + words_per_chunk < len(words) else ""
            chunks.append(segment + suffix)

    return chunks


async def stream_chat_events(request: ChatRequest) -> AsyncIterator[str]:
    yield build_sse_event("status", {"message": "Thinking and calling tools..."})
    await asyncio.sleep(0)

    try:
        response = await run_chat_request(request)
    except HTTPException as exc:
        yield build_sse_event(
            "error",
            {"detail": str(exc.detail), "status_code": exc.status_code},
        )
        return
    except Exception as exc:
        yield build_sse_event("error", {"detail": str(exc), "status_code": 500})
        return

    yield build_sse_event("status", {"message": "Streaming grounded answer..."})
    await asyncio.sleep(0)

    for chunk in chunk_text_for_streaming(response.answer):
        yield build_sse_event("chunk", {"delta": chunk})
        await asyncio.sleep(0.01)

    yield build_sse_event("done", response.model_dump())


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "DataCore Telugu Analytics Assistant",
        "environment": settings.APP_ENV,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_chat_request(request)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/data/movies", response_model=List[MovieAnalyticsRow])
def get_top_movies(
    genre: Optional[str] = Query(default=None, min_length=2, max_length=50),
    year: Optional[int] = Query(default=None, ge=2000, le=2030),
    limit: int = Query(default=10, ge=1, le=50),
) -> List[MovieAnalyticsRow]:
    try:
        tool_args = QueryMovieDataArgs(
            genre=genre,
            year=year,
            limit=min(limit, 20),
        ).model_dump(exclude_none=True)
        result = query_movie_data_tool(**tool_args)
        return result.get("rows", [])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Movie analytics query failed: {str(exc)}",
        )


@app.get("/data/regions", response_model=List[RegionAnalyticsRow])
def get_regions(
    month: Optional[str] = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}$",
        description="Month format must be YYYY-MM, example: 2025-05",
    ),
    city: Optional[str] = Query(default=None, min_length=2, max_length=80),
    country: Optional[str] = Query(default=None, min_length=2, max_length=80),
    limit: int = Query(default=20, ge=1, le=100),
) -> List[RegionAnalyticsRow]:
    try:
        tool_args = RegionalStatsArgs(
            city=city,
            country=country,
            month=month,
            limit=min(limit, 20),
        ).model_dump(exclude_none=True)
        result = get_regional_stats_tool(**tool_args)
        return result.get("rows", [])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Regional analytics query failed: {str(exc)}",
        )


@app.get("/data/genre-trends", response_model=List[GenreTrendRow])
def get_genre_trends(
    year: Optional[int] = Query(default=None, ge=2000, le=2030),
    month: Optional[str] = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}$",
        description="Month format must be YYYY-MM, example: 2025-05",
    ),
    limit: int = Query(default=10, ge=1, le=50),
) -> List[GenreTrendRow]:
    try:
        tool_args = GenreTrendArgs(
            start_date=f"{year}-01-01" if year is not None and month is None else None,
            end_date=f"{year}-12-31" if year is not None and month is None else None,
            month=month,
            limit=min(limit, 20),
        ).model_dump(exclude_none=True)
        result = get_genre_trends_tool(**tool_args)
        rows: List[Dict[str, Any]] = []
        for row in result.get("rows", []):
            if not isinstance(row, dict):
                continue

            normalized_row = dict(row)
            normalized_row["avg_rating"] = normalized_row.get(
                "avg_rating",
                normalized_row.get("avg_review_rating", 0),
            )
            rows.append(normalized_row)

        return rows
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Genre trends query failed: {str(exc)}",
        )


@app.post("/ingest/pdf")
async def upload_and_ingest_pdf(file: UploadFile = File(...)) -> PDFIngestResponse:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    safe_filename = Path(file.filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PDF file.",
        )

    docs_dir = PROJECT_ROOT / settings.DOCS_PATH
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved_path = docs_dir / safe_filename

    try:
        try:
            from backend.rag.ingest import ingest_pdf
        except Exception:
            from rag.ingest import ingest_pdf

        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_pdf(str(saved_path))

        if inspect.isawaitable(result):
            result = await result

        return {
            "message": "PDF uploaded and ingested successfully.",
            "filename": safe_filename,
            "path": str(saved_path),
            "result": result,
            "source": "PDF vector store",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF ingestion failed: {str(exc)}",
        )


@app.get("/history", response_model=HistoryResponse)
def get_history(
    session_id: str = Query(default="default", min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50),
) -> HistoryResponse:
    history = CHAT_HISTORY.get(session_id, [])

    return {
        "session_id": session_id,
        "count": min(len(history), limit),
        "history": history[-limit:],
    }
