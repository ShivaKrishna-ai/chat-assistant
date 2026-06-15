# =========================================================
# SECTION 01: IMPORTS
# Purpose:
# - Shared Pydantic models for settings, API payloads, and tool contracts.
# =========================================================

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =========================================================
# SECTION 02: SETTINGS MODEL
# Purpose:
# - Load environment variables from .env.
# - Keep backend configuration in one place.
# =========================================================

class AppSettings(BaseSettings):
    DATABASE_URL: str = "sqlite:///backend/db/datacore_telugu_movies.db"

    CHROMA_DB_PATH: str = "backend/rag/chroma_store"
    DOCS_PATH: str = "docs"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_PROVIDER: str = "auto"
    CHAT_MODEL: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-0"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    APP_ENV: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# =========================================================
# SECTION 03: CHAT REQUEST / RESPONSE MODELS
# Purpose:
# - Validate /chat requests and responses.
# - Keep chat history payloads consistent.
# =========================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User question for the analytics assistant.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Session ID used for chat history.",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of RAG chunks to retrieve.",
    )

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty.")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    session_id: str
    mode: str = "unknown"
    notice: Optional[str] = None


class ChatHistoryItem(BaseModel):
    user: str
    assistant: str
    sources: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    mode: Optional[str] = None
    notice: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    count: int
    history: List[ChatHistoryItem]


# =========================================================
# SECTION 04: API RESPONSE MODELS
# Purpose:
# - Response rows for analytics endpoints and PDF ingestion.
# =========================================================

class MovieAnalyticsRow(BaseModel):
    movie_id: str
    title: str
    genre: str
    release_date: str
    director: str
    budget_usd: float
    rating: float
    total_sessions: int
    watch_hours: float
    avg_completion_pct: float


class RegionAnalyticsRow(BaseModel):
    region_id: str
    city: str
    country: str
    total_views: int
    avg_rating: float
    growth_pct: float
    report_month: str


class GenreTrendRow(BaseModel):
    genre: str
    total_sessions: int
    watch_hours: float
    avg_completion_pct: float
    avg_rating: float


class CampaignPerformanceRow(BaseModel):
    movie_id: str
    title: str
    channel: str
    campaign_month: str
    spend_usd: float
    impressions: int
    clicks: int
    ctr_pct: float
    clicks_per_1000_spend: float


class PDFIngestResponse(BaseModel):
    message: str
    filename: str
    path: str
    result: Dict[str, Any] = Field(default_factory=dict)
    source: str = "PDF vector store"


class DocumentChunk(BaseModel):
    text: str
    source: str
    filename: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None


# =========================================================
# SECTION 05: TOOL ARGUMENT MODELS
# Purpose:
# - Validate tool inputs before they reach SQL or RAG helpers.
# - Ensure the agent passes structured parameters only.
# =========================================================

class QueryMovieDataArgs(BaseModel):
    titles: Optional[List[str]] = Field(default=None)
    title: Optional[str] = Field(default=None)
    genre: Optional[str] = Field(default=None, min_length=2, max_length=50)
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None, ge=2000, le=2030)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value

        parts = value.split("-")
        if len(parts) != 3:
            raise ValueError("Date must be in YYYY-MM-DD format.")

        year, month, day = parts
        if not (year.isdigit() and month.isdigit() and day.isdigit()):
            raise ValueError("Date must be in YYYY-MM-DD format.")

        return value


class SearchDocumentsArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Search query cannot be empty.")
        return value


class RegionalStatsArgs(BaseModel):
    city: Optional[str] = Field(default=None, min_length=2, max_length=80)
    country: Optional[str] = Field(default=None, min_length=2, max_length=80)
    month: Optional[str] = Field(default=None)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value

        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError("Month must be in YYYY-MM format.")

        year, month = parts
        if not (year.isdigit() and month.isdigit()):
            raise ValueError("Month must be in YYYY-MM format.")

        month_number = int(month)
        if month_number < 1 or month_number > 12:
            raise ValueError("Month must be between 01 and 12.")

        return value


class GenreTrendArgs(BaseModel):
    genre: Optional[str] = Field(default=None, min_length=2, max_length=50)
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    month: Optional[str] = Field(default=None)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value

        parts = value.split("-")
        if len(parts) != 3:
            raise ValueError("Date must be in YYYY-MM-DD format.")

        year, month, day = parts
        if not (year.isdigit() and month.isdigit() and day.isdigit()):
            raise ValueError("Date must be in YYYY-MM-DD format.")

        return value

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value

        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError("Month must be in YYYY-MM format.")

        year, month = parts
        if not (year.isdigit() and month.isdigit()):
            raise ValueError("Month must be in YYYY-MM format.")

        month_number = int(month)
        if month_number < 1 or month_number > 12:
            raise ValueError("Month must be between 01 and 12.")

        return value


class GenreRatingByMonthArgs(BaseModel):
    genre: str = Field(..., min_length=2, max_length=50)
    year: Optional[int] = Field(default=None, ge=2000, le=2030)
    limit: int = Field(default=12, ge=1, le=24)

    @field_validator("genre")
    @classmethod
    def clean_genre(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("genre cannot be empty.")
        return value


class CampaignPerformanceArgs(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=120)
    channel: Optional[str] = Field(default=None, min_length=2, max_length=80)
    month: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None, ge=2000, le=2030)
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value

        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError("Month must be in YYYY-MM format.")

        year, month = parts
        if not (year.isdigit() and month.isdigit()):
            raise ValueError("Month must be in YYYY-MM format.")

        month_number = int(month)
        if month_number < 1 or month_number > 12:
            raise ValueError("Month must be between 01 and 12.")

        return value


class TopCampaignTitlesArgs(BaseModel):
    year: Optional[int] = Field(default=None, ge=2000, le=2030)
    limit: int = Field(default=10, ge=1, le=20)


# =========================================================
# SECTION 06: TOOL RESULT MODELS
# Purpose:
# - Standard agent/tool output payloads.
# =========================================================

class ToolResult(BaseModel):
    tool: str
    source: str
    data: Any
    error: Optional[str] = None


class AgentResult(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
