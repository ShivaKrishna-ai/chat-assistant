# =========================================================
# SECTION 01: IMPORTS
# Purpose:
# - Database-only analytics helpers used by the agent and API routes.
# =========================================================

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# =========================================================
# SECTION 02: ENVIRONMENT AND PATH SETUP
# Purpose:
# - Resolve the SQLite database location from DATABASE_URL.
# =========================================================

KNOWN_GENRES = [
    "Action",
    "Comedy",
    "Crime Thriller",
    "Drama",
    "Historical Drama",
    "Romance",
    "Sci-Fi Thriller",
    "Thriller",
]

BASE_DIR = Path(__file__).resolve().parent        # backend/tools/
BACKEND_DIR = BASE_DIR.parent                     # backend/
PROJECT_ROOT = BACKEND_DIR.parent                 # project root/

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///backend/db/datacore_telugu_movies.db",
)

# =========================================================
# SECTION 03: DATABASE CONNECTION
# Purpose:
# - Open SQLite with row_factory enabled for dict-friendly responses.
# =========================================================

def get_sqlite_path(database_url: str = DATABASE_URL) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only SQLite DATABASE_URL is supported.")

    raw_path = database_url.replace("sqlite:///", "", 1)
    db_path = Path(raw_path)

    if db_path.is_absolute():
        return db_path

    possible_paths = [
        PROJECT_ROOT / db_path,
        Path.cwd() / db_path,
        BACKEND_DIR / db_path,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return PROJECT_ROOT / db_path


def get_connection() -> sqlite3.Connection:
    db_path = get_sqlite_path()

    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path}. Run backend/db/seed.py first."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# SECTION 04: VALIDATION HELPERS
# Purpose:
# - Clean text, clamp limits, and validate dates/months.
# =========================================================

def rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def clamp_limit(limit: Any, default: int = 10, minimum: int = 1, maximum: int = 20) -> int:
    try:
        value = int(limit)
        return max(minimum, min(maximum, value))
    except Exception:
        return default


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def validate_date(value: Optional[str], field_name: str) -> Optional[str]:
    value = clean_text(value)

    if value is None:
        return None

    parts = value.split("-")

    if len(parts) != 3:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.")

    year, month, day = parts

    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format.")

    if not (1 <= int(month) <= 12):
        raise ValueError(f"{field_name} has invalid month.")

    if not (1 <= int(day) <= 31):
        raise ValueError(f"{field_name} has invalid day.")

    return value


def validate_month(value: Optional[str]) -> Optional[str]:
    value = clean_text(value)

    if value is None:
        return None

    parts = value.split("-")

    if len(parts) != 2:
        raise ValueError("month must be in YYYY-MM format.")

    year, month = parts

    if not (year.isdigit() and month.isdigit()):
        raise ValueError("month must be in YYYY-MM format.")

    if not (1 <= int(month) <= 12):
        raise ValueError("month must be between 01 and 12.")

    return value


# =========================================================
# SECTION 05: RESULT FORMAT HELPERS
# Purpose:
# - Keep tool responses uniform and source-labeled.
# =========================================================

def build_tool_response(
    tool_name: str,
    source: str,
    filters: Dict[str, Any],
    rows: List[sqlite3.Row],
) -> Dict[str, Any]:
    return {
        "tool": tool_name,
        "source": source,
        "filters": filters,
        "rows": rows_to_dicts(rows),
    }


# =========================================================
# SECTION 06: TOOL - query_movie_data()
# Purpose:
# - Title performance, watch hours, completion, reviews, and comparisons.
# =========================================================

def query_movie_data(
    titles: Optional[List[str]] = None,
    title: Optional[str] = None,
    genre: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Tool: query_movie_data

    Executes safe parameterized SQL against:
    - movies
    - watch_activity
    - reviews

    The LLM only sends parameters.
    The LLM never sends raw SQL.
    """

    limit = clamp_limit(limit)
    genre = clean_text(genre)
    start_date = validate_date(start_date, "start_date")
    end_date = validate_date(end_date, "end_date")

    if year is not None:
        try:
            year = int(year)
        except Exception:
            raise ValueError("year must be an integer.")

        if year < 2000 or year > 2030:
            raise ValueError("year must be between 2000 and 2030.")

    title_filters: List[str] = []

    if title:
        title_filters.append(str(title).strip())

    if titles:
        for item in titles:
            item = str(item).strip()
            if item:
                title_filters.append(item)

    title_filters = list(dict.fromkeys(title_filters))

    activity_where = ["1 = 1"]
    activity_params: List[Any] = []

    review_where = ["1 = 1"]
    review_params: List[Any] = []

    movie_where = ["1 = 1"]
    movie_params: List[Any] = []

    if start_date:
        activity_where.append("watch_date >= ?")
        activity_params.append(start_date)

        review_where.append("review_date >= ?")
        review_params.append(start_date)

    if end_date:
        activity_where.append("watch_date <= ?")
        activity_params.append(end_date)

        review_where.append("review_date <= ?")
        review_params.append(end_date)

    if year:
        activity_where.append("strftime('%Y', watch_date) = ?")
        activity_params.append(str(year))

        review_where.append("strftime('%Y', review_date) = ?")
        review_params.append(str(year))

    if genre:
        movie_where.append("LOWER(m.genre) = LOWER(?)")
        movie_params.append(genre)

    if title_filters:
        placeholders = ",".join(["?"] * len(title_filters))
        movie_where.append(f"LOWER(m.title) IN ({placeholders})")
        movie_params.extend([item.lower() for item in title_filters])

    query = f"""
        WITH activity_agg AS (
            SELECT
                movie_id,
                COUNT(activity_id) AS total_sessions,
                ROUND(SUM(watch_duration_min) / 60.0, 2) AS watch_hours,
                ROUND(AVG(completion_pct), 2) AS avg_completion_pct
            FROM watch_activity
            WHERE {" AND ".join(activity_where)}
            GROUP BY movie_id
        ),
        review_agg AS (
            SELECT
                movie_id,
                COUNT(review_id) AS total_reviews,
                ROUND(AVG(rating_stars), 2) AS avg_review_rating,
                SUM(CASE WHEN LOWER(sentiment_label) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
                SUM(CASE WHEN LOWER(sentiment_label) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
                SUM(CASE WHEN LOWER(sentiment_label) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews
            FROM reviews
            WHERE {" AND ".join(review_where)}
            GROUP BY movie_id
        )
        SELECT
            m.movie_id,
            m.title,
            m.genre,
            m.release_date,
            m.director,
            m.budget_usd,
            m.rating,
            COALESCE(a.total_sessions, 0) AS total_sessions,
            COALESCE(a.watch_hours, 0) AS watch_hours,
            COALESCE(a.avg_completion_pct, 0) AS avg_completion_pct,
            COALESCE(r.total_reviews, 0) AS total_reviews,
            COALESCE(r.avg_review_rating, 0) AS avg_review_rating,
            COALESCE(r.positive_reviews, 0) AS positive_reviews,
            COALESCE(r.neutral_reviews, 0) AS neutral_reviews,
            COALESCE(r.negative_reviews, 0) AS negative_reviews
        FROM movies m
        LEFT JOIN activity_agg a
            ON m.movie_id = a.movie_id
        LEFT JOIN review_agg r
            ON m.movie_id = r.movie_id
        WHERE {" AND ".join(movie_where)}
        ORDER BY
            watch_hours DESC,
            avg_completion_pct DESC,
            avg_review_rating DESC
        LIMIT ?
    """

    params = activity_params + review_params + movie_params + [limit]

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="query_movie_data",
        source="SQL: movies, watch_activity, reviews",
        filters={
            "titles": title_filters,
            "genre": genre,
            "start_date": start_date,
            "end_date": end_date,
            "year": year,
            "limit": limit,
        },
        rows=rows,
    )


# =========================================================
# SECTION 07: TOOL - get_regional_stats()
# Purpose:
# - City and country growth, views, and average rating.
# =========================================================

def get_regional_stats(
    city: Optional[str] = None,
    country: Optional[str] = None,
    month: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Tool: get_regional_stats

    Executes safe parameterized SQL against:
    - regional_performance
    """

    limit = clamp_limit(limit)
    city = clean_text(city)
    country = clean_text(country)
    month = validate_month(month)

    where = ["1 = 1"]
    params: List[Any] = []

    if city:
        where.append("LOWER(city) = LOWER(?)")
        params.append(city)

    if country:
        where.append("LOWER(country) = LOWER(?)")
        params.append(country)

    if month:
        where.append("report_month = ?")
        params.append(month)
    else:
        where.append("report_month = (SELECT MAX(report_month) FROM regional_performance)")

    query = f"""
        SELECT
            region_id,
            city,
            country,
            total_views,
            avg_rating,
            growth_pct,
            report_month
        FROM regional_performance
        WHERE {" AND ".join(where)}
        ORDER BY
            growth_pct DESC,
            total_views DESC,
            avg_rating DESC
        LIMIT ?
    """

    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="get_regional_stats",
        source="SQL: regional_performance",
        filters={
            "city": city,
            "country": country,
            "month": month or "latest_available_month",
            "limit": limit,
        },
        rows=rows,
    )


# =========================================================
# SECTION 08: TOOL - get_genre_trends()
# Purpose:
# - Genre performance, comedy weakness, and chart support.
# =========================================================

def get_genre_trends(
    genre: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    month: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Tool: get_genre_trends

    Aggregates watch activity and review metrics by genre.
    Useful for charting and weak genre performance analysis.
    """

    limit = clamp_limit(limit)
    genre = clean_text(genre)
    start_date = validate_date(start_date, "start_date")
    end_date = validate_date(end_date, "end_date")
    month = validate_month(month)

    activity_where = ["1 = 1"]
    activity_params: List[Any] = []

    review_where = ["1 = 1"]
    review_params: List[Any] = []

    movie_where = ["1 = 1"]
    movie_params: List[Any] = []

    if start_date:
        activity_where.append("watch_date >= ?")
        activity_params.append(start_date)

        review_where.append("review_date >= ?")
        review_params.append(start_date)

    if end_date:
        activity_where.append("watch_date <= ?")
        activity_params.append(end_date)

        review_where.append("review_date <= ?")
        review_params.append(end_date)

    if month:
        activity_where.append("strftime('%Y-%m', watch_date) = ?")
        activity_params.append(month)

        review_where.append("strftime('%Y-%m', review_date) = ?")
        review_params.append(month)

    if genre:
        movie_where.append("LOWER(m.genre) = LOWER(?)")
        movie_params.append(genre)

    query = f"""
        WITH activity_agg AS (
            SELECT
                movie_id,
                COUNT(activity_id) AS total_sessions,
                ROUND(SUM(watch_duration_min) / 60.0, 2) AS watch_hours,
                ROUND(AVG(completion_pct), 2) AS avg_completion_pct
            FROM watch_activity
            WHERE {" AND ".join(activity_where)}
            GROUP BY movie_id
        ),
        review_agg AS (
            SELECT
                movie_id,
                COUNT(review_id) AS total_reviews,
                ROUND(AVG(rating_stars), 2) AS avg_review_rating,
                SUM(CASE WHEN LOWER(sentiment_label) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
                SUM(CASE WHEN LOWER(sentiment_label) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
                SUM(CASE WHEN LOWER(sentiment_label) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews
            FROM reviews
            WHERE {" AND ".join(review_where)}
            GROUP BY movie_id
        )
        SELECT
            m.genre,
            COUNT(DISTINCT m.movie_id) AS title_count,
            COALESCE(SUM(a.total_sessions), 0) AS total_sessions,
            ROUND(COALESCE(SUM(a.watch_hours), 0), 2) AS watch_hours,
            ROUND(COALESCE(AVG(a.avg_completion_pct), 0), 2) AS avg_completion_pct,
            COALESCE(SUM(r.total_reviews), 0) AS total_reviews,
            ROUND(COALESCE(AVG(r.avg_review_rating), 0), 2) AS avg_review_rating,
            COALESCE(SUM(r.positive_reviews), 0) AS positive_reviews,
            COALESCE(SUM(r.neutral_reviews), 0) AS neutral_reviews,
            COALESCE(SUM(r.negative_reviews), 0) AS negative_reviews
        FROM movies m
        LEFT JOIN activity_agg a
            ON m.movie_id = a.movie_id
        LEFT JOIN review_agg r
            ON m.movie_id = r.movie_id
        WHERE {" AND ".join(movie_where)}
        GROUP BY m.genre
        ORDER BY
            watch_hours DESC,
            avg_completion_pct DESC,
            avg_review_rating DESC
        LIMIT ?
    """

    params = activity_params + review_params + movie_params + [limit]

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="get_genre_trends",
        source="SQL: movies, watch_activity, reviews",
        filters={
            "genre": genre,
            "start_date": start_date,
            "end_date": end_date,
            "month": month,
            "limit": limit,
        },
        rows=rows,
    )


def get_genre_rating_by_month(
    genre: str,
    year: Optional[int] = None,
    limit: int = 12,
) -> Dict[str, Any]:
    """
    Tool: get_genre_rating_by_month

    Returns monthly review ratings for a genre so the agent can answer
    questions such as "In which month did Comedy have a 2.0 rating?"
    """

    genre = clean_text(genre)
    if not genre:
        raise ValueError("genre is required.")

    limit = clamp_limit(limit, default=12, minimum=1, maximum=24)

    params: List[Any] = [genre]
    where = ["LOWER(m.genre) = LOWER(?)"]

    if year is not None:
        try:
            year = int(year)
        except Exception:
            raise ValueError("year must be an integer.")

        if year < 2000 or year > 2030:
            raise ValueError("year must be between 2000 and 2030.")

        where.append("strftime('%Y', r.review_date) = ?")
        params.append(str(year))

    query = f"""
        SELECT
            strftime('%Y-%m', r.review_date) AS month,
            ROUND(AVG(r.rating_stars), 2) AS avg_review_rating,
            COUNT(r.review_id) AS total_reviews,
            SUM(CASE WHEN LOWER(r.sentiment_label) = 'positive' THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN LOWER(r.sentiment_label) = 'neutral' THEN 1 ELSE 0 END) AS neutral_reviews,
            SUM(CASE WHEN LOWER(r.sentiment_label) = 'negative' THEN 1 ELSE 0 END) AS negative_reviews
        FROM reviews r
        JOIN movies m
            ON m.movie_id = r.movie_id
        WHERE {" AND ".join(where)}
        GROUP BY strftime('%Y-%m', r.review_date)
        ORDER BY month ASC
        LIMIT ?
    """

    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="get_genre_rating_by_month",
        source="SQL: movies, reviews",
        filters={
            "genre": genre,
            "year": year,
            "limit": limit,
        },
        rows=rows,
    )


# =========================================================
# SECTION 09: TOOL - get_campaign_performance()
# Purpose:
# - Campaign spend, impressions, clicks, CTR, and ROI proxy.
# =========================================================

def get_campaign_performance(
    title: Optional[str] = None,
    channel: Optional[str] = None,
    month: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    limit = clamp_limit(limit)
    title = clean_text(title)
    channel = clean_text(channel)
    month = validate_month(month)

    if year is not None:
        try:
            year = int(year)
        except Exception:
            raise ValueError("year must be an integer.")

        if year < 2000 or year > 2030:
            raise ValueError("year must be between 2000 and 2030.")

    where = ["1 = 1"]
    params: List[Any] = []

    if title:
        where.append("LOWER(m.title) = LOWER(?)")
        params.append(title)

    if channel:
        where.append("LOWER(ms.channel) = LOWER(?)")
        params.append(channel)

    if month:
        where.append("ms.campaign_month = ?")
        params.append(month)

    if year is not None:
        where.append("strftime('%Y', ms.campaign_month || '-01') = ?")
        params.append(str(year))

    query = f"""
        SELECT
            m.movie_id,
            m.title,
            ms.channel,
            ms.campaign_month,
            ms.spend_usd,
            ms.impressions,
            ms.clicks,
            ROUND(
                CASE
                    WHEN ms.impressions = 0 THEN 0
                    ELSE (ms.clicks * 100.0) / ms.impressions
                END,
                2
            ) AS ctr_pct,
            ROUND(
                CASE
                    WHEN ms.spend_usd = 0 THEN 0
                    ELSE (ms.clicks * 1000.0) / ms.spend_usd
                END,
                2
            ) AS clicks_per_1000_spend
        FROM marketing_spend ms
        JOIN movies m
            ON m.movie_id = ms.movie_id
        WHERE {" AND ".join(where)}
        ORDER BY
            ms.spend_usd DESC,
            ms.clicks DESC,
            ctr_pct DESC
        LIMIT ?
    """

    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="get_campaign_performance",
        source="SQL: marketing_spend, movies",
        filters={
            "title": title,
            "channel": channel,
            "month": month,
            "year": year,
            "limit": limit,
        },
        rows=rows,
    )


# =========================================================
# SECTION 10: TOOL - get_top_campaign_titles()
# Purpose:
# - Aggregate campaign totals by title and rank by spend.
# =========================================================

def get_top_campaign_titles(
    year: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    limit = clamp_limit(limit)

    if year is not None:
        try:
            year = int(year)
        except Exception:
            raise ValueError("year must be an integer.")

        if year < 2000 or year > 2030:
            raise ValueError("year must be between 2000 and 2030.")

    where = ["1 = 1"]
    params: List[Any] = []

    if year is not None:
        where.append("strftime('%Y', ms.campaign_month || '-01') = ?")
        params.append(str(year))

    query = f"""
        SELECT
            m.movie_id,
            m.title,
            ROUND(SUM(ms.spend_usd), 2) AS total_spend_usd,
            SUM(ms.impressions) AS total_impressions,
            SUM(ms.clicks) AS total_clicks,
            ROUND(
                CASE
                    WHEN SUM(ms.impressions) = 0 THEN 0
                    ELSE (SUM(ms.clicks) * 100.0) / SUM(ms.impressions)
                END,
                2
            ) AS overall_ctr_pct,
            ROUND(
                CASE
                    WHEN SUM(ms.spend_usd) = 0 THEN 0
                    ELSE (SUM(ms.clicks) * 1000.0) / SUM(ms.spend_usd)
                END,
                2
            ) AS clicks_per_1000_spend
        FROM marketing_spend ms
        JOIN movies m
            ON m.movie_id = ms.movie_id
        WHERE {" AND ".join(where)}
        GROUP BY m.movie_id, m.title
        ORDER BY
            total_spend_usd DESC,
            total_clicks DESC,
            overall_ctr_pct DESC
        LIMIT ?
    """

    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return build_tool_response(
        tool_name="get_top_campaign_titles",
        source="SQL: marketing_spend, movies",
        filters={
            "year": year,
            "limit": limit,
        },
        rows=rows,
    )


# =========================================================
# SECTION 11: MANUAL TEST BLOCK
# Purpose:
# - Run this module directly for quick SQL smoke tests.
# =========================================================

if __name__ == "__main__":
    print("Testing sql_tool.py...\n")

    print("Top Q1 2025 movies:")
    print(
        query_movie_data(
            start_date="2025-01-01",
            end_date="2025-03-31",
            limit=5,
        )
    )

    print("\nLatest regional stats:")
    print(get_regional_stats(limit=5))

    print("\nGenre trends:")
    print(get_genre_trends(month="2025-05", limit=5))

    print("\nCampaign performance:")
    print(get_campaign_performance(title="Stellar Run", year=2025, limit=5))

    print("\nTop campaign titles:")
    print(get_top_campaign_titles(year=2025, limit=5))
