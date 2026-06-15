"""
Loads the six CSV files from project_root/data into SQLite.

Run from the project root with either:
    python -m backend.db.seed
or:
    python -m backend.db.seed_sqlite
"""
import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = BASE_DIR / "datacore_telugu_movies.db"

SCHEMA_SQL = """
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS viewers;
DROP TABLE IF EXISTS watch_activity;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS marketing_spend;
DROP TABLE IF EXISTS regional_performance;

CREATE TABLE movies (
    movie_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    genre TEXT NOT NULL,
    release_date TEXT NOT NULL,
    director TEXT NOT NULL,
    budget_usd INTEGER NOT NULL,
    rating REAL NOT NULL
);

CREATE TABLE viewers (
    viewer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    age_group TEXT NOT NULL,
    subscription_tier TEXT NOT NULL
);

CREATE TABLE watch_activity (
    activity_id TEXT PRIMARY KEY,
    viewer_id TEXT NOT NULL,
    movie_id TEXT NOT NULL,
    watch_date TEXT NOT NULL,
    watch_duration_min INTEGER NOT NULL,
    completion_pct INTEGER NOT NULL,
    FOREIGN KEY(viewer_id) REFERENCES viewers(viewer_id),
    FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    viewer_id TEXT NOT NULL,
    movie_id TEXT NOT NULL,
    rating_stars INTEGER NOT NULL,
    sentiment_label TEXT NOT NULL,
    review_date TEXT NOT NULL,
    FOREIGN KEY(viewer_id) REFERENCES viewers(viewer_id),
    FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE marketing_spend (
    campaign_id TEXT PRIMARY KEY,
    movie_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    spend_usd INTEGER NOT NULL,
    impressions INTEGER NOT NULL,
    clicks INTEGER NOT NULL,
    campaign_month TEXT NOT NULL,
    FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE regional_performance (
    region_id TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    total_views INTEGER NOT NULL,
    avg_rating REAL NOT NULL,
    growth_pct REAL NOT NULL,
    report_month TEXT NOT NULL
);

CREATE INDEX idx_watch_movie_date ON watch_activity(movie_id, watch_date);
CREATE INDEX idx_reviews_movie_date ON reviews(movie_id, review_date);
CREATE INDEX idx_marketing_movie_month ON marketing_spend(movie_id, campaign_month);
CREATE INDEX idx_regions_month_city ON regional_performance(report_month, city);
"""

CSV_TABLES = [
    ("movies", "movies.csv"),
    ("viewers", "viewers.csv"),
    ("watch_activity", "watch_activity.csv"),
    ("reviews", "reviews.csv"),
    ("marketing_spend", "marketing_spend.csv"),
    ("regional_performance", "regional_performance.csv"),
]


def insert_csv(cursor, table_name: str, filename: str) -> int:
    csv_path = DATA_DIR / filename
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"CSV has no header: {filename}")
        rows = [tuple(row[h] for h in headers) for row in reader]

    placeholders = ",".join(["?"] * len(headers))
    columns = ",".join(headers)
    cursor.executemany(
        f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    counts = {}
    for table_name, filename in CSV_TABLES:
        counts[table_name] = insert_csv(cursor, table_name, filename)

    conn.commit()
    conn.close()

    print(f"SQLite database created: {DB_PATH}")
    print("Loaded row counts:")
    for table_name, count in counts.items():
        print(f"- {table_name}: {count}")


if __name__ == "__main__":
    main()
