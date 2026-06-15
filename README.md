# Chat Assistant

Internal analytics assistant for the DataCore Telugu streaming assessment. The project combines FastAPI, SQLite, tool-based grounding, PDF RAG, ChromaDB, and a React frontend so answers are tied to backend evidence instead of free-form model guesses.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m backend.db.seed
.venv\Scripts\python.exe -c "from backend.rag.ingest import ingest_docs_folder; print(ingest_docs_folder())"
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

## Overview

- Tool-grounded answers only: the assistant answers with SQL tools, RAG tools, or both.
- Source attribution on every factual answer: SQL answers cite backend SQL sources and report answers cite PDF chunks.
- Safe architecture: the LLM does not write or execute raw SQL.
- Two data paths:
  - chat requests go through `backend/agent.py`
  - chart requests call backend data endpoints directly for deterministic output
- Multi-provider chat support:
  - `LLM_PROVIDER=auto` tries Anthropic first and OpenAI second
  - if live providers fail, the app falls back to deterministic backend tool routing

## Architecture Overview

### Component map

```text
+--------------------+      +----------------------+      +----------------------+
| React frontend     | ---> | FastAPI backend      | ---> | backend/agent.py     |
| ChatPanel          |      | backend/main.py      |      | tool routing         |
| InsightsChart      |      | request validation   |      | answer composition   |
+--------------------+      +----------------------+      +----------+-----------+
                                                                           |
                                              +----------------------------+---------------------------+
                                              |                                                        |
                                              v                                                        v
                                  +----------------------+                               +----------------------+
                                  | tools/sql_tool.py    |                               | tools/rag_tool.py    |
                                  | SQL analytics tools  |                               | PDF retrieval tools  |
                                  +----------+-----------+                               +----------+-----------+
                                             |                                                      |
                                             v                                                      v
                                  +----------------------+                               +----------------------+
                                  | SQLite database      |                               | vector_store.py       |
                                  | seeded from CSVs     |                               | ChromaDB PDF chunks   |
                                  +----------------------+                               +----------------------+
```

### Query-to-answer data flow

```text
User question
    |
    v
ChatPanel.jsx
    |
    | POST /chat
    v
backend/main.py
    |
    | validate ChatRequest, session_id, top_k
    v
backend/agent.py
    |
    | classify query type
    |
    +--> SQL question ------> tools/sql_tool.py ------> SQLite ------+
    |                                                                |
    +--> Report question ---> tools/rag_tool.py -----> ChromaDB -----+--> grounded evidence
    |                                                                |
    +--> Hybrid question ---> SQL tool + RAG tool -------------------+
    |
    v
backend/agent.py formats final answer + sources + tool_calls
    |
    v
backend/main.py returns JSON or SSE stream
    |
    v
Frontend renders the answer, source badges, and chart/table output
```

### Request-type flow

```text
SQL analytics query
User -> ChatPanel -> /chat -> main.py -> agent.py -> sql_tool.py -> SQLite -> answer

PDF/report query
User -> ChatPanel -> /chat -> main.py -> agent.py -> rag_tool.py -> vector_store.py / ChromaDB -> answer

Hybrid query
User -> ChatPanel -> /chat -> main.py -> agent.py -> sql_tool.py + rag_tool.py -> combined grounded answer

Chart request
User -> InsightsChart -> /data/genre-trends -> main.py -> sql_tool.py -> SQLite -> chart response
```

## Project Structure

```text
chat-assistant/
|-- backend/
|   |-- main.py
|   |-- agent.py
|   |-- models.py
|   |-- db/
|   |   `-- seed.py
|   |-- tools/
|   |   |-- sql_tool.py
|   |   `-- rag_tool.py
|   `-- rag/
|       |-- ingest.py
|       |-- vector_store.py
|       `-- chroma_store/
|-- frontend/
|   `-- src/
|-- data/
|-- docs/
|-- requirements.txt
|-- docker-compose.yml
`-- README.md
```

## Backend Responsibilities

- `backend/main.py`
  Receives API requests, validates payloads with Pydantic, manages session history, and returns grounded responses.
- `backend/agent.py`
  Applies guardrails, selects the right backend tools, collects evidence, and composes the final answer.
- `backend/tools/sql_tool.py`
  Exposes safe analytics helpers over the SQLite dataset.
- `backend/tools/rag_tool.py`
  Retrieves relevant PDF content from the vector store and document chunks.
- `backend/rag/ingest.py`
  Chunks PDFs, creates embeddings, and writes them into ChromaDB.
- `backend/rag/vector_store.py`
  Wraps ChromaDB persistence and retrieval.
- `backend/models.py`
  Defines shared request models, response models, settings, and tool argument validation.

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- PowerShell on Windows

### 1. Create the Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

Create `.env` in the project root using `.env.example` as the starting point.

Recommended `.env` shape:

```env
APP_ENV=development

DATABASE_URL=sqlite:///backend/db/datacore_telugu_movies.db
CHROMA_DB_PATH=backend/rag/chroma_store
DOCS_PATH=docs
EMBEDDING_MODEL=text-embedding-3-large

LLM_PROVIDER=auto
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-0
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-5.4-mini

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Environment variable notes:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLite database path used by SQL tools |
| `CHROMA_DB_PATH` | Local Chroma persistence directory |
| `DOCS_PATH` | Folder containing the PDF reports |
| `EMBEDDING_MODEL` | Sentence-transformers embedding model used for PDF chunks |
| `LLM_PROVIDER` | `auto`, `anthropic`, `openai`, or `none` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_CHAT_MODEL` | Anthropic model name |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_CHAT_MODEL` | OpenAI model name |
| `CORS_ORIGINS` | Frontend origins allowed by FastAPI |

Provider behavior:

- `LLM_PROVIDER=auto` tries Anthropic first, then OpenAI.
- `LLM_PROVIDER=none` disables live model calls and uses deterministic tool routing only.
- If a live provider is unavailable, the backend falls back to the tool-routed agent path.

### 3. Seed the SQLite data

```powershell
.venv\Scripts\python.exe -m backend.db.seed
```

This loads the CSV files from `data/` into the SQLite database used by the SQL tools.
The SQLite file is generated locally at `backend/db/datacore_telugu_movies.db`, so it does not need to be committed to the repository.

### 4. Ingest the PDFs into ChromaDB

```powershell
.venv\Scripts\python.exe -c "from backend.rag.ingest import ingest_docs_folder; print(ingest_docs_folder())"
```

This reads the PDFs from `docs/`, chunks them, embeds them, and stores the chunks in ChromaDB.

### 5. Run the backend

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

### 6. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

### 7. Optional: run the full stack with Docker Compose

```powershell
docker compose up
```

## Verification

Check that the vector store contains chunks:

```powershell
.venv\Scripts\python.exe -c "from backend.rag.vector_store import VectorStore; print(VectorStore().get_collection().count())"
```

Smoke-test RAG retrieval:

```powershell
.venv\Scripts\python.exe -c "from backend.tools.rag_tool import search_documents; print(search_documents('Stellar Run campaign performance', top_k=3))"
```

Smoke-test the grounded agent:

```powershell
.venv\Scripts\python.exe -c "from backend.agent import run_agent; print(run_agent('Which titles performed best in Q1 2025 by watch hours?'))"
```

Run the evaluation script:

```powershell
.venv\Scripts\python.exe scripts\evaluate_examples.py
```

Run backend tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Example Questions

- `Which titles performed best in Q1 2025 by watch hours?`
- `Why is Stellar Run trending this month?`
- `Compare audience engagement: Dark Orbit vs Last Kingdom.`
- `Which city had the strongest viewer growth last 30 days?`
- `What explains weak comedy genre performance?`
- `What strategic recommendations would you make for next quarter?`

## Frontend and API Notes

- `POST /chat` returns a complete grounded response.
- `POST /chat/stream` streams status and answer chunks through SSE.
- `GET /data/genre-trends` powers the frontend genre chart directly from the SQL tool layer.
- The chart path bypasses the LLM intentionally so dashboard data stays deterministic.

## Troubleshooting

- Backend does not start
  Check the Python environment, dependency install, and `.env` values.
- Frontend cannot reach the API
  Check `VITE_API_BASE_URL`, backend port `8000`, and `CORS_ORIGINS`.
- RAG returns weak or empty answers
  Re-run PDF ingestion and confirm the Chroma collection has chunks.
- Chat falls back instead of using a live model
  Check provider API keys, provider quotas, and `LLM_PROVIDER`.
- Wrong chart numbers
  Check `backend/tools/sql_tool.py` and the `/data/genre-trends` endpoint path.

## Assessment Alignment

- Uses backend tools only for factual data access.
- Keeps source attribution in grounded answers.
- Validates inputs with shared Pydantic models.
- Separates SQL retrieval, PDF retrieval, API routing, and answer composition cleanly.
- Includes setup, seeding, ingestion, run instructions, and architecture/data-flow diagrams in the README.
