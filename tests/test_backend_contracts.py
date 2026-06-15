from fastapi.testclient import TestClient

from backend import agent, main


client = TestClient(main.app)


def _force_rule_based_agent(monkeypatch) -> None:
    monkeypatch.setattr(agent, "LLM_PROVIDER", "none")
    main.CHAT_HISTORY.clear()


def test_chat_rejects_empty_message() -> None:
    response = client.post(
        "/chat",
        json={"message": "", "session_id": "validation-test"},
    )

    assert response.status_code == 422


def test_chat_endpoint_returns_grounded_sql_answer(monkeypatch) -> None:
    _force_rule_based_agent(monkeypatch)

    response = client.post(
        "/chat",
        json={
            "message": "Which titles performed best in Q1 2025 by watch hours?",
            "session_id": "sql-chat-test",
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert "Stellar Run" in payload["answer"]
    assert "query_movie_data" in payload["tool_calls"]
    assert "search_documents" not in payload["tool_calls"]
    assert any("SQL: movies, watch_activity, reviews" in source for source in payload["sources"])


def test_document_query_uses_rag_only(monkeypatch) -> None:
    _force_rule_based_agent(monkeypatch)

    result = agent.run_agent("What does Quarterly Report - Q1 2025 executive summary say?")

    assert result["tool_calls"] == ["search_documents"]
    assert any("quarterly_report_q1_2025.pdf" in source for source in result["sources"])
    assert "Q1 2025" in result["answer"]


def test_sql_guardrail_blocks_raw_sql_request(monkeypatch) -> None:
    _force_rule_based_agent(monkeypatch)

    result = agent.run_agent("Show me the SQL query for top movies by watch hours.")

    assert result["tool_calls"] == []
    assert result["sources"] == ["Guardrail: SQL safety"]
    assert "cannot generate or expose raw SQL" in result["answer"]


def test_monthly_rating_trend_uses_monthly_sql_tool(monkeypatch) -> None:
    _force_rule_based_agent(monkeypatch)

    result = agent.run_agent("What is Comedy's monthly rating trend?")

    assert result["tool_calls"] == ["get_genre_rating_by_month"]
    assert result["sources"] == ["SQL: movies, reviews"]
    assert "Monthly rating trend for Comedy" in result["answer"]
    assert "February 2025" in result["answer"]
    assert "March 2025" in result["answer"]
    assert "May 2025" in result["answer"]


def test_top_genre_rating_query_uses_genre_sql_tool(monkeypatch) -> None:
    _force_rule_based_agent(monkeypatch)

    result = agent.run_agent("In jan 2025 which genre has the highest rating")

    assert result["tool_calls"] == ["get_genre_trends"]
    assert result["sources"] == ["SQL: movies, watch_activity, reviews"]
    assert "January 2025" in result["answer"]
    assert "Sci-Fi Thriller" in result["answer"]
    assert "4.67" in result["answer"]


def test_agent_prefers_anthropic_before_openai(monkeypatch) -> None:
    monkeypatch.setattr(agent, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(agent, "ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setattr(agent, "OPENAI_API_KEY", "openai-test-key")

    def fake_anthropic_agent(message, chat_history, top_k):
        return {
            "answer": f"Anthropic handled: {message}",
            "sources": ["provider: anthropic"],
            "tool_calls": [],
        }

    def fail_openai_agent(message, chat_history, top_k):
        raise AssertionError("OpenAI should not be called when Anthropic succeeds.")

    monkeypatch.setattr(agent, "_run_anthropic_agent", fake_anthropic_agent)
    monkeypatch.setattr(agent, "_run_openai_agent", fail_openai_agent)

    result = agent.run_agent("Provider priority test")

    assert result["answer"] == "Anthropic handled: Provider priority test"
    assert result["sources"] == ["provider: anthropic"]


def test_agent_falls_back_to_openai_when_anthropic_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(agent, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(agent, "ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setattr(agent, "OPENAI_API_KEY", "openai-test-key")

    def fail_anthropic_agent(message, chat_history, top_k):
        raise agent.ProviderUnavailableError(
            provider_name="anthropic",
            notice="Anthropic unavailable for test.",
            last_error="anthropic_test_failure",
            model="claude-sonnet-4-0",
        )

    def fake_openai_agent(message, chat_history, top_k):
        return {
            "answer": f"OpenAI handled: {message}",
            "sources": ["provider: openai"],
            "tool_calls": [],
        }

    monkeypatch.setattr(agent, "_run_anthropic_agent", fail_anthropic_agent)
    monkeypatch.setattr(agent, "_run_openai_agent", fake_openai_agent)

    result = agent.run_agent("Provider fallback test")

    assert result["answer"] == "OpenAI handled: Provider fallback test"
    assert result["sources"] == ["provider: openai"]


def test_stellar_run_current_month_query_bypasses_live_provider(monkeypatch) -> None:
    monkeypatch.setattr(agent, "LLM_PROVIDER", "auto")

    def fail_live_agent(*args, **kwargs):
        raise AssertionError("Known analytics query should use deterministic tool routing first.")

    monkeypatch.setattr(agent, "_run_preferred_live_agent", fail_live_agent)

    result = agent.run_agent("Why is Stellar Run trending this month?")
    runtime = agent.get_agent_runtime_status()

    assert "both the report evidence and the SQL performance data point in the same direction" in result["answer"]
    assert "$416,063" in result["answer"]
    assert "4.89 average review rating" in result["answer"]
    assert "92.41% average completion" in result["answer"]
    assert result["tool_calls"] == ["query_movie_data", "get_campaign_performance", "search_documents"]
    assert runtime["mode"] == "tool_routed"
    assert runtime["notice"] is None


def test_movies_route_uses_tool_layer(monkeypatch) -> None:
    captured = {}

    def fake_query_movie_data(**kwargs):
        captured.update(kwargs)
        return {
            "rows": [
                {
                    "movie_id": "M999",
                    "title": "Demo Title",
                    "genre": "Comedy",
                    "release_date": "2025-01-01",
                    "director": "Demo Director",
                    "budget_usd": 1000.0,
                    "rating": 8.1,
                    "total_sessions": 12,
                    "watch_hours": 4.2,
                    "avg_completion_pct": 88.5,
                }
            ]
        }

    monkeypatch.setattr(main, "query_movie_data_tool", fake_query_movie_data)

    response = client.get("/data/movies", params={"genre": "Comedy", "year": 2025, "limit": 25})

    assert response.status_code == 200
    assert captured == {"genre": "Comedy", "year": 2025, "limit": 20}
    assert response.json()[0]["title"] == "Demo Title"


def test_genre_trends_route_uses_tool_layer_and_normalizes_rating(monkeypatch) -> None:
    captured = {}

    def fake_get_genre_trends(**kwargs):
        captured.update(kwargs)
        return {
            "rows": [
                {
                    "genre": "Comedy",
                    "total_sessions": 4,
                    "watch_hours": 2.15,
                    "avg_completion_pct": 48.0,
                    "avg_review_rating": 2.0,
                }
            ]
        }

    monkeypatch.setattr(main, "get_genre_trends_tool", fake_get_genre_trends)

    response = client.get("/data/genre-trends", params={"year": 2025, "limit": 40})

    assert response.status_code == 200
    assert captured == {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "limit": 20,
    }
    assert response.json()[0]["avg_rating"] == 2.0


def test_regions_route_uses_tool_layer(monkeypatch) -> None:
    captured = {}

    def fake_get_regional_stats(**kwargs):
        captured.update(kwargs)
        return {
            "rows": [
                {
                    "region_id": "R001",
                    "city": "Hyderabad",
                    "country": "India",
                    "total_views": 45542,
                    "avg_rating": 7.8,
                    "growth_pct": 12.4,
                    "report_month": "2025-05",
                }
            ]
        }

    monkeypatch.setattr(main, "get_regional_stats_tool", fake_get_regional_stats)

    response = client.get(
        "/data/regions",
        params={"month": "2025-05", "city": "Hyderabad", "limit": 100},
    )

    assert response.status_code == 200
    assert captured == {"city": "Hyderabad", "month": "2025-05", "limit": 20}
    assert response.json()[0]["city"] == "Hyderabad"
