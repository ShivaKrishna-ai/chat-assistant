import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import run_agent


@dataclass
class ExampleCase:
    prompt: str
    expected_tools: List[str]
    expected_source_fragments: List[str]
    expected_keywords: List[str]
    forbidden_tools: List[str] = field(default_factory=list)
    forbidden_source_fragments: List[str] = field(default_factory=list)


EXAMPLE_CASES = [
    ExampleCase(
        prompt="Which titles performed best in Q1 2025 by watch hours?",
        expected_tools=["query_movie_data"],
        expected_source_fragments=["SQL: movies, watch_activity, reviews"],
        expected_keywords=["Stellar Run", "Dark Orbit"],
        forbidden_tools=["search_documents"],
    ),
    ExampleCase(
        prompt="Why is Stellar Run trending this month?",
        expected_tools=["query_movie_data", "get_campaign_performance", "search_documents"],
        expected_source_fragments=["campaign_performance_2025.pdf", "SQL: marketing_spend, movies"],
        expected_keywords=["Stellar Run", "$416,063", "4.89", "92.41%"],
    ),
    ExampleCase(
        prompt="Compare audience engagement: Dark Orbit vs Last Kingdom.",
        expected_tools=["query_movie_data"],
        expected_source_fragments=["SQL: movies, watch_activity, reviews"],
        expected_keywords=["Dark Orbit", "Last Kingdom"],
        forbidden_tools=["search_documents"],
    ),
    ExampleCase(
        prompt="Which city had the strongest viewer growth last 30 days?",
        expected_tools=["get_regional_stats"],
        expected_source_fragments=["SQL: regional_performance"],
        expected_keywords=["Hyderabad"],
        forbidden_tools=["search_documents"],
    ),
    ExampleCase(
        prompt="In which month did Comedy have a 2.0 rating?",
        expected_tools=["get_genre_rating_by_month"],
        expected_source_fragments=["SQL: movies, reviews"],
        expected_keywords=["March 2025", "May 2025"],
        forbidden_tools=["search_documents", "get_genre_trends"],
    ),
    ExampleCase(
        prompt="What is Comedy's monthly rating trend?",
        expected_tools=["get_genre_rating_by_month"],
        expected_source_fragments=["SQL: movies, reviews"],
        expected_keywords=["Monthly rating trend for Comedy", "February 2025", "March 2025", "May 2025"],
        forbidden_tools=["search_documents", "get_genre_trends"],
    ),
    ExampleCase(
        prompt="In jan 2025 which genre has the highest rating",
        expected_tools=["get_genre_trends"],
        expected_source_fragments=["SQL: movies, watch_activity, reviews"],
        expected_keywords=["January 2025", "Sci-Fi Thriller", "4.67"],
        forbidden_tools=["search_documents", "get_genre_rating_by_month"],
    ),
    ExampleCase(
        prompt="What does Quarterly Report - Q1 2025 executive summary say?",
        expected_tools=["search_documents"],
        expected_source_fragments=["quarterly_report_q1_2025.pdf"],
        expected_keywords=["171.0 watch hours", "69.9%", "Comedy was the weakest tracked genre"],
        forbidden_tools=["query_movie_data", "get_genre_trends", "get_regional_stats"],
    ),
    ExampleCase(
        prompt="What are the Top campaign titles by spend in campaign_performance_2025?",
        expected_tools=["search_documents"],
        expected_source_fragments=["campaign_performance_2025.pdf"],
        expected_keywords=["Stellar Run - $811,522", "Uppal Utsavam - $270,256", "Karimnagar Kiratham - $226,358"],
        forbidden_tools=["get_top_campaign_titles", "get_campaign_performance", "query_movie_data"],
    ),
    ExampleCase(
        prompt="What does Campaign Performance Report - 2025 say about ROI proxy?",
        expected_tools=["search_documents"],
        expected_source_fragments=["campaign_performance_2025.pdf"],
        expected_keywords=["clicks per $1,000 of spend", "ROI proxy by title"],
        forbidden_tools=["get_campaign_performance", "query_movie_data"],
    ),
    ExampleCase(
        prompt="What explains weak comedy genre performance?",
        expected_tools=["get_genre_trends", "search_documents"],
        expected_source_fragments=["quarterly_report_q1_2025.pdf", "SQL: movies, watch_activity, reviews"],
        expected_keywords=["Comedy", "watch hours"],
    ),
    ExampleCase(
        prompt="What strategic recommendations would you make for next quarter?",
        expected_tools=["search_documents", "get_genre_trends"],
        expected_source_fragments=["campaign_performance_2025.pdf", "quarterly_report_q1_2025.pdf"],
        expected_keywords=["next quarter", "Stellar Run", "Dark Orbit"],
    ),
]


def evaluate_case(case: ExampleCase) -> dict:
    result = run_agent(
        message=case.prompt,
        session_id="evaluation-session",
        chat_history=[],
        top_k=4,
    )

    answer = str(result.get("answer", "") or "")
    sources = [str(item) for item in result.get("sources", [])]
    tool_calls = [str(item) for item in result.get("tool_calls", [])]
    lowered_answer = answer.lower()

    checks = {
        "answer_present": bool(answer.strip()),
        "sources_present": bool(sources),
        "tools_present": bool(tool_calls),
        "expected_tools_present": all(tool in tool_calls for tool in case.expected_tools),
        "forbidden_tools_absent": all(tool not in tool_calls for tool in case.forbidden_tools),
        "expected_sources_present": all(
            any(fragment in source for source in sources)
            for fragment in case.expected_source_fragments
        ),
        "forbidden_sources_absent": all(
            all(fragment not in source for source in sources)
            for fragment in case.forbidden_source_fragments
        ),
        "expected_keywords_present": all(
            keyword.lower() in lowered_answer for keyword in case.expected_keywords
        ),
    }

    passed = sum(1 for value in checks.values() if value)
    total = len(checks)
    score = round((passed / total) * 100, 1)

    return {
        "prompt": case.prompt,
        "score": score,
        "checks": checks,
        "answer": answer,
        "sources": sources,
        "tool_calls": tool_calls,
    }


def main() -> None:
    results = [evaluate_case(case) for case in EXAMPLE_CASES]
    overall_score = round(sum(item["score"] for item in results) / len(results), 1)
    source_failures = [
        item["prompt"]
        for item in results
        if not item["checks"]["sources_present"] or not item["checks"]["expected_sources_present"]
    ]

    summary = {
        "overall_score": overall_score,
        "case_count": len(results),
        "source_failures": source_failures,
        "results": results,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
