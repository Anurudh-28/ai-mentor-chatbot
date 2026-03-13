from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, *, serper_api_key: str | None, num_results: int = 5) -> list[WebResult]:
    """
    If SERPER_API_KEY is provided, uses Serper (Google).
    Otherwise falls back to DuckDuckGo (no key) via duckduckgo_search.
    """
    query = (query or "").strip()
    if not query:
        return []
    if serper_api_key:
        return _search_serper(query, api_key=serper_api_key, num_results=num_results)
    return _search_duckduckgo(query, num_results=num_results)


def _search_serper(query: str, *, api_key: str, num_results: int) -> list[WebResult]:
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": num_results},
        timeout=20,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    out: list[WebResult] = []
    for r in (data.get("organic") or [])[:num_results]:
        out.append(
            WebResult(
                title=str(r.get("title") or ""),
                url=str(r.get("link") or ""),
                snippet=str(r.get("snippet") or ""),
            )
        )
    return out


def _search_duckduckgo(query: str, *, num_results: int) -> list[WebResult]:
    try:
        from duckduckgo_search import DDGS
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DuckDuckGo fallback requires duckduckgo-search. Install: pip install duckduckgo-search") from e

    out: list[WebResult] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            out.append(
                WebResult(
                    title=str(r.get("title") or ""),
                    url=str(r.get("href") or ""),
                    snippet=str(r.get("body") or ""),
                )
            )
    return out


def format_web_context(results: list[WebResult]) -> str:
    if not results:
        return ""
    parts: list[str] = ["Use the following web results if relevant:\n"]
    for i, r in enumerate(results, start=1):
        parts.append(f"[web:{i}] {r.title}\n{r.snippet}\nURL: {r.url}\n")
    return "\n".join(parts).strip()


def should_web_search(user_text: str) -> bool:
    t = (user_text or "").lower()
    triggers = [
    "latest",
    "today",
    "2025",
    "2026",
    "current",
    "news",
    "ai jobs",
    "machine learning jobs",
    "salary",
    "trend",
    "hiring",
]
    return any(x in t for x in triggers)

