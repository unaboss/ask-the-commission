"""Optional Tavily web search for recent press coverage, kept separate from the record."""

import os

import httpx

TAVILY_URL = "https://api.tavily.com/search"


def enabled() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def search_news(question: str, k: int = 5) -> list[dict]:
    if not enabled():
        return []
    resp = httpx.post(TAVILY_URL, timeout=20, json={
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": f"Madlanga Commission {question}",
        "topic": "news",
        "max_results": k,
    })
    resp.raise_for_status()
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:400], "published": r.get("published_date", "")}
            for r in resp.json().get("results", [])]


def search_judgments(question: str, k: int = 5) -> list[dict]:
    """Public court judgments from SAFLII / Laws.Africa that may relate to the question."""
    if not enabled():
        return []
    resp = httpx.post(TAVILY_URL, timeout=20, json={
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": f"judgment {question}",
        "include_domains": ["saflii.org", "lawlibrary.org.za"],
        "max_results": k,
    })
    resp.raise_for_status()
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": (r.get("content") or "")[:400], "published": r.get("published_date", "")}
            for r in resp.json().get("results", [])]
