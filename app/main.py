"""FastAPI app: web UI + JSON API for the Madlanga Nemotron research agent."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import news
from .agent import Agent, MODEL
from .retrieval import Index

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Madlanga Commission Nemotron Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_index: Index | None = None
_agent: Agent | None = None


def get_agent() -> Agent:
    global _index, _agent
    if _agent is None:
        _index = Index()
        _agent = Agent(_index)
    return _agent


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    include_news: bool = False
    mode: str = Field(default="ask", pattern="^(ask|brief|contradict|verify)$")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    try:
        stats = get_agent().index.stats()
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e), "model": MODEL}
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("NEBIUS_API_KEY")),
            "news": news.enabled(), **stats}


@app.get("/api/search")
def search(q: str, k: int = 10, speaker: str | None = None):
    return [p.to_dict() for p in get_agent().index.search(q, k=min(k, 50), speaker=speaker)]


@app.post("/api/ask")
def ask(body: Ask):
    try:
        result = get_agent().run(body.question, body.mode)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # surface model/API errors to the UI
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    if body.include_news and news.enabled():
        try:
            result["news"] = news.search_news(body.question)
            result["judgments"] = news.search_judgments(body.question)
        except Exception as e:  # news is optional; never fail the answer over it
            result["news_error"] = f"{type(e).__name__}: {e}"
    return result
