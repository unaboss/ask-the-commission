"""Offline tests: a tiny fixture DB with the pipeline schema and a stubbed chat client."""

import re
import sqlite3
from types import SimpleNamespace

import pytest

from app.agent import Agent, strip_think
from app.retrieval import Index, to_fts_query

SCHEMA = """
CREATE TABLE hearings (id INTEGER PRIMARY KEY, day_number INTEGER NOT NULL, hearing_date TEXT NOT NULL,
                       title TEXT DEFAULT '', summary TEXT DEFAULT '', video_url TEXT DEFAULT '',
                       location TEXT DEFAULT '');
CREATE TABLE speakers (id INTEGER PRIMARY KEY AUTOINCREMENT, canonical_name TEXT NOT NULL UNIQUE,
                       role TEXT NOT NULL DEFAULT 'other');
CREATE TABLE transcript_segments (id INTEGER PRIMARY KEY AUTOINCREMENT, hearing_id INTEGER, speaker_id INTEGER,
                                  text TEXT NOT NULL, page_number INTEGER DEFAULT 0, line_number INTEGER DEFAULT 0);
CREATE TABLE statements (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT DEFAULT '', stype TEXT DEFAULT 'other',
                         hearing_id INTEGER, author TEXT DEFAULT '', full_text TEXT NOT NULL, file_path TEXT DEFAULT '');
CREATE VIRTUAL TABLE search_index USING fts5(source, source_id UNINDEXED, title, text, date, speaker,
                                             tokenize='porter unicode61');
"""

SEGMENTS = [  # (speaker_id, text, page, line)
    (1, "When did you first find out about the decision to disband?", 41, 7),
    (2, "When the letter disbanding the task team was circulated in the media.", 41, 8),
    (1, "Thank you, General.", 41, 9),
]


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "fixture.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO hearings (id, day_number, hearing_date) VALUES (1, 12, '2025-10-02')")
    conn.executemany("INSERT INTO speakers (id, canonical_name) VALUES (?, ?)",
                     [(1, "Adv Hassim SC"), (2, "Lt Gen Sibiya")])
    for sid, text, page, line in SEGMENTS:
        cur = conn.execute("INSERT INTO transcript_segments (hearing_id, speaker_id, text, page_number, line_number) "
                           "VALUES (1, ?, ?, ?, ?)", (sid, text, page, line))
        speaker = conn.execute("SELECT canonical_name FROM speakers WHERE id = ?", (sid,)).fetchone()[0]
        conn.execute("INSERT INTO search_index VALUES ('transcript', ?, 'Day 12', ?, '2025-10-02', ?)",
                     (cur.lastrowid, text, speaker))
    conn.commit()
    conn.close()
    return str(path)


class StubClient:
    """Mimics openai.OpenAI().chat.completions.create with canned replies, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_agent_keeps_only_valid_citations(db_path):
    client = StubClient([
        '<think>planning</think>{"action": "search", "reason": "find it", "searches": [{"query": "letter disbanding"}]}',
        '{"action": "answer", "reason": "enough"}',
        'Lt Gen Sibiya said he learned of it from the media [E1]. Unsupported claim [E9].',
    ])
    result = Agent(Index(db_path), client=client, model="stub").run("How was the task team disbanding announced?")

    assert [c["id"] for c in result["citations"]] == ["E1"]
    cite = result["citations"][0]
    assert (cite["day"], cite["page"], cite["line"]) == (12, 41, 8)
    assert cite["speaker"] == "Lt Gen Sibiya"
    assert "LT GEN SIBIYA: When the letter" in cite["text"]  # neighbouring lines are included
    assert "ADV HASSIM SC:" in cite["text"]
    assert [s["action"] for s in result["trace"]] == ["search", "answer"]


def test_fts_query_is_fully_quoted():
    q = to_fts_query('who said "task team" AND (dockets*')
    terms = q.split(" OR ")
    assert '"task team"' in terms
    assert all(re.fullmatch(r'"[^"]+"', t) for t in terms)
    assert "(" not in q and "*" not in q


def test_full_width_brackets_are_normalised():
    assert strip_think("<think>x</think>Claim 【E2】 and ［E3］.") == "Claim [E2] and [E3]."


def test_chart_aggregates(db_path):
    idx = Index(db_path)
    assert idx.timeline("task team letter") == [{"day": 12, "date": "2025-10-02", "hits": 1}]
    assert idx.timeline("nothing matches this") == [{"day": 12, "date": "2025-10-02", "hits": 0}]
    assert sorted((h["speaker"], h["hits"]) for h in idx.speaker_hits("disband")) == \
        [("Adv Hassim SC", 1), ("Lt Gen Sibiya", 1)]
    ov = idx.overview()
    assert ov["days_per_month"] == [{"month": "2025-10", "days": 1}]
    assert ov["top_speakers"][0] == {"speaker": "Adv Hassim SC", "segments": 2}


def test_truncated_answer_is_trimmed_to_last_sentence():
    from app.agent import trim_truncated
    assert trim_truncated("He said it was unlawful [E1]. She disagreed and said the [E") == \
        "He said it was unlawful [E1]. …"
    assert trim_truncated('Quote: "very concerning." [E3] Then the minis') == 'Quote: "very concerning." [E3] …'
