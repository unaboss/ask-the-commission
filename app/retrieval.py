"""Read-only retrieval over the Madlanga Commission SQLite FTS5 index.

Works against the `commission_research.db` produced by the research pipeline
(tables: hearings, speakers, transcript_segments, statements, search_index).
"""

import os
import re
import sqlite3
from dataclasses import dataclass, asdict

DB_PATH = os.environ.get("MADLANGA_DB", "data/commission_research.db")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "has", "have", "he", "her", "his", "how", "i", "in", "is", "it", "its",
    "of", "on", "or", "say", "said", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "about", "tell", "me", "any", "there", "commission", "witness",
    "testified", "evidence", "given",
}

CONTEXT_SEGMENTS = 3     # neighbouring transcript segments on each side of a hit
CONTEXT_CHARS = 900


@dataclass
class Passage:
    source: str          # "transcript" or "statement"
    source_id: int
    title: str
    speaker: str
    date: str
    day: int | None
    page: int | None
    line: int | None
    text: str
    score: float

    def to_dict(self):
        return asdict(self)


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Database not found at {path}. Build it with the research pipeline "
            "or set MADLANGA_DB to its path."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def to_fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query.

    Quoted phrases are kept as phrases; other words become an OR of terms so a
    natural-language question still matches. Everything is quoted, so FTS5
    syntax characters in user input cannot break the query.
    """
    phrases = re.findall(r'"([^"]+)"', text)
    rest = re.sub(r'"[^"]+"', " ", text)
    words = [w for w in re.findall(r"[\w']+", rest.lower())
             if w not in STOPWORDS and len(w) > 1]
    parts = [f'"{p.replace(chr(34), "")}"' for p in phrases]
    parts += [f'"{w}"' for w in dict.fromkeys(words)]
    return " OR ".join(parts)


def _snippet(text: str, query: str, width: int = 700) -> str:
    if len(text) <= width:
        return text.strip()
    terms = [t.strip('"').lower() for t in query.split(" OR ") if t]
    low = text.lower()
    hits = [low.find(t) for t in terms if low.find(t) >= 0]
    start = max(0, min(hits) - width // 3) if hits else 0
    out = text[start:start + width].strip()
    return ("…" if start > 0 else "") + out + ("…" if start + width < len(text) else "")


class Index:
    def __init__(self, path: str = DB_PATH):
        self.conn = connect(path)

    def search(self, query: str, k: int = 8, speaker: str | None = None,
               date_from: str | None = None, date_to: str | None = None) -> list[Passage]:
        fts = to_fts_query(query)
        if not fts:
            return []
        sql = ("SELECT source, source_id, title, text, date, speaker, bm25(search_index) AS score "
               "FROM search_index WHERE search_index MATCH ?")
        args: list = [fts]
        if speaker:
            sql += " AND speaker LIKE ?"
            args.append(f"%{speaker}%")
        if date_from:
            sql += " AND date >= ?"
            args.append(date_from)
        if date_to:
            sql += " AND date <= ?"
            args.append(date_to)
        sql += " ORDER BY score LIMIT ?"
        args.append(k)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._passage(r, fts) for r in rows]

    def _passage(self, r: sqlite3.Row, fts: str) -> Passage:
        day = page = line = None
        text = None
        if r["source"] == "transcript":
            meta = self.conn.execute(
                "SELECT h.day_number, ts.page_number, ts.line_number, ts.hearing_id FROM transcript_segments ts "
                "LEFT JOIN hearings h ON h.id = ts.hearing_id WHERE ts.id = ?",
                (r["source_id"],)).fetchone()
            if meta:
                day, page = meta["day_number"], meta["page_number"] or None
                line = meta["line_number"] or None
                text = self._context(int(r["source_id"]), meta["hearing_id"])
        return Passage(
            source=r["source"], source_id=int(r["source_id"]), title=r["title"] or "",
            speaker=r["speaker"] or "", date=(r["date"] or "")[:10], day=day, page=page, line=line,
            text=text or _snippet(r["text"] or "", fts), score=round(float(r["score"]), 3),
        )

    def _context(self, seg_id: int, hearing_id: int | None) -> str | None:
        """The hit plus neighbouring segments as "SPEAKER: text" lines, centred on the hit."""
        if hearing_id is None:
            return None
        rows = self.conn.execute(
            "SELECT ts.id, ts.text, s.canonical_name AS speaker FROM transcript_segments ts "
            "LEFT JOIN speakers s ON s.id = ts.speaker_id "
            "WHERE ts.hearing_id = ? AND ts.id BETWEEN ? AND ? ORDER BY ts.id",
            (hearing_id, seg_id - CONTEXT_SEGMENTS, seg_id + CONTEXT_SEGMENTS)).fetchall()
        lines = [f"{(row['speaker'] or 'Unknown').upper()}: {row['text'].strip()}" for row in rows]
        hit = next((i for i, row in enumerate(rows) if row["id"] == seg_id), None)
        if hit is None:
            return None
        if len(lines[hit]) > CONTEXT_CHARS:
            return lines[hit][:CONTEXT_CHARS].rstrip() + "…"
        lo = hi = hit
        size = len(lines[hit])
        while True:
            grew = False
            for j in (lo - 1, hi + 1):
                if 0 <= j < len(lines) and size + len(lines[j]) + 1 <= CONTEXT_CHARS:
                    size += len(lines[j]) + 1
                    lo, hi = min(lo, j), max(hi, j)
                    grew = True
            if not grew:
                break
        return "\n".join(lines[lo:hi + 1])

    def speakers(self, like: str = "", limit: int = 20) -> list[str]:
        rows = self.conn.execute(
            "SELECT canonical_name FROM speakers WHERE canonical_name LIKE ? LIMIT ?",
            (f"%{like}%", limit)).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        q = lambda s: self.conn.execute(s).fetchone()[0]
        return {
            "hearing_days": q("SELECT COUNT(*) FROM hearings"),
            "segments": q("SELECT COUNT(*) FROM transcript_segments"),
            "statements": q("SELECT COUNT(*) FROM statements"),
            "speakers": q("SELECT COUNT(*) FROM speakers"),
            "first_date": q("SELECT MIN(hearing_date) FROM hearings"),
            "last_date": q("SELECT MAX(hearing_date) FROM hearings"),
        }
