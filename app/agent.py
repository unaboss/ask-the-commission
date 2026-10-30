"""Nemotron research agent: plan searches, gather evidence, answer with citations.

The loop is model-agnostic JSON planning, so it works on any OpenAI-compatible
chat endpoint. By default it calls NVIDIA Nemotron on Nebius Token Factory.
"""

import json
import os
import re
import time

from openai import OpenAI

from .retrieval import Index, Passage

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL = os.environ.get("NEBIUS_MODEL", "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")
MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "3"))
PER_QUERY = int(os.environ.get("AGENT_PER_QUERY", "6"))
MAX_EVIDENCE = int(os.environ.get("AGENT_MAX_EVIDENCE", "24"))

PLANNER_SYSTEM = """/no_think
You are a research planner for investigators studying the Madlanga Commission of Inquiry
(South Africa) into criminality, political interference and corruption in the criminal
justice system. You can search a full-text index of hearing transcripts and witness statements.

Reply with ONE JSON object and nothing else, in one of these forms:
{"action": "search", "reason": "<short>", "searches": [{"query": "<keywords or \\"exact phrase\\">", "speaker": "<optional surname>"}]}
{"action": "answer", "reason": "<why the evidence is enough>"}

Rules:
- Use 1 to 4 searches per round. Use distinctive keywords, names, places, case numbers.
- Keyword search, not semantic: try synonyms and name variants (e.g. "Mkhwanazi", "General Mkhwanazi").
- Only set "speaker" when the question is about what a specific person said.
- Choose "answer" once the evidence covers the question, or nothing new is turning up."""

ANSWER_SYSTEM = """/no_think
You are an evidence analyst for the Madlanga Commission of Inquiry. Answer the investigator's
question using ONLY the numbered evidence passages provided.

- Cite every factual claim with evidence ids in square brackets, e.g. [E3] or [E2][E5].
- Quote short key phrases verbatim in double quotes where they matter.
- Attribute statements to the speaker ("X testified that…"); testimony is allegation, not proven fact.
- If witnesses disagree, say so and cite both sides.
- If the evidence does not answer the question, say so plainly and suggest what to search next.
- Be concise: a direct answer first, then supporting points as a short list."""


MODES = {
    "ask": "",
    "brief": ("Prepare a WITNESS PREPARATION BRIEF for the person named in the question. Sections: "
              "1) What they testified (key claims, cited); 2) Who corroborates or contradicts them (cited); "
              "3) Open questions and gaps for cross-examination."),
    "contradict": ("Find and EXPLAIN CONTRADICTIONS on this topic. Search for each side. For every conflict: "
                   "state side A and side B with citations, explain exactly where they differ (who, what, when), "
                   "and say what evidence would resolve it."),
    "verify": ("The question is a CLAIM (e.g. from a news report). Check it against the record. Start with a "
               "verdict: Supported, Contradicted, Partly supported, or Not in the record. Then explain with citations."),
}


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    # Nemotron sometimes cites with full-width brackets, e.g. 【E2】; normalise to [E2].
    return re.sub(r"[【［]\s*(E\d+)\s*[】］]", r"[\1]", text)


def parse_json(text: str) -> dict:
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class Agent:
    def __init__(self, index: Index, client: OpenAI | None = None, model: str = MODEL):
        self.index = index
        self.model = model
        self.client = client or OpenAI(base_url=BASE_URL, api_key=os.environ.get("NEBIUS_API_KEY"))

    def _chat(self, system: str, user: str, max_tokens: int = 1200) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _format_evidence(evidence: list[Passage]) -> str:
        lines = []
        for i, p in enumerate(evidence, 1):
            where = f"Day {p.day}" if p.day else p.source
            if p.page:
                where += f", p.{p.page}"
            if p.line:
                where += f", line {p.line}"
            lines.append(f"[E{i}] {p.speaker or 'Unknown'} | {where} | {p.date}\n{p.text}")
        return "\n\n".join(lines)

    def run(self, question: str, mode: str = "ask") -> dict:
        task = MODES.get(mode, "")
        if task:
            question = f"{task}\n\n{question}"
        t0 = time.time()
        evidence: list[Passage] = []
        seen: set[tuple[str, int]] = set()
        asked: set[str] = set()
        trace = []

        for round_no in range(1, MAX_ROUNDS + 1):
            summary = self._format_evidence(evidence) if evidence else "(none yet)"
            plan = parse_json(self._chat(
                PLANNER_SYSTEM,
                f"Question: {question}\n\nRound {round_no} of {MAX_ROUNDS}.\n"
                f"Evidence gathered so far:\n{summary[:6000]}",
                max_tokens=600))
            searches = [s for s in plan.get("searches") or []
                        if isinstance(s, dict) and str(s.get("query", "")).strip().lower() not in asked]
            if plan.get("action") == "answer" and evidence:
                trace.append({"round": round_no, "action": "answer", "reason": plan.get("reason", "")})
                break
            if not searches:
                if evidence:  # no usable or only repeated searches: answer with what we have
                    trace.append({"round": round_no, "action": "answer",
                                  "reason": plan.get("reason") or "No new searches proposed."})
                    break
                searches = [{"query": question}]
            step = {"round": round_no, "action": "search", "reason": plan.get("reason", ""), "searches": []}
            new = 0
            for s in searches[:4]:
                q = str(s.get("query", "")).strip()
                if not q or q.lower() in asked:
                    continue
                asked.add(q.lower())
                hits = self.index.search(q, k=PER_QUERY, speaker=s.get("speaker") or None)
                added = 0
                for h in hits:
                    key = (h.source, h.source_id)
                    if key not in seen and len(evidence) < MAX_EVIDENCE:
                        seen.add(key)
                        evidence.append(h)
                        added += 1
                new += added
                step["searches"].append({"query": q, "speaker": s.get("speaker") or None,
                                         "hits": len(hits), "new": added})
            trace.append(step)
            if new == 0 and evidence:
                break

        if not evidence:
            return {"answer": "I could not find any transcript passages about this. "
                              "Try different names, places or keywords.",
                    "citations": [], "trace": trace, "model": self.model,
                    "seconds": round(time.time() - t0, 1)}

        prompt = f"Question: {question}\n\nEvidence:\n{self._format_evidence(evidence)}"
        answer = strip_think(self._chat(ANSWER_SYSTEM, prompt, max_tokens=1500))
        if not re.search(r"\[E\d+\]", answer):  # the model occasionally drops citations; ask once more
            answer = strip_think(self._chat(
                ANSWER_SYSTEM,
                f"{prompt}\n\nYour previous answer had no [E#] citations. Rewrite it, citing every claim "
                f"with evidence ids like [E1].\n\nPrevious answer:\n{answer}",
                max_tokens=1500))

        cited = sorted({int(n) for n in re.findall(r"\[E(\d+)\]", answer)
                        if 1 <= int(n) <= len(evidence)})
        citations = [{"id": f"E{n}", **evidence[n - 1].to_dict()} for n in cited]
        return {"answer": answer, "citations": citations, "trace": trace,
                "evidence_count": len(evidence), "model": self.model,
                "seconds": round(time.time() - t0, 1)}
