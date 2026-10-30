# Ask the Commission: Madlanga Evidence Agent

**Turn months of hearings into evidence you can cite in minutes.**

Ask the Commission is an inquiry intelligence engine. Its first case study is South Africa's
Madlanga Commission (the Judicial Commission of Inquiry into Criminality, Political Interference
and Corruption in the Criminal Justice System). Investigators ask in plain English. The agent
plans its own searches over 154 hearing days and 137,000 transcript segments, then answers with
verbatim, cited quotes: speaker, day, page and line.

Built for the Nebius x NVIDIA Global AI Hackathon, track **Best Apps and Agents**.

## How it works

1. **Plan.** Nemotron returns a JSON search plan: 1 to 4 keyword searches per round, with an
   optional speaker filter.
2. **Retrieve.** SQLite FTS5 with BM25 ranking and Porter stemming. User input is always quoted,
   so it cannot inject FTS syntax. Neighbouring transcript lines are added to each hit for context.
3. **Iterate.** Up to 3 rounds. The agent stops when a round turns up nothing new or the planner
   repeats itself.
4. **Answer.** Nemotron sees only the numbered evidence. Every claim is cited, testimony is
   attributed to the speaker, and disagreements between witnesses are flagged.
5. **Verify.** Only citations that point at retrieved passages are kept, with one automatic retry if
   the model cites nothing. The UI links each `[E#]` to its quote and shows the full agent trace.

## Modes

- **Ask a question.** A direct, cited answer.
- **Witness prep brief.** What a witness testified, who corroborates or contradicts them, and open
  questions for cross-examination.
- **Explain contradictions.** Side A and side B with citations, where exactly they differ, and what
  would resolve it.
- **Verify a claim.** Paste a claim, for example from a news report, and get a verdict: *Supported*,
  *Contradicted*, *Partly supported* or *Not in the record*.

### Optional: Tavily web search

With `TAVILY_API_KEY` set, the UI can add two separate panels: recent press coverage, and possibly
related public judgments from SAFLII and lawlibrary.org.za. These are shown beside the answer and
are **never** fed into Nemotron's evidence.

## How NVIDIA models and Nebius Token Factory are used

- `nvidia/nemotron-3-super-120b-a12b` runs on **Nebius Token Factory** through its
  OpenAI-compatible API.
- Nemotron does two jobs: it plans the searches (structured JSON) and analyses the evidence (the
  cited answer).
- Reasoning blocks (`<think>…</think>`) are stripped before parsing or display.
- The model is configurable with `NEBIUS_MODEL`. Use Nemotron 3 Nano 30B for speed or Nemotron 3
  Ultra 550B for depth.
- A question costs under about 1 US cent: 3 to 4 model calls, roughly 15k tokens in and 1.5k out.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
cp /path/to/commission_research.db data/
cp .env.example .env                           # then set NEBIUS_API_KEY
.venv/bin/python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

### Docker

```bash
docker build -t ask-the-commission .           # data/commission_research.db must be present
docker run -p 8000:8000 --env-file .env ask-the-commission
```

### API

| Method | Path | Body / query |
|---|---|---|
| `POST` | `/api/ask` | `{"question": "...", "mode": "ask\|brief\|contradict\|verify", "include_news": false}` |
| `GET` | `/api/search` | `?q=...&k=10&speaker=...` returns raw ranked passages |
| `GET` | `/api/health` | index stats, model name, whether the key is set |

### Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

The tests run offline against a tiny fixture database and a stub chat client.

## Who it's for

- Legal teams and evidence leaders preparing witnesses
- Journalists and newsrooms
- Investigators and civil-society organisations
- Law students and researchers

## What's next

- Evidence bundle PDF export with line references
- One-click witness preparation pack
- A dedicated contradiction engine
- Multi-user workspaces with annotations
- Bring your own record: upload any country's commission-of-inquiry or court transcripts (African
  judicial commissions, UK public inquiries, Australian royal commissions, ICC trial transcripts)
- Private case-file workspaces for law firms

## What was built since August 26, 2026

The Madlanga research pipeline that scraped the hearings and built `commission_research.db` existed
before the hackathon and is not part of this repository; the agent only reads its database.

Everything in this repository is new work from today: the Nemotron agent loop, the FastAPI
service, the web UI with agent trace, the Token Factory integration, the Tavily panels, Docker
packaging and tests.

## Responsible use

Testimony is allegation, not proven fact. Always check the cited transcript before relying on an
answer.

## License

MIT. See [LICENSE](LICENSE).
