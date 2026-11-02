# CLAUDE.md — stoX

Project: stoX, stock analytics for Tadawul (TASI) and US markets. CPIT-498 graduation project. Read `HANDOFF.md` first for scope, decisions, milestones and open questions.

## Phase
Backend only (Python 3, FastAPI, scikit-learn, pandas, SQLite). No UI yet.

## Rules
- Descriptive analytics only. Never present output as investment advice; include a `disclaimer` in responses with picks or valuations.
- No look-ahead: features for day t use only the open of t and data up to t-1. Keep the look-ahead test.
- Never mix real and synthetic data in one training set; always label results with their data mode.
- Picks are written before the open and never edited. Results are appended after the close.
- Report average net return and profit factor next to hit rate. Say so honestly if there is no edge.
- API keys come from the environment (`FINNHUB_API_KEY`, `SAHMK_API_KEY`) and stay on the server. Never log or commit them.
- Tests must run offline (synthetic provider, mocked HTTP).
- Language: code, comments, identifiers and docs in English. User-facing text must be translatable (Arabic and English).

## Reference
`reference/` holds the working notebook prototype and its cell sources. It is a source to port from, not a package to import.

## Commands (create these in M0)
- `pytest` runs the test suite
- `uvicorn app.main:app --reload` runs the API
- `python -m ml.train` and `python -m ml.predict` train and run the model
