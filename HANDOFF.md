# stoX — Handoff for Claude Code

Read this file first, then `CLAUDE.md`. It tells you what stoX is, what the supervising doctor asked for, what has been decided, and what to build first. Build the **backend only** in this phase. No web or mobile UI yet.

## 1. What stoX is

stoX is a stock analytics platform for the Saudi exchange (Tadawul/TASI) and US markets. It will become a web app and a mobile app on one backend API. Two tracks:

- **Track A, day trading:** every morning before the market opens, an ML model picks 3 to 5 stocks per market and explains each pick in plain language. After the close the real outcome is recorded. A monthly success rate is reported.
- **Track B, long-term investing:** a 30-point scorecard (5 axes x 6 yes/no checks), Altman Z, two fair-value methods, filing sentiment, and a quarterly ranked report of all stocks.
- **Portfolio (idea from an earlier student app):** users track holdings across both markets with profit/loss in SAR or USD.

It is **descriptive analytics, never investment advice**. Every output and every screen must say so.

Course context: CPIT-498 graduation project (KAU). In CPIT-498 the deliverables are documents (idea, use cases, database, requirements). Implementation starts next phase, which is why we are building the backend now. The graduation committee also said: **if the project uses AI, the model must actually be trained.**

## 2. What the supervising doctor asked for

Source: the meeting transcript (Arabic). Mapped to deliverables:

| # | Doctor's request | Requirement | Status |
|---|---|---|---|
| D1 | "How did the AI choose this stock?" | A real ML model plus a per-pick explanation | Prototype done (notebook sections 10, 11) |
| D2 | Analyze several timeframes: 5 days, 10 days, a month, a year | Return and volatility features over 5, 10, 21, 252 trading days | Prototype done (section 9) |
| D3 | Daily picks (3 to 5), then check afterwards whether each succeeded | Store picks before the open, record the result after the close | Prototype done (section 13), needs a server job |
| D4 | The model's success rate after a month | Monthly success-rate query/endpoint | Prototype done (view `v_monthly_success`) |
| D5 | A report that refreshes every quarter, ranking stocks on the 5-axis scorecard | Quarterly job and ranked report | Prototype done (section 14) on illustrative fixtures only |
| D6 | For day trading, the dividend factor is not relevant (volume matters); keep the 5% dividend baseline for the investment track | Day-trading ranking uses volume/range/gap, not dividends | Done in the notebook |
| D7 | 498 phase = documents: idea, use cases, database, requirements | Docs, not code | Done (a Docs artifact exists, plus a presentation deck) |

## 3. Decisions already made

- **Backend first.** UIs are deferred. Then web (installable PWA is fine), then mobile.
- **Language and stack:** Python 3, FastAPI, scikit-learn, pandas/NumPy, SQLite for now (design so PostgreSQL is a config change). One language for server and ML.
- **Do not port the old Java app.** Only its idea carries over. Its CSV symbol lists can seed the stock catalog: `farefive/stoX` has `src/main/resources/stocksList.csv` (about 5,350 US symbols) and `stocksListSA.csv` (about 200 Saudi stocks).
- **Live prices:** Finnhub (US) and Sahmk (Saudi). Keys live on the server only, never in a client.
- **Training history:** Finnhub/Sahmk may not offer daily history on the plan the team has. Unverified. Use `yfinance` for **training data only** unless the team confirms a provider. Make the history provider a swappable interface.
- **Model:** `HistGradientBoostingClassifier` (sklearn), shallow on purpose. A LightGBM swap must be a one-function change.
- **Success label:** a stock "succeeds" if it closes at least +2.0% above its open. Make it configurable (`AI_SUCCESS_PCT`); 2% vs 3% is an open question for the doctor.
- **Currency:** SAR is pegged to USD at 3.75, so no FX API is needed.

## 4. Provider facts (taken from the earlier app's code)

- **Finnhub quote:** `GET https://finnhub.io/api/v1/quote?symbol={TICKER}&token={KEY}`. Current price is field `c`. Env var `FINNHUB_API_KEY`.
- **Sahmk quote:** `GET https://app.sahmk.sa/api/v1/quote/{SYMBOL}/` with header `X-API-Key: {KEY}`. Current price is field `price`. Env var `SAHMK_API_KEY`.
- **Symbol mapping trap:** Sahmk uses the numeric Tadawul code (e.g. `2222`), yfinance uses `2222.SR`. Keep one canonical id in the DB and map per provider.
- The earlier app wrapped fetchers in Retry (decorator) and Cache (proxy). Keep both behaviors: retries with backoff, TTL cache, clear errors, never silently wrong data.
- Nothing else about either API is known. Read their docs for history, batch endpoints, rate limits and plan limits before relying on them.

## 5. Build plan for this phase (backend)

Work in milestones. Each must end with passing tests and a short note in `docs/PROGRESS.md`.

- **M0 Scaffold:** repo layout, config from env/`.env`, logging, `pytest`, a `/health` endpoint, `.env.example`, README.
- **M1 Data layer:** provider interfaces (`QuoteProvider`, `HistoryProvider`), Finnhub, Sahmk, yfinance history, and a **synthetic provider** for offline tests. Retry, TTL cache, symbol mapping, stock catalog table seeded from the CSVs.
- **M2 ML engine:** port `reference/ext/c02..c05` into importable modules (`features`, `model`, `explain`, `control`) with no notebook globals. Add `train.py` (fit one final model on all history, save with joblib plus a `model_card.json`) and `predict.py` (load model, build features for today, return top-k with reasons).
- **M3 Tracking:** tables from `reference/schema_tracking.sql`. Jobs: `picks` (before open), `results` (after close), monthly stats.
- **M4 Analysis engine:** port the scorecard, Altman Z, valuation and sentiment from the notebook's sections 5 to 7 into modules with typed inputs. Add the quarterly report job.
- **M5 Accounts and portfolio:** users (hashed passwords, token auth), portfolios, holdings (fractional shares), P/L in SAR or USD, composite portfolio totals.
- **M6 API and scheduling:** endpoints below, OpenAPI docs, a job runner (cron or APScheduler) for the daily and quarterly jobs.

### Suggested layout

```
stox_server/
  app/            # FastAPI app, routers, auth, schemas
  providers/      # finnhub.py, sahmk.py, history.py, synthetic.py, cache/retry
  ml/             # features.py, model.py, explain.py, control.py, train.py, predict.py
  analysis/       # scorecard.py, altman.py, valuation.py, sentiment.py
  jobs/           # picks.py, results.py, quarterly.py
  db/             # schema, migrations, repositories
  tests/
  docs/
```

### Endpoints (first cut)

- `GET /health`
- `POST /auth/register`, `POST /auth/login`
- `GET /stocks/search?q=` (catalog; show logos later)
- `GET /quotes/{ticker}`
- `GET/POST/DELETE /portfolios`, `/portfolios/{id}/holdings`, `GET /portfolios/{id}/summary` (value, P/L, currency query param)
- `GET /picks/today?market=US|TASI` with probability and `reason` text
- `GET /picks/history?from=&to=` with outcomes
- `GET /stats/monthly` (success rate and average net return per month, per model version)
- `GET /stocks/{ticker}/analysis` (scorecard breakdown, Altman, both fair values, sentiment)
- `GET /report/quarterly` (ranked, with changes versus the previous report)

## 6. The ML spec (what the prototype already proves)

- **Features, all knowable before the open:** `gap_pct`, `rvol` (prior-day relative volume), `atr_pct`, `spread_pct` (Corwin–Schultz proxy), `ret_1d/5d/10d/21d/252d`, `vol_21d`, `dist_high_252d`, `is_tasi`. Horizons `(5, 10, 21, 252)` trading days.
- **No look-ahead:** features at day *t* use only the open of *t* and data up to *t−1*. Keep `assert_no_lookahead` as a real test.
- **Model parameters:** `max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=1.0, early_stopping=False`.
- **Evaluation (offline, honest):** expanding-window walk-forward, 250-session warm-up, retrain every 21 sessions, 1-session embargo. Compare against (a) the rule-based ranking with renormalized weights and (b) the whole universe without ranking. Report hit rate **and** average net return and profit factor, plus a daily bootstrap 95% interval and rank IC. Hit rate can rise while profit does not. Never report hit rate alone.
- **Serving (production):** train one final model on all history, save it, retrain monthly. Walk-forward is for measuring, not for serving.
- **Explanations:** global permutation importance, and a per-pick occlusion-vs-median that turns the top 3 factors into a plain sentence (`FEATURE_TEXT` in `c04_explain.py`). Arabic wording was dropped in the English notebook; the app needs both languages later, so keep the text table swappable by locale.
- **Positive control:** plant a known pattern in synthetic data and assert the pipeline finds it. Keep it as a test. It guards against a pipeline that cannot detect anything.
- **Costs:** 0.10% round-trip cost is deducted from outcomes.

## 7. Guardrails (non-negotiable)

1. **Never leak the future.** No feature may use data from the pick day beyond its open.
2. **Never mix real and synthetic series** in one training set. Label every result with its data mode. The notebook's current numbers come from **synthetic data** and say nothing about real markets.
3. **Picks are written before the open and are never edited afterwards.** Results are appended after the close. This is what makes the track record verifiable.
4. **Show profit beside hit rate.** Report an honest "no edge found" if that is the truth. A null result is acceptable for the course and must not be hidden.
5. **Not advice.** Every API response that carries picks or valuations includes a `disclaimer` field. Day trading is high risk.
6. **Secrets:** API keys only from the environment. Hash passwords. Never log keys. Do not commit `.env`.
7. **Provider failures:** retry with backoff, serve stale cached prices flagged as stale, otherwise return a clear error. Never invent a price.
8. **Data licensing:** Yahoo-sourced data is fine for study and training here. A public launch needs a licensed provider. Keep providers behind interfaces so this is swappable.

## 8. Testing and acceptance

- `pytest` runs fully **offline** using the synthetic provider and mocked HTTP for Finnhub/Sahmk.
- Required tests: look-ahead, walk-forward embargo (a pick is never scored by a model that saw its day), positive control passes, every stored pick has exactly one result, picks per market per day is between 3 and 5, the quarterly report is sorted by total with stable tie-breaks, provider symbol mapping, retry/cache behavior, auth, and P/L math including SAR/USD conversion.
- Acceptance for M2: `train.py` then `predict.py` run end to end on synthetic data and produce reasons; the evaluation script prints the same three-way comparison as the notebook.
- Acceptance for M6: a full simulated day works: picks job, results job, then `/picks/history` and `/stats/monthly` show it.

## 9. Open questions (ask the team, do not guess)

1. Do the team's Finnhub and Sahmk plans provide **daily historical prices**? If not, `yfinance` stays the training source.
2. Where do **fundamentals** come from for the scorecard in production? The notebook used `yfinance` with illustrative fallback fixtures. Check Finnhub's financials endpoints and Sahmk.
3. Success threshold: **2% or 3%** (doctor's brief said 2 to 3%).
4. Universe size: the prototype uses 25 US + 20 TASI tickers. The Saudi ticker list needs verification on the data provider.
5. Shariah-compliance filter: **proposed, not built.** Suggested approach: AAOIFI Standard 21 style screens (total debt up to 30% of market cap, deposits up to 30%, impermissible income up to 5% of revenue); some Saudi providers use 33% of total assets. Needs financial-statement data and a decision.
6. Database: SQLite for now, or PostgreSQL from the start?
7. Arabic and English output: reasons and report labels should be translatable.

## 10. Reference material in this bundle

```
reference/
  stoX_quantitative_engine_poc_v2.ipynb   full prototype, runs offline on synthetic data (~40 s)
  ext/c01_data.py       universes, display names, daily-data loader, synthetic generator
  ext/c02_features.py   features, label, dataset builder, look-ahead test
  ext/c03_model.py      model, walk-forward, picking, metrics, bootstrap, experiment
  ext/c04_explain.py    importance and per-pick plain-language reasons
  ext/c05_control.py    positive-control experiment
  ext/c06_tracking.py   SQLite tracking (see also schema_tracking.sql)
  ext/c07_quarterly.py  quarterly report and change detection
  ext/c08_export.py     JSON/CSV export and self-checks
  schema_tracking.sql   stock, model_run, daily_pick, pick_result, quarterly_score, v_monthly_success
  sample_outputs/       stox_results.json, ai_daily_picks.csv, quarterly_report.csv, stox_tracking_demo.db
```

**Porting warning.** The `ext/` files are notebook cells, not modules. They use globals defined in the original notebook cells 0 to 21: `CFG`/`DayTradingConfig`, `market_of`, `US`/`TASI`, `build_baskets` and `performance_report` (day-trading ranking and metrics), and `Fundamentals`, `companies`, `valuations`, `scorecards`, `fundamentals_sources`, `to_native` (long-term track). The scorecard, Altman, valuation and sentiment code live in the original cells (sections 5 to 7) and still need extracting. Turn each into a proper module with explicit inputs. Do not copy the globals approach.

## 11. Known gaps and honest status

- ML results so far are on **synthetic** data: ML hit rate 25.7% vs 18.8% rule-based vs 14.5% universe, but average net return stays slightly negative (about −0.02% per trade), and the interval against the universe includes 0. That is the expected result on random data. The positive control (planted pattern) is found, with average net return +0.31%.
- Real-market performance is **untested**. Nothing here should be presented as evidence of profit.
- The quarterly report currently runs on 8 tickers using **illustrative fixtures**, not real filings.
- Portfolio, accounts, API, scheduling and all UI are **not built**.
- Saudi tickers in the expanded universe are unverified on the data provider.
