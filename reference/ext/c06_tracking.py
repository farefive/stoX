import sqlite3
import os

TRACKING_DB = "stox_tracking_demo.db"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stock (
    ticker   TEXT PRIMARY KEY,
    market   TEXT NOT NULL CHECK (market IN ('US', 'TASI')),
    company_name  TEXT);
CREATE TABLE IF NOT EXISTS model_run (
    model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version   TEXT NOT NULL,          -- 'ML-v1' (learned) or 'RULE-v1' (fixed-weight formula)
    trained_through TEXT NOT NULL,          -- last session the model was allowed to see; 'n/a' for the formula
    n_train_rows    INTEGER,
    params          TEXT,
    created_at      TEXT,
    UNIQUE (model_version, trained_through));
CREATE TABLE IF NOT EXISTS daily_pick (     -- written BEFORE the market opens
    pick_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date  TEXT NOT NULL,
    ticker     TEXT NOT NULL REFERENCES stock (ticker),
    model_id   INTEGER NOT NULL REFERENCES model_run (model_id),
    rank       INTEGER NOT NULL,
    score      REAL NOT NULL,
    reason  TEXT,
    UNIQUE (pick_date, ticker, model_id));
CREATE TABLE IF NOT EXISTS pick_result (    -- written AFTER the market closes
    pick_id     INTEGER PRIMARY KEY REFERENCES daily_pick (pick_id),
    open_price  REAL,
    close_price REAL,
    change_pct  REAL,                       -- close vs open, before costs
    net_pct     REAL,                       -- after the round-trip cost assumption
    success     INTEGER NOT NULL CHECK (success IN (0, 1)),
    recorded_at TEXT);
CREATE TABLE IF NOT EXISTS quarterly_score (
    report_date TEXT NOT NULL,
    ticker      TEXT NOT NULL REFERENCES stock (ticker),
    rank        INTEGER,
    value INTEGER, dividend INTEGER, health INTEGER, past INTEGER, future INTEGER,
    total       INTEGER NOT NULL,
    rating   TEXT, yield_class TEXT, dcf_class TEXT,
    PRIMARY KEY (report_date, ticker));
CREATE VIEW IF NOT EXISTS v_monthly_success AS
    SELECT substr(p.pick_date, 1, 7) AS month, m.model_version,
           COUNT(*) AS picks, SUM(r.success) AS successes,
           ROUND(100.0 * SUM(r.success) / COUNT(*), 1) AS success_rate_pct,
           ROUND(AVG(r.net_pct), 3) AS avg_net_pct
    FROM daily_pick p JOIN pick_result r USING (pick_id) JOIN model_run m USING (model_id)
    GROUP BY month, m.model_version;
"""


def init_db(path: str = TRACKING_DB, fresh: bool = True) -> sqlite3.Connection:
    if fresh and os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


def register_stocks(conn: sqlite3.Connection, tickers) -> None:
    conn.executemany("INSERT OR IGNORE INTO stock VALUES (?, ?, ?)",
                     [(t, market_of(t).name, DISPLAY_NAMES.get(t)) for t in sorted(set(tickers))])


def log_model(conn: sqlite3.Connection, version: str, trained_through, n_rows, params: str = "") -> int:
    conn.execute("INSERT OR IGNORE INTO model_run (model_version, trained_through, n_train_rows, params, created_at) "
                 "VALUES (?, ?, ?, ?, ?)", (version, str(trained_through)[:10], n_rows, params,
                                            datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return conn.execute("SELECT model_id FROM model_run WHERE model_version = ? AND trained_through = ?",
                        (version, str(trained_through)[:10])).fetchone()[0]


def log_picks(conn: sqlite3.Connection, picks: pd.DataFrame) -> None:
    """Step 1 (before the open): store the day's picks. `picks` needs date, ticker, rank, score, reason, model_id."""
    conn.executemany(
        "INSERT OR IGNORE INTO daily_pick (pick_date, ticker, model_id, rank, score, reason) VALUES (?, ?, ?, ?, ?, ?)",
        [(r.date.strftime("%Y-%m-%d"), r.ticker, int(r.model_id), int(r.rank), float(r.score), r.reason)
         for r in picks.itertuples()])


def record_results(conn: sqlite3.Connection, picks: pd.DataFrame) -> None:
    """Step 2 (after the close): attach what actually happened. Needs open, close, gross_pct, net_pct, success."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO pick_result SELECT pick_id, ?, ?, ?, ?, ?, ? FROM daily_pick "
        "WHERE pick_date = ? AND ticker = ? AND model_id = ?",
        [(float(r.open), float(r.close), float(r.gross_pct), float(r.net_pct), int(r.success), stamp,
          r.date.strftime("%Y-%m-%d"), r.ticker, int(r.model_id)) for r in picks.itertuples()])


def replay_into_db(conn: sqlite3.Connection, exp: dict, explained: pd.DataFrame) -> None:
    """Replay the out-of-sample period day by day through the same two steps production would run each session."""
    register_stocks(conn, exp["data"].index.get_level_values("ticker").unique())
    ml = explained.copy()
    ml["model_id"] = [log_model(conn, "ML-v1", exp["blocks"][int(b)]["trained_through"],
                                exp["blocks"][int(b)]["n_train_rows"], "HistGradientBoosting depth3 150 iters")
                      for b in ml["block"]]
    ml["score"] = ml["p_success"]
    rule = exp["rule_picks"].copy()
    rule["model_id"] = log_model(conn, "RULE-v1", "n/a", None, "fixed weights rvol/atr/gap + spread penalty")
    rule["reason"] = "Ranked by fixed weights (trading volume, daily range, opening gap)"
    for picks in (ml, rule):
        log_picks(conn, picks)
        record_results(conn, picks)
    conn.commit()


TRACKING_PATH = os.path.abspath(TRACKING_DB)
tracking = init_db(TRACKING_DB)
replay_into_db(tracking, main_exp, ml_picks_explained)

monthly = pd.read_sql_query("SELECT * FROM v_monthly_success ORDER BY month, model_version", tracking)
overall = pd.read_sql_query(
    """SELECT m.model_version, COUNT(*) AS picks, SUM(r.success) AS successes,
              ROUND(100.0 * SUM(r.success) / COUNT(*), 1) AS success_rate_pct, ROUND(AVG(r.net_pct), 3) AS avg_net_pct
       FROM daily_pick p JOIN pick_result r USING (pick_id) JOIN model_run m USING (model_id)
       GROUP BY m.model_version""", tracking)
print(f"Tracking database written to {TRACKING_PATH}")
print("\nEvery pick is stored before the open and its result after the close. Monthly success rate (last 6 months):")
display(monthly.tail(12))
print("\nOverall over the whole out-of-sample period:")
display(overall)
print("\nOne day as stored in the database (picks + what happened):")
display(pd.read_sql_query(
    """SELECT p.pick_date, m.model_version, p.rank, p.ticker, ROUND(p.score, 3) AS score,
              ROUND(r.change_pct, 2) AS change_pct, r.success
       FROM daily_pick p JOIN pick_result r USING (pick_id) JOIN model_run m USING (model_id)
       WHERE p.pick_date = (SELECT MAX(pick_date) FROM daily_pick) ORDER BY m.model_version, p.rank""", tracking))
