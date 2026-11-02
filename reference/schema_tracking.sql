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
