RATING_BANDS = [(24, "Excellent"), (18, "Good"), (12, "Average"), (0, "Weak")]   # display convention only, not a statistical cut-off


def rating(total: int) -> str:
    return next(label for floor, label in RATING_BANDS if total >= floor)


def quarterly_report(companies: Dict[str, "Fundamentals"], valuations: Dict[str, dict], scorecards: Dict[str, dict],
                     report_date: str) -> pd.DataFrame:
    """One ranked table for the whole universe: re-issued after every results season (the pipeline of section 4.3)."""
    rows = []
    for ticker, card in scorecards.items():
        axes = card["axes"]
        rows.append({"report_date": report_date, "ticker": ticker, "company_name": DISPLAY_NAMES.get(ticker, ticker),
                     "market": companies[ticker].market, "total": card["total"],
                     "value": axes["Value"], "dividend": axes["Dividend"], "health": axes["Health"],
                     "past": axes["Past"], "future": axes["Future"], "rating": rating(card["total"]),
                     "yield_class": valuations[ticker]["target_yield_class"], "dcf_class": valuations[ticker]["dcf_class"]})
    report = pd.DataFrame(rows).sort_values(["total", "health", "ticker"], ascending=[False, False, True])
    report.insert(0, "rank", np.arange(1, len(report) + 1))
    return report.reset_index(drop=True)


def save_quarterly(conn: sqlite3.Connection, report: pd.DataFrame) -> None:
    register_stocks(conn, report["ticker"])
    cols = ["report_date", "ticker", "rank", "value", "dividend", "health", "past", "future", "total", "rating",
            "yield_class", "dcf_class"]
    conn.executemany(f"INSERT OR REPLACE INTO quarterly_score ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [tuple(None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else v) for v in row)
                      for row in report[cols].itertuples(index=False)])
    conn.commit()


def scorecard_changes(conn: sqlite3.Connection, report: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The `scorecard_changed` event: per-axis difference versus the previous stored report (None for the first one)."""
    previous = pd.read_sql_query(
        "SELECT * FROM quarterly_score WHERE report_date = (SELECT MAX(report_date) FROM quarterly_score "
        "WHERE report_date < ?)", conn, params=(report["report_date"].iloc[0],))
    if previous.empty:
        return None
    merged = report.merge(previous, on="ticker", suffixes=("", "_prev"))
    out = merged[["ticker"]].copy()
    for axis in ["value", "dividend", "health", "past", "future", "total"]:
        out[f"d_{axis}"] = merged[axis] - merged[f"{axis}_prev"]
    return out[(out.drop(columns="ticker") != 0).any(axis=1)]


REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
quarterly = quarterly_report(companies, valuations, scorecards, REPORT_DATE)
save_quarterly(tracking, quarterly)
quarterly.to_csv("quarterly_report.csv", index=False, encoding="utf-8-sig")   # opens correctly in Excel

print(f"Quarterly report as of {REPORT_DATE} — every stock ranked by its 30-point score "
      f"(fundamentals source: {sorted(set(fundamentals_sources.values()))})")
display(quarterly[["rank", "ticker", "company_name", "market", "total", "value", "dividend", "health", "past", "future",
                   "rating", "yield_class", "dcf_class"]])
changes = scorecard_changes(tracking, quarterly)
print("Changes versus the previous stored report:", "none (this is the first stored report)" if changes is None else "")
if changes is not None:
    display(changes)
