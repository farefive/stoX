def metrics_to_dict(frame: pd.DataFrame) -> dict:
    return {col: {k: (None if pd.isna(v) or np.isinf(v) else float(v)) for k, v in frame[col].items()} for col in frame}


ai_payload = {
    "data_mode": DATA_MODE,
    "note": ("Descriptive screening statistics, not advice. With DATA_MODE=SYNTHETIC nothing here says anything about "
             "real markets: it only proves the machinery works."),
    "universe": {"tickers": len(daily_data), "US": sum(market_of(t) is US for t in daily_data),
                 "TASI": sum(market_of(t) is TASI for t in daily_data)},
    "success_definition": f"closes at least +{AI_SUCCESS_PCT}% above its open",
    "horizons_days": list(HORIZONS),
    "out_of_sample": {"first": main_exp["days"][0].strftime("%Y-%m-%d"), "last": main_exp["days"][-1].strftime("%Y-%m-%d"),
                      "sessions": len(main_exp["days"]), "retrainings": len(main_exp["blocks"]),
                      "base_rate_pct": main_exp["base_rate_pct"]},
    "metrics": metrics_to_dict(main_exp["metrics"]),
    "ml_vs_universe_ci_pct": main_exp["ml_vs_universe_ci"], "ml_vs_rule_ci_pct": main_exp["ml_vs_rule_ci"],
    "rank_ic_mean": main_exp["rank_ic_mean"],
    "importance": importance.to_dict(),
    "latest_session_picks": [
        {"date": r.date, "market": r.market, "rank": r.rank, "ticker": r.ticker, "company_name": DISPLAY_NAMES.get(r.ticker),
         "probability": r.p_success, "change_pct": r.gross_pct, "success": r.success, "reason": r.reason}
        for r in latest_picks.itertuples()],
    "monthly": monthly.to_dict(orient="records"),
    "overall": overall.to_dict(orient="records"),
    "positive_control": {"passed": control_found, "effect_planted": CONTROL_EFFECT,
                         "metrics": metrics_to_dict(control_exp["metrics"]),
                         "ml_vs_universe_ci_pct": control_exp["ml_vs_universe_ci"],
                         "gap_importance_noise": float(importance["gap_pct"]),
                         "gap_importance_control": float(control_importance["gap_pct"])},
}
payload["ai_day_trading"] = ai_payload
payload["quarterly_report"] = quarterly.to_dict(orient="records")
payload["fundamentals_source"] = sorted(set(fundamentals_sources.values()))
payload = to_native(payload)
with open("stox_results.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=1)
ml_picks_explained[["date", "market", "rank", "ticker", "p_success", "reason", "open", "close", "gross_pct", "net_pct",
                    "success"]].to_csv("ai_daily_picks.csv", index=False, encoding="utf-8-sig")
print("Wrote stox_results.json (full output contract incl. AI + tracking + quarterly), ai_daily_picks.csv, "
      "quarterly_report.csv and", TRACKING_DB)

# ---- Self-checks: fail loudly if a later change breaks a guarantee -------------------------------------------------
ml_check = main_exp["ml_picks"]
assert ml_check.groupby(["date", "market"]).size().between(1, 5).all() and ml_check["rank"].max() <= CFG.top_k
assert ml_check["p_success"].notna().all()
assert all(main_exp["blocks"][int(b)]["trained_through"] < d for b, d in zip(ml_check["block"], ml_check["date"])), \
    "a pick was scored by a model that had seen its own day"
assert main_exp["days"][0] >= dataset.index.get_level_values("date").unique().sort_values()[250]
n_picks, n_results = (tracking.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("daily_pick", "pick_result"))
assert n_picks == n_results > 0, "every stored pick must have a stored result"
assert tracking.execute("PRAGMA foreign_key_check").fetchall() == []
assert list(quarterly["total"]) == sorted(quarterly["total"], reverse=True)
assert control_found, "positive control failed: the pipeline cannot detect a planted pattern"
json.dumps(payload)
print("All AI-extension self-checks passed.")
