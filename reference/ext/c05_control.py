CONTROL_EFFECT = 0.010
control_tickers = ML_UNIVERSE_US[:12] + ML_UNIVERSE_TASI[:12]
control_daily = {t: synthetic_daily(t, market_of(t), planted_effect=CONTROL_EFFECT) for t in control_tickers}
control_exp = run_experiment(control_daily, CFG, label="POSITIVE CONTROL — synthetic data with a PLANTED pattern")
print("This is NOT a result about markets. A pattern was planted on purpose (on days opening >1% above the previous\n"
      "close the stock drifts up ~1% more). If the machinery is sound, the ML ranker must find it; the rule-based\n"
      "ranker, which has no gap-continuation weighting beyond its fixed gap score, is shown for contrast.\n")
show_experiment(control_exp)
control_importance = global_importance(control_exp["data"])
print("\nWhat the model learned in the control (the opening gap should now matter far more than in the pure-noise run):")
display(control_importance.rename(index=FEATURE_EN).head(4).to_frame("importance"))

control_lo, control_hi = control_exp["ml_vs_universe_ci"]
gap_now, gap_before = control_importance["gap_pct"], importance["gap_pct"]
print(f"Opening-gap importance: {gap_before:.4f} on noise -> {gap_now:.4f} with the planted pattern")
control_found = bool(control_lo > 0 and gap_now > max(5 * gap_before, 0.005))
print(f"\nControl verdict: {'PASSED — the ML ranker detects a real pattern when one exists' if control_found else 'FAILED — investigate before trusting any real-data result'}")
