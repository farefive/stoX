# (name, wording when the stock is ABOVE the typical value, wording when it is BELOW) — used for plain-language reasons.
FEATURE_TEXT = {
    "gap_pct": ("Opening gap", "opened above yesterday's close", "opened near or below yesterday's close"),
    "rvol": ("Yesterday's trading activity", "trading yesterday was busier than usual", "trading yesterday was quieter than usual"),
    "atr_pct": ("Typical daily range", "usually moves over a wide range during the day", "usually moves over a narrow range during the day"),
    "spread_pct": ("Cost of getting in and out", "the gap between buy and sell prices is wide", "the gap between buy and sell prices is narrow"),
    "ret_1d": ("Yesterday's return", "rose more than usual yesterday", "rose less than usual yesterday, or fell"),
    "ret_5d": ("Last-week return", "did better than usual over the last week", "did worse than usual over the last week"),
    "ret_10d": ("Last-2-weeks return", "did better than usual over the last two weeks", "did worse than usual over the last two weeks"),
    "ret_21d": ("Last-month return", "did better than usual over the last month", "did worse than usual over the last month"),
    "ret_252d": ("Last-year return", "did better than usual over the last year", "did worse than usual over the last year"),
    "vol_21d": ("Last-month volatility", "has been volatile over the last month", "has been calm over the last month"),
    "dist_high_252d": ("Distance from the 1-year high", "is close to its highest price of the year", "is far below its highest price of the year"),
    "is_tasi": ("Market", "Saudi market", "US market"),
}
FEATURE_EN = {  # short English labels for charts (short labels for plots)
    "gap_pct": "Opening gap", "rvol": "Prior-day relative volume", "atr_pct": "Typical daily range (ATR%)",
    "spread_pct": "Spread proxy", "ret_1d": "Return 1d", "ret_5d": "Return 5d", "ret_10d": "Return 10d",
    "ret_21d": "Return 21d", "ret_252d": "Return 252d", "vol_21d": "Volatility 21d",
    "dist_high_252d": "Distance from 52-week high", "is_tasi": "Market (TASI=1)"}


def global_importance(dataset: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Which features matter overall: fit on the first 80 % of sessions, then shuffle each feature on the last 20 %
    and measure how much the model's log-loss gets worse (permutation importance)."""
    dates = dataset.index.get_level_values("date").unique().sort_values()
    cut = dates[int(len(dates) * 0.8)]
    row_dates = dataset.index.get_level_values("date")
    train, valid = dataset[row_dates < cut - pd.Timedelta(days=3)], dataset[row_dates >= cut]
    model = make_model(seed).fit(train[ML_FEATURES], train["success"])
    result = permutation_importance(model, valid[ML_FEATURES], valid["success"], scoring="neg_log_loss",
                                    n_repeats=5, random_state=seed)
    return pd.Series(result.importances_mean, index=ML_FEATURES).sort_values(ascending=False)


def occlusion_deltas(model, X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Per-stock explanation: how much does the model's probability fall when one feature is replaced by its typical value?"""
    base = model.predict_proba(X)[:, 1]
    deltas = {}
    for feature in X.columns:
        neutral = X.copy()
        neutral[feature] = medians[feature]
        deltas[feature] = base - model.predict_proba(neutral)[:, 1]
    return pd.DataFrame(deltas, index=X.index)


def reason_text(row: pd.Series, deltas: pd.Series, medians: pd.Series, k: int = 3) -> str:
    parts = []
    for feature in deltas.abs().sort_values(ascending=False).index[:k]:
        name, above, below = FEATURE_TEXT[feature]
        wording = above if row[feature] > medians[feature] else below
        parts.append(f"{wording} ({'raised' if deltas[feature] > 0 else 'lowered'} its rank)")
    return "; ".join(parts).capitalize()


def add_reasons(exp: dict) -> pd.DataFrame:
    """Attach a plain-English 'why was this stock picked' sentence to every ML pick (each pick explained by the model that scored it)."""
    picks, medians = exp["ml_picks"].copy(), exp["data"][ML_FEATURES].median()
    picks["reason"] = ""
    for block_id, group in picks.groupby("block"):
        deltas = occlusion_deltas(exp["blocks"][int(block_id)]["model"], group[ML_FEATURES], medians)
        picks.loc[group.index, "reason"] = [reason_text(group.loc[i], deltas.loc[i], medians) for i in group.index]
    return picks


importance = global_importance(main_exp["data"])
ml_picks_explained = add_reasons(main_exp)
last_date = ml_picks_explained["date"].max()
latest_picks = ml_picks_explained[ml_picks_explained["date"] == last_date]

print("Overall feature importance (permutation, higher = the model leans on it more):")
display(importance.rename(index=FEATURE_EN).to_frame("importance"))
print(f"\nMost recent session ({last_date.date()}) — the model's picks and the reason for each:")
display(latest_picks[["market", "rank", "ticker", "p_success", "gross_pct", "reason"]])

fig, ax = plt.subplots(figsize=(8, 3.8))
ordered = importance.rename(index=FEATURE_EN)[::-1]
ax.barh(ordered.index, ordered.values, color=["#2a6f97" if v > 0 else "#b8c0c8" for v in ordered.values])
ax.set_title(f"What the model leans on (permutation importance) — {main_exp['label']}")
ax.axvline(0, color="black", lw=0.6)
plt.tight_layout()
plt.show()
