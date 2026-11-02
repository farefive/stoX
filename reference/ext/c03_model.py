from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    """Small, regularised gradient-boosted trees (same family as LightGBM/XGBoost; shallow on purpose to resist overfitting)."""
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def walk_forward_predict(dataset: pd.DataFrame, features: List[str] = ML_FEATURES, initial_days: int = 250,
                         retrain_every: int = 21, embargo_days: int = 1, seed: int = 0):
    """Expanding-window walk-forward: every block of `retrain_every` sessions is scored by a model that only ever saw
    sessions that ended at least `embargo_days` before the block starts. Returns (probabilities, list of block records)."""
    dates = dataset.index.get_level_values("date").unique().sort_values()
    date_of_row = dataset.index.get_level_values("date")
    proba = pd.Series(np.nan, index=dataset.index, name="p_success")
    block_of_row = pd.Series(-1, index=dataset.index, name="block")
    blocks = []
    for block_id, start in enumerate(range(initial_days, len(dates), retrain_every)):
        block_dates = dates[start:start + retrain_every]
        train_mask = date_of_row <= dates[start - embargo_days - 1]
        test_mask = date_of_row.isin(block_dates)
        model = make_model(seed).fit(dataset.loc[train_mask, features], dataset.loc[train_mask, "success"])
        proba[test_mask] = model.predict_proba(dataset.loc[test_mask, features])[:, 1]
        block_of_row[test_mask] = block_id
        blocks.append({"block": block_id, "model": model, "trained_through": dates[start - embargo_days - 1],
                       "first_test_date": block_dates[0], "last_test_date": block_dates[-1],
                       "n_train_rows": int(train_mask.sum())})
    return proba, block_of_row, blocks


def pick_top_k(scored: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    """Top-k per (date, market); ties broken by ticker A->Z so the result is fully deterministic."""
    flat = scored.reset_index().sort_values(["date", "market", score_col, "ticker"], ascending=[True, True, False, True])
    flat["rank"] = flat.groupby(["date", "market"]).cumcount() + 1
    return flat[flat["rank"] <= top_k].reset_index(drop=True)


def pick_metrics(picks: pd.DataFrame, trading_days: pd.DatetimeIndex) -> dict:
    metrics, _, _ = performance_report(picks, trading_days)
    metrics["hit_rate_pct"] = 100 * picks["success"].mean()
    return metrics


def daily_mean_return(picks: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    return picks.groupby("date")["net_pct"].mean().reindex(days, fill_value=0.0)


def bootstrap_ci(diff: pd.Series, reps: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95 % bootstrap interval for the mean of a daily difference series (resampling whole days)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff.to_numpy(), size=(reps, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run_experiment(daily: Dict[str, pd.DataFrame], cfg: DayTradingConfig = CFG, label: str = "main") -> dict:
    """The complete walk-forward study: ML ranking vs rule-based ranking vs the whole universe."""
    data = build_dataset(daily, cfg)
    proba, block_of_row, blocks = walk_forward_predict(data)
    data = data.join([proba, block_of_row])
    test = data[data["block"] >= 0].copy()
    days = pd.DatetimeIndex(sorted(test.index.get_level_values("date").unique()))

    ml_picks = pick_top_k(test, "p_success", cfg.top_k)

    shared = {w: cfg.weights[w] for w in ("rvol", "atr_pct", "gap_score")}      # factors a daily bar can still provide
    rule_cfg = replace(cfg, weights={k: v / sum(shared.values()) for k, v in shared.items()})
    rule_rank = build_baskets(test.drop(columns=["p_success", "block"]), rule_cfg)
    rule_picks = rule_rank[rule_rank["selected"]].reset_index(drop=True)

    universe = test.reset_index()
    results = {"ML ranker": pick_metrics(ml_picks, days), "Rule-based (spec weights)": pick_metrics(rule_picks, days),
               "Whole universe (no ranking)": pick_metrics(universe, days)}
    ml_daily, base_daily = daily_mean_return(ml_picks, days), daily_mean_return(universe, days)
    rule_daily = daily_mean_return(rule_picks, days)
    ic = pd.Series({key: group["p_success"].corr(group["net_pct"], method="spearman")
                    for key, group in test.groupby(["date", "market"])}).dropna()
    return {"label": label, "data": data, "test": test, "days": days, "blocks": blocks, "ml_picks": ml_picks,
            "rule_picks": rule_picks, "metrics": pd.DataFrame(results),
            "ml_vs_universe_ci": bootstrap_ci(ml_daily - base_daily), "ml_vs_rule_ci": bootstrap_ci(ml_daily - rule_daily),
            "rank_ic_mean": float(ic.mean()), "rank_ic_positive_share": float((ic > 0).mean()),
            "base_rate_pct": 100 * float(test["success"].mean())}


def show_experiment(exp: dict) -> None:
    print(f"[{exp['label']}] out-of-sample window: {exp['days'][0].date()} -> {exp['days'][-1].date()} "
          f"({len(exp['days'])} sessions, {len(exp['blocks'])} monthly re-trainings, top {CFG.top_k} per market)")
    display(exp["metrics"].loc[["trades", "hit_rate_pct", "win_rate_pct", "avg_trade_pct", "profit_factor",
                                "sharpe_annualised", "max_drawdown_pct", "total_return_pct"]])
    lo, hi = exp["ml_vs_universe_ci"]
    verdict = ("significantly BETTER than the universe average" if lo > 0 else
               "significantly WORSE than the universe average" if hi < 0 else
               "NOT distinguishable from luck (the interval includes 0)")
    print(f"ML minus whole-universe average net return/day: 95% bootstrap interval [{lo:+.3f}%, {hi:+.3f}%] -> {verdict}")
    lo2, hi2 = exp["ml_vs_rule_ci"]
    print(f"ML minus rule-based: [{lo2:+.3f}%, {hi2:+.3f}%];  rank information coefficient (prob vs result): "
          f"{exp['rank_ic_mean']:+.4f} (positive on {exp['rank_ic_positive_share'] * 100:.0f}% of days)")


main_exp = run_experiment(daily_data, CFG, label=f"{DATA_MODE} data")
show_experiment(main_exp)

fig, ax = plt.subplots(figsize=(10, 3.6))
for name, picks in [("ML ranker", main_exp["ml_picks"]), ("Rule-based", main_exp["rule_picks"]),
                    ("Whole universe", main_exp["test"].reset_index())]:
    curve = (1 + daily_mean_return(picks, main_exp["days"]) / 100).cumprod()
    ax.plot(curve.index, curve.values, label=name, lw=2.2 if name == "ML ranker" else 1.4)
ax.set_title(f"Out-of-sample growth of 1.0 (net of costs) — {main_exp['label']}")
ax.legend()
plt.tight_layout()
plt.show()
