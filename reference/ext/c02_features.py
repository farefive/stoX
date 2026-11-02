HORIZONS = (5, 10, 21, 252)    # trading days ≈ 1 week, 2 weeks, 1 month, 1 year (the "timeframes" asked for in review)
ML_FEATURES = ["gap_pct", "rvol", "atr_pct", "spread_pct", "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_252d",
               "vol_21d", "dist_high_252d", "is_tasi"]


def daily_features(bars: pd.DataFrame, cfg: DayTradingConfig = CFG) -> pd.DataFrame:
    """Pre-open feature row for every session t. Only the OPEN of day t and data up to day t-1 are used (no look-ahead)."""
    o, h, l, c, v = (bars[col] for col in OHLCV)
    prev_close = c.shift(1)
    true_range = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    day_return = c.pct_change()
    out = pd.DataFrame(index=bars.index)
    out["gap_pct"] = (o / prev_close - 1) * 100                                   # known the moment the market opens
    out["rvol"] = v.shift(1) / v.shift(2).rolling(cfg.lookback_days).mean()       # prior session's relative volume
    out["atr_pct"] = true_range.rolling(cfg.atr_days).mean().shift(1) / prev_close * 100
    out["spread_pct"] = corwin_schultz_spread(h, l).rolling(cfg.lookback_days).mean().shift(1) * 100
    out["ret_1d"] = day_return.shift(1) * 100
    for n in HORIZONS:
        out[f"ret_{n}d"] = (c.shift(1) / c.shift(1 + n) - 1) * 100                # momentum over the last n sessions
    out["vol_21d"] = day_return.shift(1).rolling(21).std() * 100
    out["dist_high_252d"] = (c.shift(1) / h.shift(1).rolling(252).max() - 1) * 100
    return out.replace([np.inf, -np.inf], np.nan)


AI_SUCCESS_PCT = 2.0   # "success" = the stock closed at least this far above its open (low end of the 2-3 % brief)


def daily_outcome(bars: pd.DataFrame, cfg: DayTradingConfig = CFG) -> pd.DataFrame:
    """Simplest unambiguous day trade: buy at the open, sell at the close.

    Daily bars cannot show whether an intraday target or stop came first, so the long-history study deliberately avoids
    path-dependent rules; the faithful target/stop simulation stays in the 5-minute backtester (section 4).
    """
    gross = (bars["Close"] / bars["Open"] - 1) * 100
    success = (gross >= AI_SUCCESS_PCT).astype(int)
    return pd.DataFrame({"gross_pct": gross, "net_pct": gross - cfg.round_trip_cost_pct, "success": success,
                         "reason": np.where(success == 1, "SUCCESS", np.where(gross > 0, "GAIN", "LOSS"))},
                        index=bars.index)


def build_dataset(daily: Dict[str, pd.DataFrame], cfg: DayTradingConfig = CFG) -> pd.DataFrame:
    """One row per (date, ticker): features known pre-open + what happened that day. Rows with missing features are dropped."""
    tables = []
    for ticker, bars in daily.items():
        table = daily_features(bars, cfg).join(daily_outcome(bars, cfg))
        table["open"], table["close"] = bars["Open"], bars["Close"]
        table["ticker"], table["market"] = ticker, market_of(ticker).name
        table["is_tasi"] = int(market_of(ticker) is TASI)
        tables.append(table)
    data = pd.concat(tables)
    data.index.name = "date"
    data = data.dropna(subset=ML_FEATURES).reset_index().set_index(["date", "ticker"]).sort_index()
    return data


def assert_no_lookahead(daily: Dict[str, pd.DataFrame], sample: int = 3) -> None:
    """Structural leakage test: features computed on history cut at day t must equal those computed with the future present."""
    for ticker in list(daily)[:sample]:
        bars = daily[ticker]
        full = daily_features(bars)
        for cut in (400, 600, len(bars) - 5):
            part = daily_features(bars.iloc[:cut])
            pd.testing.assert_frame_equal(part, full.iloc[:cut], check_exact=False, rtol=1e-9, atol=1e-9)


assert_no_lookahead(daily_data)
dataset = build_dataset(daily_data)
print(f"Dataset: {len(dataset):,} rows ({dataset.index.get_level_values('date').nunique()} sessions x up to "
      f"{len(daily_data)} tickers), {len(ML_FEATURES)} features.  Look-ahead test: PASSED")
print(f"Base rate: a stock closes at least +{AI_SUCCESS_PCT}% above its open on {dataset['success'].mean() * 100:.1f}% "
      f"of ticker-days (average open-to-close change {dataset['gross_pct'].mean():+.3f}%)")
display(dataset[ML_FEATURES + ["success", "net_pct"]].tail(5))
