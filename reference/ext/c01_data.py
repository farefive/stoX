ML_UNIVERSE_US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO", "INTC", "CRM",
                  "ORCL", "ADBE", "QCOM", "MU", "PYPL", "UBER", "COIN", "PLTR", "SHOP", "BA", "DIS", "JPM", "XOM"]
ML_UNIVERSE_TASI = ["2222.SR", "1120.SR", "2010.SR", "7010.SR", "1180.SR", "2082.SR", "7200.SR", "1211.SR", "2280.SR",
                    "1150.SR", "1010.SR", "1060.SR", "1050.SR", "5110.SR", "2020.SR", "2350.SR", "4190.SR", "4001.SR",
                    "2310.SR", "2290.SR"]
ML_UNIVERSE = ML_UNIVERSE_US + ML_UNIVERSE_TASI     # >= 20 liquid names per market, as the spec recommends (section 3.5)

DISPLAY_NAMES = {   # English display names (used by reports only; Yahoo symbols stay the keys)
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "AVGO": "Broadcom",
    "INTC": "Intel",
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "QCOM": "Qualcomm",
    "MU": "Micron",
    "PYPL": "PayPal",
    "UBER": "Uber",
    "COIN": "Coinbase",
    "PLTR": "Palantir",
    "SHOP": "Shopify",
    "BA": "Boeing",
    "DIS": "Disney",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "2222.SR": "Saudi Aramco",
    "1120.SR": "Al Rajhi Bank",
    "2010.SR": "SABIC",
    "7010.SR": "stc",
    "1180.SR": "Saudi National Bank",
    "2082.SR": "ACWA Power",
    "7200.SR": "Almoammar Information Systems",
    "1211.SR": "Ma'aden",
    "2280.SR": "Almarai",
    "1150.SR": "Alinma Bank",
    "1010.SR": "Riyad Bank",
    "1060.SR": "Saudi Awwal Bank",
    "1050.SR": "Banque Saudi Fransi",
    "5110.SR": "Saudi Electricity",
    "2020.SR": "SABIC Agri-Nutrients",
    "2350.SR": "Saudi Kayan",
    "4190.SR": "Jarir",
    "4001.SR": "Othaim Markets",
    "2310.SR": "Sipchem",
    "2290.SR": "Yansab",
}


def download_daily(ticker: str, years: int = 5) -> pd.DataFrame:
    """Daily OHLCV, split/dividend-adjusted (adjusting every column keeps intraday ratios intact)."""
    raw = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw[OHLCV].dropna()
    bars = bars[bars["Volume"] > 0]
    if len(bars) < 600:
        raise ValueError(f"only {len(bars)} daily bars for {ticker}")
    index = pd.DatetimeIndex(bars.index)
    bars.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    return bars


def synthetic_daily(ticker: str, market: Market, days: int = 1000, planted_effect: float = 0.0) -> pd.DataFrame:
    """Deterministic stand-in for daily bars: a random walk with volatility clustering and path-consistent OHLC.

    It contains NO market information. `planted_effect` > 0 is used ONLY by the labelled positive-control experiment
    (section 11): on days that open > +1 % above the previous close the intraday drift is raised by that fraction.
    """
    rng = np.random.default_rng(zlib.crc32(ticker.encode()) + (7919 if planted_effect else 0))
    weekmask = "Sun Mon Tue Wed Thu" if market is TASI else "Mon Tue Wed Thu Fri"
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days, freq="C", weekmask=weekmask)
    n, steps = len(dates), 78
    log_sigma = np.zeros(n)                                   # persistent volatility regime (volatility clusters)
    shocks = rng.normal(0, 0.15, n)
    for t in range(1, n):
        log_sigma[t] = 0.92 * log_sigma[t - 1] + shocks[t]
    sigma = rng.uniform(0.012, 0.030) * np.exp(log_sigma)     # daily volatility, differs per stock
    gap = sigma * 0.35 * rng.standard_t(4, n) / np.sqrt(2)
    drift = planted_effect * (gap > 0.01) / steps
    path = np.cumsum(rng.normal(0, 1, (n, steps)) * (sigma * 0.9 / np.sqrt(steps))[:, None] + drift[:, None], axis=1)
    close = rng.uniform(30, 300) * np.exp(np.cumsum(gap + path[:, -1]))
    open_ = np.r_[close[0] * np.exp(-(gap[0] + path[0, -1])), close[:-1]] * np.exp(gap)
    high = open_ * np.exp(np.maximum(path.max(axis=1), 0))
    low = open_ * np.exp(np.minimum(path.min(axis=1), 0))
    volume_ar = np.zeros(n)
    volume_shocks = rng.normal(0, 0.35, n)
    for t in range(1, n):
        volume_ar[t] = 0.8 * volume_ar[t - 1] + volume_shocks[t]
    volume = np.exp(rng.uniform(11, 16)) * np.exp(0.6 * log_sigma + 0.5 * volume_ar)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.round()}, index=dates)


def load_daily(tickers: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    data, sources = {}, {}
    for ticker in tickers:
        try:
            if yf is None or FORCE_FALLBACK_DATA:
                raise ImportError("yfinance unavailable or fallback forced")
            data[ticker], sources[ticker] = download_daily(ticker), "yfinance (real)"
        except Exception as exc:
            data[ticker] = synthetic_daily(ticker, market_of(ticker))
            sources[ticker] = f"SYNTHETIC ({type(exc).__name__})"
    return data, sources


def training_universe(daily: Dict[str, pd.DataFrame], sources: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Never train on a mix: if any real series exist, keep only the real ones; otherwise the whole run is synthetic."""
    real = {t: d for t, d in daily.items() if sources[t].startswith("yfinance")}
    return real if real else daily


daily_all, daily_sources = load_daily(ML_UNIVERSE)
daily_data = training_universe(daily_all, daily_sources)
DATA_MODE = "REAL" if all(daily_sources[t].startswith("yfinance") for t in daily_data) else "SYNTHETIC"
print(f"AI experiment data mode: {DATA_MODE} — {len(daily_data)} tickers "
      f"({sum(market_of(t) is US for t in daily_data)} US, {sum(market_of(t) is TASI for t in daily_data)} TASI), "
      f"{min(len(d) for d in daily_data.values())}-{max(len(d) for d in daily_data.values())} daily bars each")
if DATA_MODE == "SYNTHETIC":
    print("WARNING: synthetic data contains no market information — any 'result' below only proves the machinery works.")
