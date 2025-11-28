"""
PLT_Levels.py

Static "Levels & Setups" arm L(t).

Goal:
- For a given ticker (index or stock), compute key breakout/breakdown levels
  and a simple readiness score for long/short setups.

Inputs:
- Daily OHLC from yfinance.

Outputs per ticker:
- levels: prev_high, prev_low, close, 20d high/low, 52w high/low, ATR, MAs.
- L_long: breakout readiness score in [0, 1] for long side.
- L_short: breakdown readiness score in [0, 1] for short side.
- Categories for each: Cold / Watching / Hot + one-line explanation.

Later:
- You can extend this module to:
  - ingest your RSS feed (ind-stock-rss),
  - add a news_heat component to long/short setup scores,
  - and integrate L(t) into the regime engine.
"""

from datetime import date, timedelta
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import logging

# Set global default (optional)
logging.basicConfig(level=logging.INFO)

# Silence noisy libs
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)

# -------------------------------------------------------------------
# Normalise OHLC columns from yfinance
# -------------------------------------------------------------------

def _normalize_ohlc_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a yfinance DataFrame (possibly with MultiIndex columns or
    weird casing) and return a DataFrame with simple columns:
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    where available.
    """
    df = price_df.copy()

    # 1) Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            # col is a tuple; pick the part that looks like OHLCV field
            chosen = None
            for part in col:
                if str(part).lower() in (
                    "open", "high", "low", "close", "adj close", "volume"
                ):
                    chosen = str(part)
                    break
            if chosen is None:
                # fallback to first element
                chosen = str(col[0])
            flat_cols.append(chosen)
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    # 2) Map case-insensitively to canonical names
    lower_to_col = {c.lower(): c for c in df.columns}

    out = pd.DataFrame(index=df.index)

    for canonical in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        key = canonical.lower()
        if key in lower_to_col:
            out[canonical] = df[lower_to_col[key]]

    return out

# -------------------------------------------------------------------
# 1) Generic history fetch via yfinance
# -------------------------------------------------------------------

def get_history_yf(
    ticker: str,
    years: int = 1,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance for the last `years` years.

    Works for:
      - Index: "^NSEI", "^NSEBANK", etc.
      - Stocks: "RELIANCE.NS", "TCS.NS", etc.

    Returns:
        DataFrame indexed by Date with columns:
        [Open, High, Low, Close, Adj Close, Volume]
    """
    end = date.today()
    start = end - timedelta(days=365 * years)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    df = df.sort_index()
    return df


# -------------------------------------------------------------------
# 2) ATR, MAs and breakout levels
# -------------------------------------------------------------------

def compute_atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """
    Compute a simple ATR over `window` days.

    Assumes df has columns: High, Low, Close.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def compute_breakout_levels(
    price_df: pd.DataFrame,
    breakout_lookback: int = 20,
    atr_window: int = 14,
) -> Dict[str, Any]:
    """
    Compute basic breakout/breakdown levels and volatility stats.

    Returns:
        {
          "close": float,
          "prev_close": float,
          "prev_high": float,
          "prev_low": float,
          "high_20d": float,
          "low_20d": float,
          "high_52w": float,
          "low_52w": float,
          "atr_14": float,
          "ma_20": float,
          "ma_50": float,
        }
    """
    # Normalise columns first (handles MultiIndex / casing)
    df = _normalize_ohlc_df(price_df)

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Normalised price data missing required columns: {missing}")

    df = df.dropna(subset=required_cols)

    if df.empty or len(df) < max(breakout_lookback, atr_window, 50):
        raise ValueError("Not enough data to compute breakout levels.")

    close_series = df["Close"]

    close = float(close_series.iloc[-1])
    prev_close = float(close_series.iloc[-2])
    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])

    # Short-term breakout range
    high_20d = float(df["High"].tail(breakout_lookback).max())
    low_20d = float(df["Low"].tail(breakout_lookback).min())

    # 52-week high/low (approx 252 trading days)
    high_52w = float(df["High"].tail(252).max())
    low_52w = float(df["Low"].tail(252).min())

    # ATR
    atr_series = compute_atr(df, window=atr_window)
    atr_14 = float(atr_series.iloc[-1])

    # Moving averages
    ma_20 = float(close_series.rolling(20).mean().iloc[-1])
    ma_50 = float(close_series.rolling(50).mean().iloc[-1])

    return {
        "close": close,
        "prev_close": prev_close,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "high_20d": high_20d,
        "low_20d": low_20d,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "atr_14": atr_14,
        "ma_20": ma_20,
        "ma_50": ma_50,
    }


# -------------------------------------------------------------------
# 3) Scoring: How "ready" is this for a breakout/breakdown?
# -------------------------------------------------------------------

def _cold_hot_bucket(x: float) -> str:
    """
    Bucket a score in [0,1] into a label.

    <=0.33  -> Cold
    0.33-0.66 -> Watching
    >0.66  -> Hot
    """
    if x is None or np.isnan(x):
        return "N/A"
    if x <= 0.33:
        return "Cold"
    elif x <= 0.66:
        return "Watching"
    else:
        return "Hot"


def _proximity_score(
    price: float,
    level: float,
    max_distance_pct: float = 0.05,
) -> float:
    """
    Simple 0-1 score for how close price is to a level, symmetric:

    - 1.0 if price == level.
    - 0.0 if price is >= max_distance_pct away (either side).
    """
    if price <= 0 or level <= 0 or max_distance_pct <= 0:
        return 0.0

    dist_pct = abs(price / level - 1.0)  # absolute percent distance
    if dist_pct >= max_distance_pct:
        return 0.0

    # linearly decay from 1 at 0% distance to 0 at max_distance_pct
    return float(1.0 - dist_pct / max_distance_pct)


def _trend_filter_score(ma_fast: float, ma_slow: float) -> float:
    """
    Very simple trend filter:

    - 1.0 if ma_fast > ma_slow (uptrend),
    - 0.0 if ma_fast <= ma_slow (down/flat).
    """
    if np.isnan(ma_fast) or np.isnan(ma_slow):
        return 0.5  # neutral if no info
    return 1.0 if ma_fast > ma_slow else 0.0


def compute_breakout_scores(
    levels: Dict[str, float],
    proximity_window_pct: float = 0.05,
    trend_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Compute long and short breakout readiness scores:

    - L_long: readiness for bullish breakout above recent highs.
    - L_short: readiness for bearish breakdown below recent lows.

    Both in [0,1], plus category + explanation strings.
    """
    close = levels["close"]
    high_20d = levels["high_20d"]
    low_20d = levels["low_20d"]
    ma_20 = levels["ma_20"]
    ma_50 = levels["ma_50"]

    # Proximity to breakout/breakdown levels
    prox_long = _proximity_score(close, high_20d, max_distance_pct=proximity_window_pct)
    prox_short = _proximity_score(close, low_20d, max_distance_pct=proximity_window_pct)

    # Trend filter (uptrend favours long, downtrend favours short)
    trend_up = _trend_filter_score(ma_20, ma_50)
    trend_down = 1.0 - trend_up

    # Combine
    long_score = (1.0 - trend_weight) * prox_long + trend_weight * trend_up
    short_score = (1.0 - trend_weight) * prox_short + trend_weight * trend_down

    # Clip to [0,1]
    long_score = float(np.clip(long_score, 0.0, 1.0))
    short_score = float(np.clip(short_score, 0.0, 1.0))

    # Buckets
    long_cat = _cold_hot_bucket(long_score)
    short_cat = _cold_hot_bucket(short_score)

    # Explanations
    long_expl = _explain_breakout("long", long_cat)
    short_expl = _explain_breakout("short", short_cat)

    return {
        "L_long": {
            "score": long_score,
            "category": long_cat,
            "explanation": long_expl,
        },
        "L_short": {
            "score": short_score,
            "category": short_cat,
            "explanation": short_expl,
        },
        "components": {
            "prox_long": prox_long,
            "prox_short": prox_short,
            "trend_up": trend_up,
        },
    }


def _explain_breakout(side: str, category: str) -> str:
    """
    One-liner explanations for long/short breakout readiness.
    """
    if category == "N/A":
        return "Not enough data to evaluate this setup."

    if side == "long":
        if category == "Cold":
            return "Price is far from recent highs or trend is not supportive—no urgent long-breakout setup."
        if category == "Watching":
            return "Price is approaching recent highs with a somewhat supportive trend—keep on radar for a long breakout."
        if category == "Hot":
            return "Price is very close to (or sitting on) recent highs in an uptrend—strong long-breakout candidate."
    else:  # short
        if category == "Cold":
            return "Price is far from recent lows or trend is not supportive—no urgent short-breakdown setup."
        if category == "Watching":
            return "Price is approaching recent lows with a somewhat supportive downtrend—keep on radar for a short breakdown."
        if category == "Hot":
            return "Price is very close to (or sitting on) recent lows in a downtrend—strong short-breakdown candidate."

    return "Category indicates how ready this instrument is for a breakout/breakdown."


# -------------------------------------------------------------------
# 4) High-level helpers / public API
# -------------------------------------------------------------------

def get_ticker_levels_state(
    ticker: str,
    years_hist: int = 1,
    breakout_lookback: int = 20,
    atr_window: int = 14,
    proximity_window_pct: float = 0.05,
    trend_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    High-level function for a single ticker.

    Steps:
      - Fetch OHLC from yfinance.
      - Compute breakout levels (20d, 52w, ATR, MAs).
      - Compute L_long and L_short scores and categories.

    Returns:
        {
          "ticker": <str>,
          "as_of": <last_date>,
          "levels": {...},
          "scores": {
            "L_long": {...},
            "L_short": {...},
            "components": {...},
          },
        }
    """
    hist = get_history_yf(ticker=ticker, years=years_hist, interval="1d")
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}.")

    levels = compute_breakout_levels(
        hist,
        breakout_lookback=breakout_lookback,
        atr_window=atr_window,
    )

    scores = compute_breakout_scores(
        levels,
        proximity_window_pct=proximity_window_pct,
        trend_weight=trend_weight,
    )

    return {
        "ticker": ticker,
        "as_of": hist.index[-1].strftime("%Y-%m-%d"),
        "levels": levels,
        "scores": scores,
    }


def get_universe_levels_state(
    tickers: List[str],
    years_hist: int = 1,
    breakout_lookback: int = 20,
    atr_window: int = 14,
    proximity_window_pct: float = 0.05,
    trend_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Convenience wrapper for multiple tickers.

    Returns:
        {
          "as_of": <date>,
          "tickers": {
            "<ticker>": { ... single-ticker state ... },
            ...
          }
        }
    """
    result: Dict[str, Any] = {"tickers": {}}
    last_dates = []

    for t in tickers:
        try:
            state = get_ticker_levels_state(
                ticker=t,
                years_hist=years_hist,
                breakout_lookback=breakout_lookback,
                atr_window=atr_window,
                proximity_window_pct=proximity_window_pct,
                trend_weight=trend_weight,
            )
            result["tickers"][t] = state
            last_dates.append(state["as_of"])
        except Exception as e:
            print(f"[WARN] get_universe_levels_state: failed for {t}: {e}")

    # as_of: max date across tickers
    if last_dates:
        result["as_of"] = max(last_dates)
    else:
        result["as_of"] = None

    return result


# -------------------------------------------------------------------
# 5) Manual test / demo
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example: NIFTY index + a couple of stocks
    universe = ["^NSEI", "RELIANCE.NS", "TCS.NS"]

    state = get_universe_levels_state(
        tickers=universe,
        years_hist=1,
        breakout_lookback=20,
        atr_window=14,
        proximity_window_pct=0.05,
        trend_weight=0.3,
    )

    print("Levels & Setups snapshot as of:", state["as_of"])
    for t, info in state["tickers"].items():
        lv = info["levels"]
        sc_long = info["scores"]["L_long"]
        sc_short = info["scores"]["L_short"]

        print(f"\nTicker: {t}  (as of {info['as_of']})")
        print(f"  Close: {lv['close']:.2f}")
        print(f"  20d High / Low: {lv['high_20d']:.2f} / {lv['low_20d']:.2f}")
        print(f"  ATR(14): {lv['atr_14']:.2f}")
        print(f"  MA20 / MA50: {lv['ma_20']:.2f} / {lv['ma_50']:.2f}")

        print(f"  L_long : {sc_long['score']:.2f} [{sc_long['category']}]")
        print(f"    -> {sc_long['explanation']}")
        print(f"  L_short: {sc_short['score']:.2f} [{sc_short['category']}]")
        print(f"    -> {sc_short['explanation']}")
