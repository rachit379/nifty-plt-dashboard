"""
PLT_Shocks.py

NIFTY Shock & Trigger Engine S(t)

This module measures how "shocked" the market was on the latest trading day,
relative to its own recent history.

It uses:
- NIFTY daily OHLCV from yfinance (^NSEI) via PLT_Critical.get_nifty_history_yf()
- A rolling lookback window (e.g., last 60 trading days) to normalize today's move

Definitions
-----------
S(t): Shock Intensity for the latest day.
      0 = no meaningful shock, 1 = very large / extreme shock.

Components (all mapped to [0, 1]):

- gap_score:
    Size of the overnight gap (Open_t vs Close_{t-1}), relative to the
    recent distribution of gaps.

- intraday_score:
    Size of the intraday range (High_t vs Low_t), relative to recent ranges.

- volume_score:
    Volume spike vs recent history (log-volume z-score).

You can call `get_nifty_shock_state()` from other files to get:
- shock_features
- shock_score S(t)
- shock_components
- shock_categories (value + category + one-line explanation)
"""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Reuse the same history function from your C(t) module
from PLT_Critical import get_nifty_history_yf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_01(x, lo, hi):
    """
    Linearly map x in [lo, hi] -> [0, 1], clipped.
    Returns NaN if x is NaN or None.
    """
    if x is None:
        return np.nan
    try:
        if np.isnan(x):
            return np.nan
    except TypeError:
        # x is not a float-like; just bail
        return np.nan

    if hi == lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _bucket_score(x: float) -> str:
    """
    Map a score in [0,1] to a category label.
    If x is NaN or None -> 'N/A'.
    """
    if x is None:
        return "N/A"
    try:
        if np.isnan(x):
            return "N/A"
    except TypeError:
        return "N/A"

    if x < 0.33:
        return "Low"
    elif x < 0.66:
        return "Moderate"
    elif x < 0.85:
        return "Elevated"
    else:
        return "Critical"


def _category_explanation(metric: str, category: str) -> str:
    """
    Return a one-line explanation for a given metric/category combo.

    metric: "S", "gap_score", "intraday_score", "volume_score"
    category: "Low", "Moderate", "Elevated", "Critical", or "N/A"
    """

    if category == "N/A":
        return "Not enough data to interpret this metric."

    # Overall S(t)
    if metric == "S":
        if category == "Low":
            return "No meaningful shock—price and volume are well within recent norms."
        if category == "Moderate":
            return "Mild shock—moves and volumes are a bit larger than typical, but not extreme."
        if category == "Elevated":
            return "Significant shock—today’s move stands out versus recent history."
        if category == "Critical":
            return "Very large shock—price and/or volume are at extreme levels relative to recent history."

    # Gap
    if metric == "gap_score":
        if category == "Low":
            return "Overnight gap is small relative to typical recent gaps."
        if category == "Moderate":
            return "Overnight gap is noticeable but still within a normal range."
        if category == "Elevated":
            return "Overnight gap is large compared to most recent days."
        if category == "Critical":
            return "Overnight gap is among the largest seen in the recent lookback window."

    # Intraday range
    if metric == "intraday_score":
        if category == "Low":
            return "Intraday range is tight—no unusual swing during the session."
        if category == "Moderate":
            return "Intraday range is somewhat wide versus typical days."
        if category == "Elevated":
            return "Intraday range is significantly wider than usual."
        if category == "Critical":
            return "Intraday range is extremely wide—very volatile session."

    # Volume spike
    if metric == "volume_score":
        if category == "Low":
            return "Volume is around or below its recent average."
        if category == "Moderate":
            return "Volume is somewhat above average."
        if category == "Elevated":
            return "Volume is clearly elevated—strong participation or forced activity."
        if category == "Critical":
            return "Volume is at extreme levels—very heavy trading relative to the recent past."

    # Fallback
    return "Category indicates relative shock intensity for this metric."


# ---------------------------------------------------------------------------
# Core shock feature engine
# ---------------------------------------------------------------------------
def compute_shock_features(
    price_df: pd.DataFrame,
    window_days: int = 60,
) -> Dict[str, Any]:
    """
    Compute shock-related features for the MOST RECENT day in price_df.

    Uses:
    - gap:      (Open_t - Close_{t-1}) / Close_{t-1}
    - intraday: (High_t - Low_t) / Close_{t-1}
    - volume:   log(Volume_t) vs rolling window

    window_days:
        Number of prior days used as a comparison window for "normal" behaviour.
    """
    df = price_df.copy()

    # --- 1) Handle MultiIndex columns from yfinance, if any ---
    if isinstance(df.columns, pd.MultiIndex):
        # Use only the first level (e.g., 'Open', 'High', ...)
        df.columns = [c[0] for c in df.columns]

    # --- 2) Normalize OHLCV column names (case-insensitive) ---
    col_map = {c.lower(): c for c in df.columns}
    required_lower = ["open", "high", "low", "close", "volume"]

    missing = [name for name in required_lower if name not in col_map]
    if missing:
        raise KeyError(
            f"Missing required OHLCV columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Rename actual columns to canonical form
    rename_dict = {col_map["open"]: "Open",
                   col_map["high"]: "High",
                   col_map["low"]: "Low",
                   col_map["close"]: "Close",
                   col_map["volume"]: "Volume"}
    df = df.rename(columns=rename_dict)

    # --- 3) Now we can safely drop NA rows on canonical columns ---
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(df) < window_days + 2:
        raise ValueError(f"Need at least {window_days + 2} rows of data, got {len(df)}.")

    # Basic derived columns
    df["prev_close"] = df["Close"].shift(1)
    df["gap"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_range"] = (df["High"] - df["Low"]) / df["prev_close"]
    df["volume_log"] = np.log(df["Volume"].replace(0, np.nan))

    df = df.dropna(subset=["gap", "intraday_range", "volume_log"])

    # Latest day
    latest = df.iloc[-1]
    # Lookback window (exclude latest row)
    hist = df.iloc[-(window_days + 1):-1]

    # --- Gap features ---
    gaps_hist_abs = hist["gap"].abs()
    gap_today = float(latest["gap"])
    gap_abs_today = abs(gap_today)

    gap_median = float(np.median(gaps_hist_abs))
    gap_p95 = float(np.percentile(gaps_hist_abs, 95)) if len(gaps_hist_abs) > 0 else gap_median

    gap_score = _norm_01(gap_abs_today, gap_median, gap_p95)

    # --- Intraday range features ---
    ranges_hist = hist["intraday_range"]
    intraday_range_today = float(latest["intraday_range"])

    range_median = float(np.median(ranges_hist))
    range_p95 = float(np.percentile(ranges_hist, 95)) if len(ranges_hist) > 0 else range_median

    intraday_score = _norm_01(intraday_range_today, range_median, range_p95)

    # --- Volume features ---
    vol_hist = hist["volume_log"]
    volume_today = float(latest["Volume"])
    volume_log_today = float(latest["volume_log"])

    vol_mu = float(vol_hist.mean())
    vol_sigma = float(vol_hist.std(ddof=0))

    if vol_sigma > 0:
        vol_z = (volume_log_today - vol_mu) / vol_sigma
        # Map z to [0,1] with a sigmoid; positive z => >0.5
        volume_score = 1.0 / (1.0 + np.exp(-vol_z))
    else:
        vol_z = np.nan
        volume_score = np.nan

    return {
        "window_days": window_days,

        "gap_today": gap_today,
        "gap_abs_today": gap_abs_today,
        "gap_median": gap_median,
        "gap_p95": gap_p95,
        "gap_score": gap_score,

        "intraday_range_today": intraday_range_today,
        "intraday_range_median": range_median,
        "intraday_range_p95": range_p95,
        "intraday_score": intraday_score,

        "volume_today": volume_today,
        "volume_log_today": volume_log_today,
        "volume_log_mu": vol_mu,
        "volume_log_sigma": vol_sigma,
        "volume_z": float(vol_z) if not np.isnan(vol_z) else np.nan,
        "volume_score": float(volume_score) if not np.isnan(volume_score) else np.nan,
    }

def compute_shock_score(
    shock_feats: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Combine component shock scores into a single S(t) in [0,1].

    Components:
      - gap_score
      - intraday_score
      - volume_score
    """
    gap_score = shock_feats.get("gap_score", np.nan)
    intraday_score = shock_feats.get("intraday_score", np.nan)
    volume_score = shock_feats.get("volume_score", np.nan)

    components = {
        "gap_score": gap_score,
        "intraday_score": intraday_score,
        "volume_score": volume_score,
    }

    weights = {
        "gap_score": 0.4,
        "intraday_score": 0.3,
        "volume_score": 0.3,
    }

    num = 0.0
    den = 0.0
    for k, s in components.items():
        if s is not None:
            try:
                if not np.isnan(s):
                    w = weights[k]
                    num += w * s
                    den += w
            except TypeError:
                continue

    S = num / den if den > 0 else np.nan
    return S, components


def categorize_shocks(S: float, components: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict that holds numeric value, category, and a one-line explanation
    for S(t) and each component.

    Returns:
        {
          "S": {
            "value": <float>,
            "category": <str>,
            "explanation": <str>,
          },
          "gap_score": {...},
          "intraday_score": {...},
          "volume_score": {...},
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Overall S
    S_cat = _bucket_score(S)
    result["S"] = {
        "value": S,
        "category": S_cat,
        "explanation": _category_explanation("S", S_cat),
    }

    # Sub-components
    for name, val in components.items():
        cat = _bucket_score(val)
        result[name] = {
            "value": val,
            "category": cat,
            "explanation": _category_explanation(name, cat),
        }

    return result


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def get_nifty_shock_state(
    years_hist: int = 1,
    window_days: int = 60,
) -> Dict[str, Any]:
    """
    High-level function:

    - Fetch ~`years_hist` of NIFTY daily data via yfinance.
    - Compute shock features for the latest day using `window_days` of history.
    - Compute S(t) and component scores.
    - Attach categories + explanations.

    Returns:
        {
          "as_of": <latest date as string>,
          "shock_features": {...},
          "shock_score": <float>,
          "shock_components": {...},
          "shock_categories": {...},
        }
    """
    price_df = get_nifty_history_yf(years=years_hist)
    if price_df.empty:
        raise ValueError("NIFTY history from yfinance is empty.")

    shock_feats = compute_shock_features(price_df, window_days=window_days)
    S, components = compute_shock_score(shock_feats)
    categories = categorize_shocks(S, components)

    latest_ts = price_df.index[-1]
    if hasattr(latest_ts, "strftime"):
        as_of_str = latest_ts.strftime("%Y-%m-%d")
    else:
        as_of_str = str(latest_ts)

    return {
        "as_of": as_of_str,
        "shock_features": shock_feats,
        "shock_score": S,
        "shock_components": components,
        "shock_categories": categories,
    }


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = get_nifty_shock_state(years_hist=1, window_days=60)

    print("Shock snapshot as of:", state["as_of"])

    S = state["shock_score"]
    cats = state["shock_categories"]

    # Overall S(t)
    S_info = cats["S"]
    S_val = S_info["value"]
    S_cat = S_info["category"]
    S_expl = S_info["explanation"]

    if S_val is not None and not np.isnan(S_val):
        print(f"\n=== Shock Score S(t) ===")
        print(f"S(t): {S_val:.2f}  [{S_cat}]  (0 = no shock, 1 = extreme shock)")
        print(f"      -> {S_expl}")
    else:
        print("\nS(t): NaN  [N/A]")
        print(f"      -> {S_expl}")

    print("\nComponents:")
    for name in ["gap_score", "intraday_score", "volume_score"]:
        info = cats.get(name, {"value": np.nan, "category": "N/A", "explanation": "N/A"})
        val = info["value"]
        cat = info["category"]
        expl = info["explanation"]

        if val is not None and not np.isnan(val):
            print(f"  {name}: {val:.2f}  [{cat}]")
            print(f"      -> {expl}")
        else:
            print(f"  {name}: NaN  [{cat}]")
            print(f"      -> {expl}")
