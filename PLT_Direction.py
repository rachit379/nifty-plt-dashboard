"""
PLT_Direction.py

NIFTY Directional Bias Engine D(t)

This module estimates how *directionally tilted* the options complex is,
based mainly on index option OI patterns.

Definitions
-----------
D(t): Directional bias score in [-1, 1]
      -1   = strongly bearish positioning
       0   = neutral / balanced
      +1   = strongly bullish positioning

Components (all in [-1, 1]):

- far_oi_bias:
    OI balance between far OTM puts (downside region) and far OTM calls
    (upside region). Positive => more far OTM put OI (bullish / downside
    cushioning); negative => more far OTM call OI (bearish / upside overwriting).

- near_oi_bias:
    OI balance between near-OTM puts (just below spot) and near-OTM calls
    (just above spot). Positive => more put OI; negative => more call OI.

- pcr_bias:
    PCR(OI) deviation from 1, mapped to [-1, 1]. Positive => PCR > 1
    (put-heavy, often bullish-consensus in index world). Negative => PCR < 1
    (call-heavy, often bearish / overwriting).

You can call `get_nifty_direction_state()` to get:
- direction_features (raw),
- directional_score D(t),
- directional_components,
- directional_categories (value + label + one-line explanation).
"""

from typing import Dict, Any, Tuple
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

from PLT_Critical import (
    get_nifty_history_yf,
    fetch_nifty_derivatives_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_minus1_1(x) -> float:
    """Safely clip any numeric to [-1, 1]; NaN stays NaN."""
    if x is None:
        return np.nan
    try:
        if np.isnan(x):
            return np.nan
    except TypeError:
        return np.nan
    return float(np.clip(x, -1.0, 1.0))


def _bucket_direction(x: float) -> str:
    """
    Map a directional score in [-1,1] to a category label.

    Returns:
        'Strongly Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strongly Bullish', or 'N/A'
    """
    if x is None:
        return "N/A"
    try:
        if np.isnan(x):
            return "N/A"
    except TypeError:
        return "N/A"

    if x <= -0.6:
        return "Strongly Bearish"
    elif x <= -0.2:
        return "Bearish"
    elif x < 0.2:
        return "Neutral"
    elif x < 0.6:
        return "Bullish"
    else:
        return "Strongly Bullish"


def _category_explanation(metric: str, category: str) -> str:
    """
    Return a one-line explanation for a given metric/category combo.

    metric: "D", "far_oi_bias", "near_oi_bias", "pcr_bias"
    category: one of the direction labels or "N/A".
    """
    if category == "N/A":
        return "Not enough data to interpret this metric."

    # Overall D(t)
    if metric == "D":
        if category == "Strongly Bearish":
            return "Positioning is strongly skewed to the bearish side."
        if category == "Bearish":
            return "Positioning leans bearish, but not extremely so."
        if category == "Neutral":
            return "Positioning is broadly balanced between bullish and bearish."
        if category == "Bullish":
            return "Positioning leans bullish, with more downside cushioning than upside overwriting."
        if category == "Strongly Bullish":
            return "Positioning is strongly skewed bullish, with heavy downside cushioning vs upside caps."

    # Far OTM OI bias
    if metric == "far_oi_bias":
        if category == "Strongly Bearish":
            return "Far OTM call OI dominates far OTM put OI—market is heavily overwriting upside or hedging against rallies."
        if category == "Bearish":
            return "Far OTM call OI is larger than far OTM put OI, suggesting more upside overwriting than downside cushioning."
        if category == "Neutral":
            return "Far OTM put and call OI are broadly balanced."
        if category == "Bullish":
            return "Far OTM put OI dominates, indicating stronger downside cushioning than upside overwriting."
        if category == "Strongly Bullish":
            return "Far OTM put OI massively dominates far OTM call OI, consistent with strong downside cushioning / bullish tilt."

    # Near-OTM OI bias
    if metric == "near_oi_bias":
        if category == "Strongly Bearish":
            return "Near-OTM calls around spot dominate near-OTM puts—heavy upside supply / capping behaviour."
        if category == "Bearish":
            return "Near-OTM call OI is meaningfully larger than near-OTM put OI."
        if category == "Neutral":
            return "Near-OTM put and call OI near spot are roughly balanced."
        if category == "Bullish":
            return "Near-OTM put OI exceeds near-OTM call OI, consistent with downside support."
        if category == "Strongly Bullish":
            return "Near-OTM puts dominate strongly—significant put positioning just below spot."

    # PCR bias
    if metric == "pcr_bias":
        if category == "Strongly Bearish":
            return "Put–call ratio is very low—positioning is skewed to calls, often reflecting upside overwriting or lack of downside hedging."
        if category == "Bearish":
            return "Put–call ratio is below 1—positioning is somewhat skewed to calls."
        if category == "Neutral":
            return "Put–call ratio is near 1—no strong tilt toward puts or calls."
        if category == "Bullish":
            return "Put–call ratio is modestly above 1—more put positioning than calls, often consistent with bullish consensus in index options."
        if category == "Strongly Bullish":
            return "Put–call ratio is very high—heavy put positioning relative to calls, consistent with strong downside cushioning / bullish sentiment."

    # Fallback
    return "Category indicates the relative directional tilt for this metric."


# ---------------------------------------------------------------------------
# Core directional feature engine
# ---------------------------------------------------------------------------

def compute_directional_features(
    der_state: Dict[str, Any],
    last_close: float,
) -> Dict[str, Any]:
    """
    Compute directional features from the option chain + PCR.

    Inputs:
        der_state: dict from fetch_nifty_derivatives_state(...)
        last_close: latest NIFTY spot close

    Returns:
        {
          "last_close": float,
          "far_put_oi": float,
          "far_call_oi": float,
          "far_oi_bias": float,   # [-1,1]
          "near_put_oi": float,
          "near_call_oi": float,
          "near_oi_bias": float,  # [-1,1]
          "pcr_oi": float or None,
          "pcr_bias": float,      # [-1,1]
        }
    """
    oc = der_state.get("option_chain", None)
    pcr_oi = der_state.get("pcr_oi", None)

    far_put_oi = far_call_oi = 0.0
    near_put_oi = near_call_oi = 0.0
    far_oi_bias = np.nan
    near_oi_bias = np.nan
    pcr_bias = np.nan

    if isinstance(oc, pd.DataFrame) and not oc.empty and last_close and last_close > 0:
        oc_local = oc.copy()
        cols = oc_local.columns

        # Find OI columns
        ce_oi_col = next((c for c in cols if "CE_openInterest" in c), None)
        pe_oi_col = next((c for c in cols if "PE_openInterest" in c), None)

        if ce_oi_col and pe_oi_col:
            oc_local[ce_oi_col] = oc_local[ce_oi_col].fillna(0.0)
            oc_local[pe_oi_col] = oc_local[pe_oi_col].fillna(0.0)

            # Strikes
            if "strikePrice" in oc_local.columns:
                strikes = oc_local["strikePrice"].astype(float)
            else:
                # Assume index contains strikePrice
                strikes = oc_local.index.to_series().astype(float)

            # -----------------
            # Far OTM regions
            # -----------------
            # Far OTM puts: 90–98% of spot
            far_put_mask = (strikes >= last_close * 0.90) & (strikes <= last_close * 0.98)
            # Far OTM calls: 102–110% of spot
            far_call_mask = (strikes >= last_close * 1.02) & (strikes <= last_close * 1.10)

            far_put_oi = float(oc_local.loc[far_put_mask, pe_oi_col].sum())
            far_call_oi = float(oc_local.loc[far_call_mask, ce_oi_col].sum())

            denom_far = far_put_oi + far_call_oi
            if denom_far > 0:
                far_oi_bias = _clip_minus1_1((far_put_oi - far_call_oi) / denom_far)
            else:
                far_oi_bias = np.nan

            # -----------------
            # Near-OTM regions
            # -----------------
            # Near OTM puts: 98–100% of spot
            near_put_mask = (strikes >= last_close * 0.98) & (strikes <= last_close)
            # Near OTM calls: 100–102% of spot
            near_call_mask = (strikes >= last_close) & (strikes <= last_close * 1.02)

            near_put_oi = float(oc_local.loc[near_put_mask, pe_oi_col].sum())
            near_call_oi = float(oc_local.loc[near_call_mask, ce_oi_col].sum())

            denom_near = near_put_oi + near_call_oi
            if denom_near > 0:
                near_oi_bias = _clip_minus1_1((near_put_oi - near_call_oi) / denom_near)
            else:
                near_oi_bias = np.nan

    # PCR bias: treat PCR > 1 as put-heavy (bullish), PCR < 1 as call-heavy (bearish)
    if pcr_oi is not None:
        try:
            if not np.isnan(pcr_oi):
                # map PCR in [0.3, 1.7] roughly into [-1,1]
                # centred at 1.0
                raw = (pcr_oi - 1.0) / 0.7  # 1.7 => +1, 0.3 => -1
                pcr_bias = _clip_minus1_1(raw)
        except TypeError:
            pcr_bias = np.nan

    return {
        "last_close": float(last_close),
        "far_put_oi": far_put_oi,
        "far_call_oi": far_call_oi,
        "far_oi_bias": far_oi_bias,
        "near_put_oi": near_put_oi,
        "near_call_oi": near_call_oi,
        "near_oi_bias": near_oi_bias,
        "pcr_oi": pcr_oi,
        "pcr_bias": pcr_bias,
    }


def compute_directional_score(
    dir_feats: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Combine component directional biases into a single D(t) in [-1,1].

    Components:
      - far_oi_bias
      - near_oi_bias
      - pcr_bias
    """
    far_bias = dir_feats.get("far_oi_bias", np.nan)
    near_bias = dir_feats.get("near_oi_bias", np.nan)
    pcr_bias = dir_feats.get("pcr_bias", np.nan)

    components = {
        "far_oi_bias": far_bias,
        "near_oi_bias": near_bias,
        "pcr_bias": pcr_bias,
    }

    weights = {
        "far_oi_bias": 0.4,
        "near_oi_bias": 0.4,
        "pcr_bias": 0.2,
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

    D = num / den if den > 0 else np.nan
    return _clip_minus1_1(D), components


def categorize_direction(
    D: float,
    components: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict that holds numeric value, category, and explanation
    for D(t) and each component.

    Returns:
        {
          "D": {
            "value": <float>,
            "category": <str>,
            "explanation": <str>,
          },
          "far_oi_bias": {...},
          "near_oi_bias": {...},
          "pcr_bias": {...},
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Overall D
    D_cat = _bucket_direction(D)
    result["D"] = {
        "value": D,
        "category": D_cat,
        "explanation": _category_explanation("D", D_cat),
    }

    for name, val in components.items():
        cat = _bucket_direction(val)
        result[name] = {
            "value": val,
            "category": cat,
            "explanation": _category_explanation(name, cat),
        }

    return result


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def get_nifty_direction_state(
    years_hist: int = 1,
    index_ticker: str = "NIFTY",
) -> Dict[str, Any]:
    """
    High-level function:

    - Fetch NIFTY history via yfinance (for last close).
    - Fetch derivatives state via Bharat-sm-data.
    - Compute directional features.
    - Compute D(t) and component scores.
    - Attach categories + explanations.

    Returns:
        {
          "as_of": <latest cash close date as string>,
          "direction_features": {...},
          "directional_score": D,
          "directional_components": {...},
          "directional_categories": {...},
        }
    """
    # Spot history for last close
    price_df = get_nifty_history_yf(years=years_hist)
    if price_df.empty:
        raise ValueError("NIFTY history from yfinance is empty.")

    if "Adj Close" in price_df.columns:
        last_close = float(price_df["Adj Close"].iloc[-1])
    else:
        last_close = float(price_df["Close"].iloc[-1])

    latest_ts = price_df.index[-1]
    if hasattr(latest_ts, "strftime"):
        as_of_str = latest_ts.strftime("%Y-%m-%d")
    else:
        as_of_str = str(latest_ts)

    # Derivatives state (option chain + PCR)
    der_state = fetch_nifty_derivatives_state(index_ticker=index_ticker)

    dir_feats = compute_directional_features(der_state, last_close)
    D, components = compute_directional_score(dir_feats)
    categories = categorize_direction(D, components)

    return {
        "as_of": as_of_str,
        "direction_features": dir_feats,
        "directional_score": D,
        "directional_components": components,
        "directional_categories": categories,
    }


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = get_nifty_direction_state(years_hist=1, index_ticker="NIFTY")

    print("Directional snapshot as of:", state["as_of"])

    D = state["directional_score"]
    cats = state["directional_categories"]

    D_info = cats["D"]
    D_val = D_info["value"]
    D_cat = D_info["category"]
    D_expl = D_info["explanation"]

    if D_val is not None and not np.isnan(D_val):
        print("\n=== Directional Bias D(t) ===")
        print(f"D(t): {D_val:.2f}  [{D_cat}]  (-1 = strongly bearish, +1 = strongly bullish)")
        print(f"      -> {D_expl}")
    else:
        print("\nD(t): NaN  [N/A]")
        print(f"      -> {D_expl}")

    print("\nComponents:")
    for name in ["far_oi_bias", "near_oi_bias", "pcr_bias"]:
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
