"""
PLT_Contagion.py

NIFTY Contagion / Co-movement Engine K(t)

This module measures how "coupled" / "contagious" the market is:
- Are a large share of stocks moving together in the same direction?
- Are many stocks making large moves on the same day?
- Are key indices (NIFTY, BANKNIFTY) moving in a highly correlated way?

Definitions
-----------
K(t): Contagion / Co-movement intensity for the latest day.
      0 = very low co-movement (idiosyncratic), 1 = very high co-movement (systemic).

Components (all mapped to [0, 1]):

- breadth_sync_score:
    Share of F&O universe advancing or declining in the dominant direction.
    High score = most stocks moved in the same direction (up or down).

- wide_move_score:
    Share of F&O universe with absolute return above a threshold (e.g. 1.5%).
    High score = many names had large moves simultaneously.

- index_corr_score:
    Correlation of daily returns between NIFTY (^NSEI) and BANKNIFTY (^NSEBANK)
    over a rolling lookback window (e.g. 60 days).
    High score = major indices are moving in lockstep.

You can call `get_nifty_contagion_state()` from other files to get:
- breadth_features,
- index_corr_features,
- contagion_score K(t),
- contagion_components,
- contagion_categories (value + category + one-line explanation).
"""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Reuse history from C(t) engine
from PLT_Critical import get_nifty_history_yf

# Bharat-sm-data
from Technical.NSE import NSE as TechNSE


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

    metric: "K", "breadth_sync_score", "wide_move_score", "index_corr_score"
    category: "Low", "Moderate", "Elevated", "Critical", or "N/A"
    """

    if category == "N/A":
        return "Not enough data to interpret this metric."

    # Overall K(t)
    if metric == "K":
        if category == "Low":
            return "Market moves look idiosyncratic—limited evidence of systemic co-movement."
        if category == "Moderate":
            return "Co-movement is at a normal level—some common factor, but not dominating."
        if category == "Elevated":
            return "Co-movement is strong—many assets are moving together, contagion risk is higher."
        if category == "Critical":
            return "Co-movement is extremely high—market is trading as one trade, systemic moves dominate."

    # Breadth sync
    if metric == "breadth_sync_score":
        if category == "Low":
            return "Advancers and decliners are fairly balanced—no dominant one-sided move across stocks."
        if category == "Moderate":
            return "One side (up or down) is ahead, but not overwhelmingly."
        if category == "Elevated":
            return "Most stocks moved in the same direction, suggesting strong common factor."
        if category == "Critical":
            return "An overwhelming majority of stocks moved in the same direction—very strong market-wide sweep."

    # Wide moves
    if metric == "wide_move_score":
        if category == "Low":
            return "Only a small fraction of stocks had large moves—most names were relatively quiet."
        if category == "Moderate":
            return "A noticeable share of stocks had large moves, but not extreme."
        if category == "Elevated":
            return "Many stocks had large moves—the market is seeing broad, significant price action."
        if category == "Critical":
            return "A very large share of stocks had big moves—broad, high-intensity market action."

    # Index correlation
    if metric == "index_corr_score":
        if category == "Low":
            return "Key indices are not moving in lockstep—sector/index behaviour is differentiated."
        if category == "Moderate":
            return "Indices are somewhat correlated, consistent with normal macro influence."
        if category == "Elevated":
            return "Indices are highly correlated—macro/common factors are dominating."
        if category == "Critical":
            return "Indices are extremely correlated—market is largely trading as a single risk-on/risk-off block."

    # Fallback
    return "Category indicates relative co-movement intensity for this metric."


# ---------------------------------------------------------------------------
# Breadth features from F&O universe
# ---------------------------------------------------------------------------

def compute_breadth_features_from_fo(
    index_label: str = "SECURITIES IN F&O",
    min_move_abs: float = 0.015,
) -> Dict[str, Any]:
    """
    Use Bharat-sm-data to pull the F&O universe and compute breadth metrics
    based on today's return (lastPrice vs previousClose).

    Returns:
        {
          "n_total": int,
          "n_adv": int,
          "n_dec": int,
          "n_flat": int,
          "n_wide": int,
          "adv_frac": float,
          "dec_frac": float,
          "wide_frac": float,
          "dominant_side": "adv" | "dec" | "balanced",
          "breadth_sync_score": float,
          "wide_move_score": float,
        }
    """
    tech = TechNSE()
    df = tech.get_equities_data_from_index(index=index_label)

    if df is None or df.empty:
        raise ValueError(f"get_equities_data_from_index('{index_label}') returned empty DataFrame.")

    df = df.copy()

    # Normalize column names
    col_map = {c.lower(): c for c in df.columns}

    last_price_col = None
    prev_close_col = None

    for key, col in col_map.items():
        if "lastprice" in key or key == "ltp":
            last_price_col = col
        if "previousclose" in key:
            prev_close_col = col

    if last_price_col is None or prev_close_col is None:
        raise KeyError(
            f"Could not find lastPrice/previousClose columns in F&O data. "
            f"Columns: {list(df.columns)}"
        )

    # Compute simple daily return
    lp = df[last_price_col].astype(float)
    pc = df[prev_close_col].astype(float).replace(0, np.nan)

    ret = (lp - pc) / pc
    df["ret"] = ret

    # Basic counts
    n_total = df["ret"].notna().sum()
    if n_total == 0:
        raise ValueError("No valid returns in F&O breadth data.")

    # Adv/dec/flat with a small neutral band
    adv = df["ret"] > +0.0
    dec = df["ret"] < -0.0
    flat = ~(adv | dec)

    n_adv = int(adv.sum())
    n_dec = int(dec.sum())
    n_flat = int(flat.sum())

    adv_frac = n_adv / n_total
    dec_frac = n_dec / n_total

    # Wide moves: abs(ret) >= min_move_abs
    wide = df["ret"].abs() >= min_move_abs
    n_wide = int(wide.sum())
    wide_frac = n_wide / n_total

    # Dominant side for breadth sync
    if adv_frac > dec_frac + 0.05:
        dominant = "adv"
        dominant_frac = adv_frac
    elif dec_frac > adv_frac + 0.05:
        dominant = "dec"
        dominant_frac = dec_frac
    else:
        dominant = "balanced"
        dominant_frac = max(adv_frac, dec_frac)

    # Map dominant fraction into a sync score:
    # - Around 50% => 0
    # - Around 80%+ => 1
    breadth_sync_score = _norm_01(dominant_frac, 0.5, 0.8)

    # Map wide_frac similarly: e.g., 20% => 0, 60%+ => 1
    wide_move_score = _norm_01(wide_frac, 0.2, 0.6)

    return {
        "n_total": n_total,
        "n_adv": n_adv,
        "n_dec": n_dec,
        "n_flat": n_flat,
        "n_wide": n_wide,
        "adv_frac": adv_frac,
        "dec_frac": dec_frac,
        "wide_frac": wide_frac,
        "dominant_side": dominant,
        "dominant_frac": dominant_frac,
        "breadth_sync_score": breadth_sync_score,
        "wide_move_score": wide_move_score,
        "min_move_abs": min_move_abs,
    }


# ---------------------------------------------------------------------------
# Index correlation features (NIFTY vs BANKNIFTY)
# ---------------------------------------------------------------------------

def compute_index_corr_features(
    years_hist: int = 1,
    lookback_days: int = 60,
) -> Dict[str, Any]:
    """
    Compute correlation of daily returns between NIFTY (^NSEI)
    and BANKNIFTY (^NSEBANK) over the last `lookback_days`.

    Returns:
        {
          "corr_window_days": int,
          "corr_nb": float,
          "index_corr_score": float,
        }
    """
    # Fetch histories (re-use same helper, just override ticker)
    nifty_df = get_nifty_history_yf(years=years_hist, ticker="^NSEI")
    bank_df = get_nifty_history_yf(years=years_hist, ticker="^NSEBANK")

    if nifty_df.empty or bank_df.empty:
        raise ValueError("NIFTY or BANKNIFTY history from yfinance is empty.")

    # Align on dates
    df = pd.DataFrame()
    df["nifty_close"] = (nifty_df["Adj Close"] if "Adj Close" in nifty_df.columns else nifty_df["Close"])
    df["bank_close"] = (bank_df["Adj Close"] if "Adj Close" in bank_df.columns else bank_df["Close"])
    df = df.dropna()

    # Daily log returns
    df["nifty_ret"] = np.log(df["nifty_close"] / df["nifty_close"].shift(1))
    df["bank_ret"] = np.log(df["bank_close"] / df["bank_close"].shift(1))
    df = df.dropna()

    if len(df) < lookback_days:
        raise ValueError(f"Need at least {lookback_days} days of returns, got {len(df)}.")

    recent = df.iloc[-lookback_days:]
    corr_nb = float(recent["nifty_ret"].corr(recent["bank_ret"]))

    # Map correlation [-1,1] into [0,1] but emphasize high positive corr:
    # - corr <= 0.0 => 0
    # - corr >= 0.8 => 1
    index_corr_score = _norm_01(corr_nb, 0.0, 0.8)

    return {
        "corr_window_days": lookback_days,
        "corr_nb": corr_nb,
        "index_corr_score": index_corr_score,
    }


# ---------------------------------------------------------------------------
# Combine into K(t)
# ---------------------------------------------------------------------------

def compute_contagion_score(
    breadth_feats: Dict[str, Any],
    corr_feats: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Combine breadth & correlation features into a single K(t) in [0,1].

    Components:
      - breadth_sync_score
      - wide_move_score
      - index_corr_score
    """
    breadth_sync = breadth_feats.get("breadth_sync_score", np.nan)
    wide_move = breadth_feats.get("wide_move_score", np.nan)
    index_corr = corr_feats.get("index_corr_score", np.nan)

    components = {
        "breadth_sync_score": breadth_sync,
        "wide_move_score": wide_move,
        "index_corr_score": index_corr,
    }

    weights = {
        "breadth_sync_score": 0.4,
        "wide_move_score": 0.3,
        "index_corr_score": 0.3,
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

    K = num / den if den > 0 else np.nan
    return K, components


def categorize_contagion(K: float, components: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict that holds numeric value, category, and a one-line explanation
    for K(t) and each component.

    Returns:
        {
          "K": {
            "value": <float>,
            "category": <str>,
            "explanation": <str>,
          },
          "breadth_sync_score": {...},
          "wide_move_score": {...},
          "index_corr_score": {...},
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Overall K
    K_cat = _bucket_score(K)
    result["K"] = {
        "value": K,
        "category": K_cat,
        "explanation": _category_explanation("K", K_cat),
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

def get_nifty_contagion_state(
    years_hist: int = 1,
    breadth_index: str = "SECURITIES IN F&O",
    lookback_days: int = 60,
    min_move_abs: float = 0.015,
) -> Dict[str, Any]:
    """
    High-level function:

    - Compute breadth features from F&O universe.
    - Compute NIFTY/BANKNIFTY correlation features.
    - Compute K(t) and component scores.
    - Attach categories + explanations.

    Returns:
        {
          "as_of": <latest date as string>,
          "breadth_features": {...},
          "index_corr_features": {...},
          "contagion_score": K,
          "contagion_components": {...},
          "contagion_categories": {...},
        }
    """
    breadth_feats = compute_breadth_features_from_fo(
        index_label=breadth_index,
        min_move_abs=min_move_abs,
    )
    corr_feats = compute_index_corr_features(
        years_hist=years_hist,
        lookback_days=lookback_days,
    )

    K, components = compute_contagion_score(breadth_feats, corr_feats)
    categories = categorize_contagion(K, components)

    # as_of: we use today's date as NSE data is "today's snapshot"
    as_of_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    return {
        "as_of": as_of_str,
        "breadth_features": breadth_feats,
        "index_corr_features": corr_feats,
        "contagion_score": K,
        "contagion_components": components,
        "contagion_categories": categories,
    }


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = get_nifty_contagion_state(
        years_hist=1,
        breadth_index="SECURITIES IN F&O",
        lookback_days=60,
        min_move_abs=0.015,
    )

    print("Contagion snapshot as of:", state["as_of"])

    K = state["contagion_score"]
    cats = state["contagion_categories"]

    # Overall K(t)
    K_info = cats["K"]
    K_val = K_info["value"]
    K_cat = K_info["category"]
    K_expl = K_info["explanation"]

    if K_val is not None and not np.isnan(K_val):
        print("\n=== Contagion Score K(t) ===")
        print(f"K(t): {K_val:.2f}  [{K_cat}]  (0 = idiosyncratic, 1 = fully systemic)")
        print(f"      -> {K_expl}")
    else:
        print("\nK(t): NaN  [N/A]")
        print(f"      -> {K_expl}")

    print("\nComponents:")
    for name in ["breadth_sync_score", "wide_move_score", "index_corr_score"]:
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
