"""
PLT_Nifty_OptionTriggers.py

NIFTY-Only Trigger/Stop/Target + Option Candidate Selector

This module does 3 things:

1) Pulls NIFTY (^NSEI) daily OHLC via yfinance (through PLT_Levels).
2) Builds a simple breakout/breakdown plan on the index:
   - Long breakout: above 20d high
   - Short breakdown: below 20d low
   - Uses ATR(14) to define entry buffer, stops, and targets.
3) Fetches the NIFTY option chain for the nearest expiry via Bharat-sm-data
   and selects "candidate" options for long-up / long-down structures.

Everything here is *framework* / tooling — you still need to decide execution,
risk size, and whether a trade fits your view & constraints.

Dependencies:
- PLT_Levels.get_ticker_levels_state (for NIFTY spot levels)
- Bharat-sm-data: Derivatives.NSE.NSE

"""

from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from PLT_Levels import get_ticker_levels_state

# Bharat-sm-data
from Derivatives.NSE import NSE as DerivNSE


# ----------------------------------------------------------------------
# 1) Helper: normalise/choose expiry for NIFTY options
# ----------------------------------------------------------------------

def _normalize_expiry(expiry_raw) -> datetime:
    """
    Ensure 'expiry_raw' is a single datetime-like object.

    Handles:
    - list of expiries: take the first (nearest)
    - string dates: parsed via pandas
    - pandas.Timestamp / datetime: returned as-is
    """
    if isinstance(expiry_raw, list):
        if not expiry_raw:
            raise ValueError("get_options_expiry returned an empty list.")
        expiry_raw = expiry_raw[0]

    if isinstance(expiry_raw, (datetime, pd.Timestamp)):
        return expiry_raw if isinstance(expiry_raw, datetime) else expiry_raw.to_pydatetime()

    expiry_dt = pd.to_datetime(expiry_raw)
    if isinstance(expiry_dt, pd.Timestamp):
        return expiry_dt.to_pydatetime()
    return expiry_dt


def fetch_nifty_option_chain_nearest_expiry(
    index_ticker: str = "NIFTY",
) -> Tuple[pd.DataFrame, datetime]:
    """
    Fetch NIFTY option chain for the *nearest expiry* using Bharat-sm-data.

    Returns:
        (option_chain_df, expiry_dt)

    On failure, raises or returns empty DataFrame.
    """
    deriv = DerivNSE()

    # 1) Get nearest expiry
    expiry_raw = deriv.get_options_expiry(ticker=index_ticker, is_index=True)
    expiry_dt = _normalize_expiry(expiry_raw)

    # 2) Fetch option chain for that expiry
    oc = deriv.get_option_chain(
        ticker=index_ticker,
        expiry=expiry_dt,
        is_index=True,
    )
    if oc is None:
        oc = pd.DataFrame()

    return oc, expiry_dt


# ----------------------------------------------------------------------
# 2) Build NIFTY breakout/breakdown plan from levels + ATR
# ----------------------------------------------------------------------

def build_nifty_breakout_plan(
    breakout_lookback: int = 20,
    atr_window: int = 14,
    proximity_window_pct: float = 0.05,
    trend_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Use PLT_Levels to get NIFTY levels and compute a breakout/breakdown plan.

    For NIFTY (^NSEI), we use:
      - high_20d / low_20d as swing breakout/breakdown levels,
      - ATR(14) for buffers, stops, and targets.

    Long breakout (upside):
      entry_spot   = high_20d + 0.20 * ATR
      stop_spot    = entry_spot - 1.00 * ATR
      target1_spot = entry_spot + 1.00 * ATR
      target2_spot = entry_spot + 2.00 * ATR

    Short breakdown (downside):
      entry_spot   = low_20d - 0.20 * ATR
      stop_spot    = entry_spot + 1.00 * ATR
      target1_spot = entry_spot - 1.00 * ATR
      target2_spot = entry_spot - 2.00 * ATR

    Returns:
        {
          "as_of": <date string>,
          "spot_levels": {...},
          "breakout_scores": {...},  # L_long / L_short etc.
          "long_plan": {...},
          "short_plan": {...},
        }
    """
    # PLT_Levels gives us everything we need for NIFTY
    nifty_state = get_ticker_levels_state(
        ticker="^NSEI",
        years_hist=1,
        breakout_lookback=breakout_lookback,
        atr_window=atr_window,
        proximity_window_pct=proximity_window_pct,
        trend_weight=trend_weight,
    )

    levels = nifty_state["levels"]
    scores = nifty_state["scores"]

    close = levels["close"]
    high_20d = levels["high_20d"]
    low_20d = levels["low_20d"]
    atr_14 = levels["atr_14"]

    # -------------------------
    # Long breakout plan
    # -------------------------
    long_entry = high_20d + 0.20 * atr_14
    long_stop = long_entry - 1.00 * atr_14
    long_tgt1 = long_entry + 1.00 * atr_14
    long_tgt2 = long_entry + 2.00 * atr_14

    long_risk = long_entry - long_stop  # in index points
    long_r_multiple_t1 = (long_tgt1 - long_entry) / long_risk if long_risk > 0 else np.nan
    long_r_multiple_t2 = (long_tgt2 - long_entry) / long_risk if long_risk > 0 else np.nan

    long_plan = {
        "direction": "long",
        "spot_now": close,
        "entry_spot": long_entry,
        "stop_spot": long_stop,
        "target1_spot": long_tgt1,
        "target2_spot": long_tgt2,
        "risk_per_contract_points": long_risk,
        "R_multiple_target1": long_r_multiple_t1,
        "R_multiple_target2": long_r_multiple_t2,
        "L_long_score": scores["L_long"]["score"],
        "L_long_category": scores["L_long"]["category"],
        "L_long_explanation": scores["L_long"]["explanation"],
    }

    # -------------------------
    # Short breakdown plan
    # -------------------------
    short_entry = low_20d - 0.20 * atr_14
    short_stop = short_entry + 1.00 * atr_14
    short_tgt1 = short_entry - 1.00 * atr_14
    short_tgt2 = short_entry - 2.00 * atr_14

    short_risk = short_stop - short_entry  # in index points
    short_r_multiple_t1 = (short_entry - short_tgt1) / short_risk if short_risk > 0 else np.nan
    short_r_multiple_t2 = (short_entry - short_tgt2) / short_risk if short_risk > 0 else np.nan

    short_plan = {
        "direction": "short",
        "spot_now": close,
        "entry_spot": short_entry,
        "stop_spot": short_stop,
        "target1_spot": short_tgt1,
        "target2_spot": short_tgt2,
        "risk_per_contract_points": short_risk,
        "R_multiple_target1": short_r_multiple_t1,
        "R_multiple_target2": short_r_multiple_t2,
        "L_short_score": scores["L_short"]["score"],
        "L_short_category": scores["L_short"]["category"],
        "L_short_explanation": scores["L_short"]["explanation"],
    }

    return {
        "as_of": nifty_state["as_of"],
        "spot_levels": levels,
        "breakout_scores": scores,
        "long_plan": long_plan,
        "short_plan": short_plan,
    }


# ----------------------------------------------------------------------
# 3) Option selection logic (NIFTY index options only)
# ----------------------------------------------------------------------

def _get_strike_series(oc: pd.DataFrame) -> pd.Series:
    """
    Extract strike prices as a numeric Series from option_chain DataFrame.

    Uses explicit 'strikePrice' column if present, else index.
    """
    if "strikePrice" in oc.columns:
        strikes = oc["strikePrice"]
    else:
        strikes = oc.index.to_series()
    return strikes.astype(float)


def _select_nifty_option_from_chain(
    oc: pd.DataFrame,
    spot: float,
    expiry_dt: datetime,
    direction: str = "long",
    moneyness: str = "ATM",
    min_oi: int = 1000,
) -> Optional[Dict[str, Any]]:
    """
    Select a candidate NIFTY option from the option chain.

    direction:
      - "long_up"  -> long call (CE)
      - "long_down"-> long put (PE)

    moneyness:
      - currently only uses 'ATM' (nearest strike to spot).

    Returns dict like:
        {
          "type": "CE" or "PE",
          "strike": <float>,
          "expiry": <datetime>,
          "last_price": <float or NaN>,
          "open_interest": <float or NaN>,
          "iv": <float or NaN>,
        }
    or None if nothing suitable found.
    """
    if oc is None or oc.empty or spot <= 0:
        return None

    strikes = _get_strike_series(oc)
    cols = list(oc.columns)

    # Find relevant columns
    ce_lp_col = next((c for c in cols if "CE_lastPrice" in c), None)
    pe_lp_col = next((c for c in cols if "PE_lastPrice" in c), None)
    ce_oi_col = next((c for c in cols if "CE_openInterest" in c), None)
    pe_oi_col = next((c for c in cols if "PE_openInterest" in c), None)
    ce_iv_col = next((c for c in cols if "CE_impliedVolatility" in c), None)
    pe_iv_col = next((c for c in cols if "PE_impliedVolatility" in c), None)

    if direction == "long_up":
        # Candidate: CE near ATM, with decent OI
        if ce_lp_col is None or ce_oi_col is None:
            return None

        oc_local = oc.copy()
        oc_local["strike"] = strikes
        oc_local["oi_ce"] = oc_local[ce_oi_col].fillna(0)

        # basic filter: positive OI above threshold
        mask = oc_local["oi_ce"] >= min_oi
        oc_filt = oc_local.loc[mask].copy()
        if oc_filt.empty:
            oc_filt = oc_local  # fall back: ignore OI filter

        # distance to spot (ATM)
        oc_filt["dist_abs"] = (oc_filt["strike"] - spot).abs()
        oc_filt = oc_filt.sort_values(["dist_abs", "oi_ce"], ascending=[True, False])

        row = oc_filt.iloc[0]
        return {
            "type": "CE",
            "strike": float(row["strike"]),
            "expiry": expiry_dt,
            "last_price": float(row.get(ce_lp_col, np.nan)),
            "open_interest": float(row.get(ce_oi_col, np.nan)),
            "iv": float(row.get(ce_iv_col, np.nan)) if ce_iv_col else np.nan,
        }

    elif direction == "long_down":
        # Candidate: PE near ATM, with decent OI
        if pe_lp_col is None or pe_oi_col is None:
            return None

        oc_local = oc.copy()
        oc_local["strike"] = strikes
        oc_local["oi_pe"] = oc_local[pe_oi_col].fillna(0)

        mask = oc_local["oi_pe"] >= min_oi
        oc_filt = oc_local.loc[mask].copy()
        if oc_filt.empty:
            oc_filt = oc_local

        oc_filt["dist_abs"] = (oc_filt["strike"] - spot).abs()
        oc_filt = oc_filt.sort_values(["dist_abs", "oi_pe"], ascending=[True, False])

        row = oc_filt.iloc[0]
        return {
            "type": "PE",
            "strike": float(row["strike"]),
            "expiry": expiry_dt,
            "last_price": float(row.get(pe_lp_col, np.nan)),
            "open_interest": float(row.get(pe_oi_col, np.nan)),
            "iv": float(row.get(pe_iv_col, np.nan)) if pe_iv_col else np.nan,
        }

    else:
        return None


# ----------------------------------------------------------------------
# 4) High-level: NIFTY breakout plan + option candidates
# ----------------------------------------------------------------------

def get_nifty_options_breakout_state(
    breakout_lookback: int = 20,
    atr_window: int = 14,
    proximity_window_pct: float = 0.05,
    trend_weight: float = 0.3,
    min_oi: int = 1000,
) -> Dict[str, Any]:
    """
    High-level function tying everything together for NIFTY:

    - Builds long & short breakout plans on the index (spot).
    - Fetches nearest-expiry NIFTY option chain.
    - Selects candidate long-up (call) and long-down (put) options.

    Returns:
        {
          "as_of": {
            "spot_levels": <date str>,
            "options_expiry": <expiry date>,
          },
          "spot_levels": {...},
          "breakout_scores": {...},
          "long_plan": {...},
          "short_plan": {...},
          "options": {
            "long_up_call": {... or None},
            "long_down_put": {... or None},
          },
        }
    """
    # 1) Spot breakout plan
    plan = build_nifty_breakout_plan(
        breakout_lookback=breakout_lookback,
        atr_window=atr_window,
        proximity_window_pct=proximity_window_pct,
        trend_weight=trend_weight,
    )

    levels = plan["spot_levels"]
    long_plan = plan["long_plan"]
    short_plan = plan["short_plan"]

    spot_now = levels["close"]

    # 2) Option chain for nearest expiry
    try:
        oc, expiry_dt = fetch_nifty_option_chain_nearest_expiry(index_ticker="NIFTY")
    except Exception as e:
        print(f"[WARN] fetch_nifty_option_chain_nearest_expiry failed: {e}")
        oc = pd.DataFrame()
        expiry_dt = None

    # 3) Candidate options based on direction
    long_up_call = None
    long_down_put = None

    if expiry_dt is not None and not oc.empty:
        long_up_call = _select_nifty_option_from_chain(
            oc=oc,
            spot=spot_now,
            expiry_dt=expiry_dt,
            direction="long_up",
            moneyness="ATM",
            min_oi=min_oi,
        )
        long_down_put = _select_nifty_option_from_chain(
            oc=oc,
            spot=spot_now,
            expiry_dt=expiry_dt,
            direction="long_down",
            moneyness="ATM",
            min_oi=min_oi,
        )

    return {
        "as_of": {
            "spot_levels": plan["as_of"],
            "options_expiry": expiry_dt.strftime("%Y-%m-%d") if expiry_dt else None,
        },
        "spot_levels": levels,
        "breakout_scores": plan["breakout_scores"],
        "long_plan": long_plan,
        "short_plan": short_plan,
        "options": {
            "long_up_call": long_up_call,
            "long_down_put": long_down_put,
        },
    }


# ----------------------------------------------------------------------
# 5) Manual test / demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    state = get_nifty_options_breakout_state()

    print("NIFTY Options Breakout Snapshot:")
    print("  Spot levels as of:", state["as_of"]["spot_levels"])
    print("  Options expiry   :", state["as_of"]["options_expiry"])

    lv = state["spot_levels"]
    print(f"\nNIFTY Close: {lv['close']:.2f}")
    print(f"20d High / Low: {lv['high_20d']:.2f} / {lv['low_20d']:.2f}")
    print(f"ATR(14): {lv['atr_14']:.2f}")

    long_plan = state["long_plan"]
    short_plan = state["short_plan"]

    print("\n=== Long Breakout Plan (Index) ===")
    print(f"  Spot now      : {long_plan['spot_now']:.2f}")
    print(f"  Entry (spot)  : {long_plan['entry_spot']:.2f}")
    print(f"  Stop (spot)   : {long_plan['stop_spot']:.2f}")
    print(f"  Target1 (spot): {long_plan['target1_spot']:.2f}")
    print(f"  Target2 (spot): {long_plan['target2_spot']:.2f}")
    print(f"  Risk/contract : {long_plan['risk_per_contract_points']:.2f} pts")
    print(f"  R@T1 / R@T2   : {long_plan['R_multiple_target1']:.2f} / {long_plan['R_multiple_target2']:.2f}")
    print(f"  L_long        : {long_plan['L_long_score']:.2f} [{long_plan['L_long_category']}]")
    print(f"    -> {long_plan['L_long_explanation']}")

    print("\n=== Short Breakdown Plan (Index) ===")
    print(f"  Spot now      : {short_plan['spot_now']:.2f}")
    print(f"  Entry (spot)  : {short_plan['entry_spot']:.2f}")
    print(f"  Stop (spot)   : {short_plan['stop_spot']:.2f}")
    print(f"  Target1 (spot): {short_plan['target1_spot']:.2f}")
    print(f"  Target2 (spot): {short_plan['target2_spot']:.2f}")
    print(f"  Risk/contract : {short_plan['risk_per_contract_points']:.2f} pts")
    print(f"  R@T1 / R@T2   : {short_plan['R_multiple_target1']:.2f} / {short_plan['R_multiple_target2']:.2f}")
    print(f"  L_short       : {short_plan['L_short_score']:.2f} [{short_plan['L_short_category']}]")
    print(f"    -> {short_plan['L_short_explanation']}")

    print("\n=== Candidate Options (Nearest Expiry) ===")
    long_up = state["options"]["long_up_call"]
    long_down = state["options"]["long_down_put"]

    if long_up:
        print("  Long-up (Bull) candidate CE:")
        print(f"    Type        : {long_up['type']}")
        print(f"    Strike      : {long_up['strike']:.2f}")
        print(f"    Expiry      : {long_up['expiry'].strftime('%Y-%m-%d')}")
        print(f"    Last price  : {long_up['last_price']:.2f}")
        print(f"    Open Int    : {long_up['open_interest']:.0f}")
        print(f"    IV (if avail): {long_up['iv']:.2f}")
    else:
        print("  Long-up (Bull) candidate CE: None (no suitable option found)")

    if long_down:
        print("  Long-down (Bear) candidate PE:")
        print(f"    Type        : {long_down['type']}")
        print(f"    Strike      : {long_down['strike']:.2f}")
        print(f"    Expiry      : {long_down['expiry'].strftime('%Y-%m-%d')}")
        print(f"    Last price  : {long_down['last_price']:.2f}")
        print(f"    Open Int    : {long_down['open_interest']:.0f}")
        print(f"    IV (if avail): {long_down['iv']:.2f}")
    else:
        print("  Long-down (Bear) candidate PE: None (no suitable option found)")
