"""
generate_snapshot.py

Builds a single JSON snapshot for the GitHub dashboard:

- regime_state: overall regime + playbooks
- options_state: NIFTY triggers & candidate options
- inputs: C(t), S(t), K(t), D(t) scores + categories + explanations
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import json
import math

import numpy as np

# Our own modules
from PLT_Regimes import get_nifty_regime_state
from PLT_Nifty_OptionTriggers import get_nifty_options_breakout_state
from PLT_Critical import (
    build_nifty_market_state,
    compute_criticality_score,
    categorize_criticality,
)
from PLT_Shocks import get_nifty_shock_state
from PLT_Contagion import get_nifty_contagion_state
from PLT_Direction import get_nifty_direction_state


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _clean_numbers(obj):
    """
    Recursively replace NaN/Inf with None so the JSON is strict and
    parseable in the browser.
    """
    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _json_default(obj):
    """Handle datetimes / numpy types when dumping JSON."""
    from datetime import datetime as _dt

    if isinstance(obj, _dt):
        return obj.isoformat()

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)

    return str(obj)


def _first(state: Dict[str, Any], keys) -> Optional[Any]:
    """Return the first non-None value for any key in `keys`."""
    for k in keys:
        if k in state and state[k] is not None:
            return state[k]
    return None


def _bucket_01(x: Optional[float]) -> str:
    """
    Map a score in [0,1] to a category.
    Used for C(t), S(t), K(t).
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


def _bucket_direction(x: Optional[float]) -> str:
    """
    Map direction score in [-1,1] to a label.
    """
    if x is None:
        return "N/A"
    try:
        if np.isnan(x):
            return "N/A"
    except TypeError:
        return "N/A"

    if x <= -0.5:
        return "Strongly Bearish"
    elif x < -0.15:
        return "Bearish"
    elif x <= 0.15:
        return "Neutral"
    elif x < 0.5:
        return "Bullish"
    else:
        return "Strongly Bullish"


def _expl_criticality(cat: str) -> str:
    if cat == "Low":
        return "System appears calm with limited build-up of fragility."
    if cat == "Moderate":
        return "System is in a normal, watchful regime—neither very calm nor very fragile."
    if cat == "Elevated":
        return "System shows elevated fragility; modest shocks can cause outsized moves."
    if cat == "Critical":
        return "System is highly fragile; a meaningful trigger could cause large, disorderly moves."
    return "Not enough data to interpret criticality."


def _expl_shock(cat: str) -> str:
    if cat == "Low":
        return "No meaningful shock—price and volume are well within recent norms."
    if cat == "Moderate":
        return "Some signs of stress, but still within the range of normal volatility."
    if cat == "Elevated":
        return "Recent move is large vs recent history—watch for follow-through or mean-reversion."
    if cat == "Critical":
        return "Extreme shock vs recent history—very large move or volume spike."
    return "Not enough data to interpret shocks."


def _expl_contagion(cat: str) -> str:
    if cat == "Low":
        return "Moves look mostly idiosyncratic—breadth and co-movement are normal."
    if cat == "Moderate":
        return "Co-movement is at a normal level—some common factor, but not dominating."
    if cat == "Elevated":
        return "Market is moving more in sync—common factors are dominating stock behaviour."
    if cat == "Critical":
        return "Very high co-movement—market trading like a single risk-on / risk-off block."
    return "Not enough data to interpret contagion."


def _expl_direction(cat: str) -> str:
    if cat == "Strongly Bearish":
        return "Positioning and options skew are strongly bearish."
    if cat == "Bearish":
        return "Positioning leans bearish, but not extreme."
    if cat == "Neutral":
        return "Positioning is broadly balanced between bullish and bearish."
    if cat == "Bullish":
        return "Positioning leans bullish, but not extreme."
    if cat == "Strongly Bullish":
        return "Positioning and options skew are strongly bullish."
    return "Not enough data to interpret directional bias."


# ---------------------------------------------------------------------
# Build C, S, K, D blocks for the dashboard
# ---------------------------------------------------------------------


def build_inputs_block() -> Dict[str, Dict[str, Any]]:
    """
    Compute C(t), S(t), K(t), D(t) from the underlying modules and
    return a dict:

    {
      "C": {"value": ..., "category": ..., "explanation": ...},
      "S": {...},
      "K": {...},
      "D": {...},
    }
    """

    # ----- C(t): Criticality -----
    crit_state = build_nifty_market_state(years_hist=3)
    C_val, crit_components = compute_criticality_score(crit_state)
    crit_summary = categorize_criticality(C_val, crit_components)
    C_info = crit_summary.get("C", {})
    C_value = C_info.get("value", C_val)
    C_category = C_info.get("category") or _bucket_01(C_value)
    C_expl = C_info.get("explanation") or _expl_criticality(C_category)

    # ----- S(t): Shocks -----
    shocks_state = get_nifty_shock_state()
    S_raw = _first(shocks_state, ["score", "shock_score", "S", "s_score"])
    try:
        S_value = float(S_raw) if S_raw is not None else np.nan
    except Exception:
        S_value = np.nan
    S_category = _bucket_01(S_value)
    S_expl = shocks_state.get("explanation") or shocks_state.get(
        "shock_explanation", ""
    ) or _expl_shock(S_category)

    # ----- K(t): Contagion -----
    cont_state = get_nifty_contagion_state()
    K_raw = _first(cont_state, ["score", "contagion_score", "K", "k_score"])
    try:
        K_value = float(K_raw) if K_raw is not None else np.nan
    except Exception:
        K_value = np.nan
    K_category = _bucket_01(K_value)
    K_expl = cont_state.get("explanation") or cont_state.get(
        "contagion_explanation", ""
    ) or _expl_contagion(K_category)

    # ----- D(t): Direction -----
    dir_state = get_nifty_direction_state()
    dir_cats = dir_state.get("directional_categories", {}) or {}
    D_entry = dir_cats.get("D", {}) if isinstance(dir_cats, dict) else {}

    D_raw = D_entry.get("value")
    if D_raw is None:
        D_raw = dir_state.get("directional_score")

    try:
        D_value = float(D_raw) if D_raw is not None else np.nan
    except Exception:
        D_value = np.nan

    D_category = D_entry.get("category") or _bucket_direction(D_value)
    D_expl = D_entry.get("explanation") or _expl_direction(D_category)

    return {
        "C": {
            "value": C_value,
            "category": C_category,
            "explanation": C_expl,
        },
        "S": {
            "value": S_value,
            "category": S_category,
            "explanation": S_expl,
        },
        "K": {
            "value": K_value,
            "category": K_category,
            "explanation": K_expl,
        },
        "D": {
            "value": D_value,
            "category": D_category,
            "explanation": D_expl,
        },
    }


# ---------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------


def build_snapshot() -> Dict[str, Any]:
    """
    High-level snapshot for the GitHub dashboard.
    """
    regime_state = get_nifty_regime_state()
    options_state = get_nifty_options_breakout_state()

    # Build / refresh C,S,K,D inputs for the UI
    inputs = build_inputs_block()
    existing_inputs = regime_state.get("inputs", {})

    # If for some reason new D is NaN, fall back to regime_state.inputs.D
    try:
        new_D_val = inputs.get("D", {}).get("value")
        if new_D_val is None or np.isnan(new_D_val):
            if "D" in existing_inputs:
                inputs["D"] = existing_inputs["D"]
    except TypeError:
        pass

    merged_inputs = {**existing_inputs, **inputs}
    regime_state["inputs"] = merged_inputs

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regime_state": regime_state,
        "options_state": options_state,
    }


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------


def main():
    snap = build_snapshot()
    snap = _clean_numbers(snap)  # sanitize NaN / Inf

    out_dir = Path("docs") / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nifty_snapshot.json"

    out_path.write_text(
        json.dumps(snap, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote snapshot -> {out_path}")


if __name__ == "__main__":
    main()
