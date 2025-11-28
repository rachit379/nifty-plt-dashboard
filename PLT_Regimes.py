"""
PLT_Regimes.py

NIFTY Regime & Playbook Engine R(t)

This module combines:
- C(t): Criticality / Fragility of the NIFTY options complex (from PLT_Critical)
- S(t): Shock intensity for the latest day (from PLT_Shocks)
- K(t): Contagion / Co-movement intensity (from PLT_Contagion)
- D(t): Directional bias of the options complex (from PLT_Direction)

…into a higher-level "regime" and a suggested trading playbook.

Definitions
-----------
C(t): 0 = very calm/robust, 1 = very fragile/loaded.
S(t): 0 = no meaningful shock, 1 = extreme shock.
K(t): 0 = very idiosyncratic, 1 = fully systemic co-movement.
D(t): -1 = strongly bearish, 0 = neutral, +1 = strongly bullish.

Regime logic (v1, primarily based on C & S, then adjusted by K and D):
- Calm / Normal      : C low–mod, S low–mod
- Loaded, No Shock   : C elevated–crit, S low–mod
- Isolated Shock     : C low–mod, S elevated–crit
- Fragile + Shock    : C elevated–crit, S elevated–crit
- Transitional       : anything else / mixed

K(t) adjusts systemic vs idiosyncratic focus.
D(t) adds bullish/bearish/neutral bias to the playbook.

NOTE: Framework for research and experimentation, NOT trading advice.
"""

from typing import Dict, Any

import numpy as np

from PLT_Critical import (
    build_nifty_market_state,
    compute_criticality_score,
    categorize_criticality,
)
from PLT_Shocks import (
    get_nifty_shock_state,
)
from PLT_Contagion import (
    get_nifty_contagion_state,
)
from PLT_Direction import (
    get_nifty_direction_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAT_LEVEL_MAP = {
    "Low": 0,
    "Moderate": 1,
    "Elevated": 2,
    "Critical": 3,
}


def _cat_to_level(cat: str) -> int:
    """Map bucket label -> ordinal level for logic; default to 1 (Moderate)."""
    if cat is None:
        return 1
    return _CAT_LEVEL_MAP.get(cat, 1)


def _adjust_regime_for_K(
    regime: Dict[str, Any],
    K_cat: str,
    K_val: float,
) -> Dict[str, Any]:
    """
    Tweak regime description and playbook based on contagion K(t).

    K low  -> emphasise stock-picking / idiosyncratic trades.
    K high -> emphasise index-level trades / systemic macro risk.
    """
    # Ensure notes arrays exist
    intraday_notes = regime["intraday"].setdefault("notes", [])
    options_notes = regime["options"].setdefault("notes", [])

    K_level = _cat_to_level(K_cat)

    # Attach a short K summary to regime itself
    regime["contagion_context"] = {
        "K_category": K_cat,
        "K_value": K_val,
    }

    # Low contagion: idiosyncratic environment
    if K_level == 0:  # Low
        extra_desc = (
            " Co-movement is low, suggesting a more idiosyncratic, "
            "stock-picking-friendly environment."
        )
        regime["description"] += extra_desc

        intraday_notes.append(
            "Low contagion: prioritise stock-specific setups and relative value trades."
        )
        intraday_notes.append(
            "Index-level signals may be weaker; focus on strong/weak single-name themes."
        )
        options_notes.append(
            "Consider single-stock options for idiosyncratic ideas; index options may be less efficient here."
        )

    # Very high contagion: systemic risk-on/off block
    elif K_level >= 2:  # Elevated or Critical
        extra_desc = (
            " Co-movement is strong, indicating a more systemic risk-on/off regime."
        )
        regime["description"] += extra_desc

        intraday_notes.append(
            "High contagion: favour index/sector baskets over very concentrated single-name bets."
        )
        intraday_notes.append(
            "Correlation risk is elevated—pairs trades may behave like outright index exposure."
        )
        options_notes.append(
            "Index options (NIFTY/BANKNIFTY) become attractive vehicles for expressing macro views."
        )
        options_notes.append(
            "Be aware that correlation spikes can reduce diversification benefits across positions."
        )

    # Moderate contagion: normal mixed state
    else:
        extra_desc = (
            " Co-movement is at a normal level—both index and stock-specific trades are viable."
        )
        regime["description"] += extra_desc

        intraday_notes.append(
            "Moderate contagion: balanced opportunity between index-level and stock-level ideas."
        )

    return regime


def _direction_bias_from_cat(D_cat: str) -> str:
    """
    Map the D(t) category into a simpler directional_bias flag:

    Returns:
        "bullish", "bearish", or "none"
    """
    if D_cat in ("Strongly Bullish", "Bullish"):
        return "bullish"
    if D_cat in ("Strongly Bearish", "Bearish"):
        return "bearish"
    return "none"


def _apply_direction_to_regime(
    regime: Dict[str, Any],
    D_cat: str,
    D_val: float,
) -> Dict[str, Any]:
    """
    Tweak regime based on directional bias D(t).

    - Sets regime['intraday']['directional_bias'] to "bullish"/"bearish"/"none".
    - Adds notes on whether to favour long-side or short-side structures.
    """
    intraday = regime["intraday"]
    options = regime["options"]
    intraday_notes = intraday.setdefault("notes", [])
    options_notes = options.setdefault("notes", [])

    bias = _direction_bias_from_cat(D_cat)

    # Store context
    regime["directional_context"] = {
        "D_category": D_cat,
        "D_value": D_val,
        "bias": bias,
    }

    # Default if not set
    if "directional_bias" not in intraday:
        intraday["directional_bias"] = "none"

    # If no strong bias, just note it and leave intraday behaviour to price action
    if bias == "none":
        intraday["directional_bias"] = "none"
        intraday_notes.append(
            "Directional positioning is broadly balanced—let intraday price action drive long vs short bias."
        )
        options_notes.append(
            "No strong directional tilt from index options; favour symmetric structures or spreads hedged by levels."
        )
        return regime

    # Bullish bias
    if bias == "bullish":
        intraday["directional_bias"] = "bullish"
        intraday_notes.append(
            "Directional bias from options positioning is bullish—favour long-side breakouts and dips that hold key support."
        )
        options_notes.append(
            "Bullish tilt: call spreads / bull call diagonals / put credit spreads can be favoured structures."
        )
        return regime

    # Bearish bias
    if bias == "bearish":
        intraday["directional_bias"] = "bearish"
        intraday_notes.append(
            "Directional bias from options positioning is bearish—favour short-side breaks and rallies that stall near resistance."
        )
        options_notes.append(
            "Bearish tilt: put spreads / bear put diagonals / call credit spreads can be favoured structures."
        )
        return regime


# ---------------------------------------------------------------------------
# Regime classification & playbook
# ---------------------------------------------------------------------------

def classify_regime(
    C_cat: str,
    S_cat: str,
    K_cat: str,
    D_cat: str,
    C_val: float,
    S_val: float,
    K_val: float,
    D_val: float,
) -> Dict[str, Any]:
    """
    Map (C,S,K,D categories/values) -> regime & playbook.

    Returns a dict like:
        {
          "name": <str>,
          "description": <str>,
          "intraday": {...},
          "options": {...},
          "contagion_context": {...},
          "directional_context": {...},
        }
    """
    C_level = _cat_to_level(C_cat)
    S_level = _cat_to_level(S_cat)

    # Default structure
    regime: Dict[str, Any] = {
        "name": "Transitional / Mixed",
        "description": (
            "Signals from fragility and shocks are mixed—treat this as a "
            "transition regime, defaulting to conservative risk."
        ),
        "intraday": {
            "mode": "balanced",
            "enable_mean_reversion": True,
            "enable_breakout_trend": True,
            "volatility_scaling": "normal",   # low / normal / high
            "directional_bias": "none",       # none / bullish / bearish
            "notes": [
                "Keep position sizes moderate while signals are mixed.",
                "Prefer clean technical levels; avoid forcing trades.",
            ],
        },
        "options": {
            "allow_short_gamma_intraday": True,
            "allow_short_gamma_overnight": False,
            "allow_long_gamma": True,
            "notes": [
                "Stick to defined-risk structures (spreads) if selling options.",
                "Use small sizing for long gamma unless you have a strong view.",
            ],
        },
    }

    # -------------------------
    # 1) Calm / Normal regime
    # -------------------------
    if C_level <= 1 and S_level <= 1:
        regime["name"] = "Calm / Normal Regime"
        regime["description"] = (
            "System fragility is low-to-moderate and recent shocks are small—"
            "market behaviour is relatively stable and idiosyncratic."
        )
        regime["intraday"] = {
            "mode": "range / mean-reversion",
            "enable_mean_reversion": True,
            "enable_breakout_trend": False,
            "volatility_scaling": "low",
            "directional_bias": "none",
            "notes": [
                "Focus on range-bound setups: fade extremes, VWAP/mean reversion.",
                "Avoid chasing breakouts; many will fail in calm regimes.",
                "Tight stops and modest targets are appropriate.",
            ],
        }
        regime["options"] = {
            "allow_short_gamma_intraday": True,
            "allow_short_gamma_overnight": True,
            "allow_long_gamma": False,
            "notes": [
                "Calmer realized vol favours intraday and short-dated premium selling.",
                "Consider credit spreads / iron condors around well-defined ranges.",
                "Avoid paying too much for gamma unless you see stock-specific catalysts.",
            ],
        }
        regime = _adjust_regime_for_K(regime, K_cat, K_val)
        regime = _apply_direction_to_regime(regime, D_cat, D_val)
        return regime

    # -------------------------
    # 2) Loaded, No Shock yet
    # -------------------------
    if C_level >= 2 and S_level <= 1:
        regime["name"] = "Loaded, Waiting for a Shock"
        regime["description"] = (
            "System looks fragile/loaded (high criticality) but recent shocks "
            "have been small—conditions where a normal catalyst can trigger "
            "outsized moves."
        )
        regime["intraday"] = {
            "mode": "prep-for-breakout",
            "enable_mean_reversion": False,
            "enable_breakout_trend": True,
            "volatility_scaling": "normal",
            "directional_bias": "none",
            "notes": [
                "Be selective but ready to trade breakouts with follow-through.",
                "Avoid overtrading minor intraday wiggles—wait for decisive moves.",
                "Use alerts around key levels on NIFTY/BANKNIFTY.",
            ],
        }
        regime["options"] = {
            "allow_short_gamma_intraday": False,
            "allow_short_gamma_overnight": False,
            "allow_long_gamma": True,
            "notes": [
                "Fragility + no shock favours long gamma / long vol structures.",
                "Consider small-size long straddles/strangles in NIFTY with clear exits.",
                "Avoid naked short options; use defined-risk spreads if selling.",
            ],
        }
        regime = _adjust_regime_for_K(regime, K_cat, K_val)
        regime = _apply_direction_to_regime(regime, D_cat, D_val)
        return regime

    # -------------------------
    # 3) Isolated Shock regime
    # -------------------------
    if C_level <= 1 and S_level >= 2:
        regime["name"] = "Isolated Shock Regime"
        regime["description"] = (
            "Recent day showed a large shock, but underlying system fragility "
            "is low-to-moderate—often an event-driven or stock-specific move."
        )
        regime["intraday"] = {
            "mode": "post-shock reaction",
            "enable_mean_reversion": True,
            "enable_breakout_trend": True,
            "volatility_scaling": "high",
            "directional_bias": "none",
            "notes": [
                "Expect elevated intraday ranges following a big move.",
                "Consider day+1 mean reversion or continuation setups driven by how the shock closed.",
                "Reduce position size per trade but allow for larger targets.",
            ],
        }
        regime["options"] = {
            "allow_short_gamma_intraday": True,
            "allow_short_gamma_overnight": False,
            "allow_long_gamma": True,
            "notes": [
                "Post-shock IV may be elevated—good spot to sell premium *if* the move looks overdone.",
                "Alternatively, ride follow-through with call/put spreads rather than naked options.",
                "Be cautious with overnight risk; news flow can remain active.",
            ],
        }
        regime = _adjust_regime_for_K(regime, K_cat, K_val)
        regime = _apply_direction_to_regime(regime, D_cat, D_val)
        return regime

    # -------------------------
    # 4) Fragile + Shock regime
    # -------------------------
    if C_level >= 2 and S_level >= 2:
        regime["name"] = "Fragile + Shock Regime"
        regime["description"] = (
            "System was fragile/loaded and a strong shock has just occurred—"
            "this is where large, disorderly moves and cascades are more likely."
        )
        regime["intraday"] = {
            "mode": "trend / high-vol",
            "enable_mean_reversion": False,
            "enable_breakout_trend": True,
            "volatility_scaling": "high",
            "directional_bias": "none",  # direction can be refined by D or price action
            "notes": [
                "Favour trend-following setups over mean-reversion; respect the direction of the shock.",
                "Use wider stops with smaller size—volatility is elevated.",
                "Avoid fighting the primary move unless you have strong evidence of exhaustion.",
            ],
        }
        regime["options"] = {
            "allow_short_gamma_intraday": False,
            "allow_short_gamma_overnight": False,
            "allow_long_gamma": True,
            "notes": [
                "This is a dangerous regime for naked short options—avoid being short gamma.",
                "Use long gamma or defined-risk spreads to participate in large moves.",
                "If IV is extremely high, consider spreads (debit/credit) rather than outright options.",
            ],
        }
        regime = _adjust_regime_for_K(regime, K_cat, K_val)
        regime = _apply_direction_to_regime(regime, D_cat, D_val)
        return regime

    # If none of the above matched, keep default "Transitional / Mixed"
    regime = _adjust_regime_for_K(regime, K_cat, K_val)
    regime = _apply_direction_to_regime(regime, D_cat, D_val)
    return regime


# ---------------------------------------------------------------------------
# Public entrypoint: full regime snapshot
# ---------------------------------------------------------------------------

def get_nifty_regime_state(
    years_hist_critical: int = 3,
    years_hist_shocks: int = 1,
    shock_window_days: int = 60,
    years_hist_contagion: int = 1,
    lookback_days_contagion: int = 60,
    breadth_index: str = "SECURITIES IN F&O",
    min_move_abs: float = 0.015,
    years_hist_direction: int = 1,
) -> Dict[str, Any]:
    """
    High-level function:

    - Compute NIFTY criticality C(t) via PLT_Critical.
    - Compute NIFTY shock S(t) via PLT_Shocks.
    - Compute NIFTY contagion K(t) via PLT_Contagion.
    - Compute NIFTY directional bias D(t) via PLT_Direction.
    - Classify the current regime and build a playbook.

    Returns:
        {
          "as_of": {...},
          "criticality": {...},
          "shocks": {...},
          "contagion": {...},
          "direction": {...},
          "regime": {...},
        }
    """
    # --- Criticality C(t) ---
    crit_state = build_nifty_market_state(years_hist=years_hist_critical)
    C, C_components = compute_criticality_score(crit_state)
    C_categories = categorize_criticality(C, C_components)

    # --- Shocks S(t) ---
    shock_state = get_nifty_shock_state(
        years_hist=years_hist_shocks,
        window_days=shock_window_days,
    )
    S = shock_state["shock_score"]
    S_categories = shock_state["shock_categories"]

    # --- Contagion K(t) ---
    cont_state = get_nifty_contagion_state(
        years_hist=years_hist_contagion,
        breadth_index=breadth_index,
        lookback_days=lookback_days_contagion,
        min_move_abs=min_move_abs,
    )
    K = cont_state["contagion_score"]
    K_components = cont_state["contagion_components"]
    K_categories = cont_state["contagion_categories"]

    # --- Direction D(t) ---
    dir_state = get_nifty_direction_state(
        years_hist=years_hist_direction,
        index_ticker="NIFTY",
    )
    D = dir_state["directional_score"]
    D_components = dir_state["directional_components"]
    D_categories = dir_state["directional_categories"]

    # Extract top-level categories
    C_cat = C_categories["C"]["category"]
    S_cat = S_categories["S"]["category"]
    K_cat = K_categories["K"]["category"]
    D_cat = D_categories["D"]["category"]

    regime = classify_regime(
        C_cat=C_cat,
        S_cat=S_cat,
        K_cat=K_cat,
        D_cat=D_cat,
        C_val=C,
        S_val=S,
        K_val=K,
        D_val=D,
    )

    return {
        "as_of": {
            "critical": crit_state["as_of"],
            "shock": shock_state["as_of"],
            "contagion": cont_state["as_of"],
            "direction": dir_state["as_of"],
        },
        "criticality": {
            "score": C,
            "components": C_components,
            "categories": C_categories,
        },
        "shocks": {
            "score": S,
            "features": shock_state["shock_features"],
            "categories": S_categories,
        },
        "contagion": {
            "score": K,
            "components": K_components,
            "categories": K_categories,
        },
        "direction": {
            "score": D,
            "components": D_components,
            "categories": D_categories,
        },
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = get_nifty_regime_state(
        years_hist_critical=3,
        years_hist_shocks=1,
        shock_window_days=60,
        years_hist_contagion=1,
        lookback_days_contagion=60,
        breadth_index="SECURITIES IN F&O",
        min_move_abs=0.015,
        years_hist_direction=1,
    )

    print("Regime snapshot as of:")
    print("  Criticality as of:", state["as_of"]["critical"])
    print("  Shocks as of     :", state["as_of"]["shock"])
    print("  Contagion as of  :", state["as_of"]["contagion"])
    print("  Direction as of  :", state["as_of"]["direction"])

    C = state["criticality"]["score"]
    S = state["shocks"]["score"]
    K = state["contagion"]["score"]
    D = state["direction"]["score"]

    C_cat = state["criticality"]["categories"]["C"]["category"]
    S_cat = state["shocks"]["categories"]["S"]["category"]
    K_cat = state["contagion"]["categories"]["K"]["category"]
    D_cat = state["direction"]["categories"]["D"]["category"]

    print("\nInputs:")
    print(f"  C(t): {C:.2f}  [{C_cat}]")
    print(f"  S(t): {S:.2f}  [{S_cat}]")
    print(f"  K(t): {K:.2f}  [{K_cat}]")
    print(f"  D(t): {D:.2f}  [{D_cat}]")

    regime = state["regime"]
    print("\n=== Regime & Playbook ===")
    print(f"Regime: {regime['name']}")
    print(f"Summary: {regime['description']}")

    print("\nIntraday playbook:")
    for k, v in regime["intraday"].items():
        if k == "notes":
            continue
        print(f"  {k}: {v}")
    print("  notes:")
    for note in regime["intraday"]["notes"]:
        print(f"    - {note}")

    print("\nOptions playbook:")
    for k, v in regime["options"].items():
        if k == "notes":
            continue
        print(f"  {k}: {v}")
    print("  notes:")
    for note in regime["options"]["notes"]:
        print(f"    - {note}")
