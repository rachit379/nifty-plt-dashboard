"""
nifty_market_state.py

End-to-end “market state” snapshot for NIFTY:

- NIFTY daily OHLC from yfinance (^NSEI)
- Realized volatility (10d / 20d / 60d, annualized)
- India VIX from Moneycontrol via Bharat-sm-data
- NIFTY option chain for nearest expiry (from NSE via Bharat-sm-data)
- PCR (OI & Volume)

This is the backbone for building a Criticality C(t) metric.

NIFTY Market Criticality Engine (C(t))

This module:
- Pulls NIFTY daily OHLC from yfinance (^NSEI) for the last N years.
- Computes realized volatility (10d / 20d / 60d, annualized).
- Uses Bharat-sm-data to fetch:
    - India VIX (from Moneycontrol),
    - NIFTY option chain for nearest expiry,
    - PCR (OI & Volume).
- Builds a “state_features” dict.
- Computes a Criticality Score C(t) in [0, 1] with interpretable components.

Definitions
-----------
C(t): Criticality / Fragility of the NIFTY options complex.
      0 = very calm / robust, 1 = very fragile / loaded.

Components:
- dryness_score:
    How calm recent realized volatility is relative to its own history.
    High dryness_score = realized vol is low vs its 3Y history
                         (market looks calm, fuel can quietly build up).

- ivrv_score:
    How rich implied vol (VIX) is vs 20d realized vol.
    High ivrv_score = options pricing much more movement than has been realized
                      (tension / nervousness in options).

- crowding_score:
    How much of total options OI is crowded near the current spot (±2%).
    High crowding_score = lots of OI around ATM (gamma crowding / fragility).

- pcr_score:
    How one-sided the book is via PCR(OI).
    High pcr_score = PCR far from 1.0 (everyone piled onto one side).

You can call `get_nifty_criticality_state()` from other files to get a
clean dict containing:
- state_features,
- criticality_score,
- criticality_components,
- criticality_categories.
"""


from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Bharat-sm-data imports
from Technical.NSE import NSE as TechNSE
from Derivatives.NSE import NSE as DerivNSE
from Fundamentals import MoneyControl


# -----------------------------------------------------------------------------
# 1) NIFTY history via yfinance (^NSEI) and realized volatility
# -----------------------------------------------------------------------------

def get_nifty_history_yf(
    years: int = 3,
    interval: str = "1d",
    ticker: str = "^NSEI",
) -> pd.DataFrame:
    """
    Fetch NIFTY OHLC data from Yahoo Finance (^NSEI) for the last `years` years.

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
        auto_adjust=False,  # keep both Close and Adj Close
        progress=False,
    )

    df = df.sort_index()
    return df


def compute_realized_vol_features(
    price_df: pd.DataFrame,
    trading_days: int = 252,
    windows: Tuple[int, ...] = (10, 20, 60),
) -> Dict[str, Any]:
    """
    Given a NIFTY price DataFrame (with 'Adj Close' or 'Close'),
    compute rolling realized volatility (annualized) for specified windows.

    Returns:
        {
          "rv_series": {window: pd.Series, ...},
          "rv_latest": {window: float, ...},
          "returns": pd.Series,
        }
    """
    if "Adj Close" in price_df.columns:
        px = price_df["Adj Close"].copy()
    else:
        px = price_df["Close"].copy()

    # Log returns
    rets = np.log(px / px.shift(1)).dropna()

    rv_series = {}
    rv_latest = {}

    for w in windows:
        # Rolling std of daily log returns, annualized
        rv = rets.rolling(w).std() * np.sqrt(trading_days)
        rv_series[w] = rv
        rv_latest[w] = float(rv.iloc[-1]) if not rv.dropna().empty else np.nan

    return {
        "rv_series": rv_series,
        "rv_latest": rv_latest,
        "returns": rets,
    }


# -----------------------------------------------------------------------------
# 2) Bharat-sm-data helpers (India VIX + NIFTY option chain + PCR)
# -----------------------------------------------------------------------------

INDEX_NAME_MAP = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
}


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
        return expiry_raw

    expiry_dt = pd.to_datetime(expiry_raw)
    if isinstance(expiry_dt, pd.Timestamp):
        return expiry_dt.to_pydatetime()
    return expiry_dt


def _safe_df(fn, *args, **kwargs) -> pd.DataFrame:
    """
    Call a function expected to return a DataFrame.
    On any error, log and return an empty DataFrame.
    """
    try:
        res = fn(*args, **kwargs)
        if res is None:
            return pd.DataFrame()
        if isinstance(res, pd.DataFrame):
            return res
        return pd.DataFrame(res)
    except Exception as e:
        print(f"[WARN] {fn.__name__} failed: {e}")
        return pd.DataFrame()


def _safe_scalar(fn, *args, **kwargs):
    """
    Call a function expected to return a scalar-like value.
    On error, log and return None.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {fn.__name__} failed: {e}")
        return None


def fetch_nifty_derivatives_state(
    index_ticker: str = "NIFTY",
) -> Dict[str, Any]:
    """
    Use Bharat-sm-data to fetch:

    - India VIX (from Moneycontrol)
    - NIFTY option chain for nearest expiry
    - PCR (OI & Volume)
    - All indices + filtered row for NIFTY

    Returns:
        Dict with DataFrames and scalar PCR values.
    """
    tech = TechNSE()
    deriv = DerivNSE()
    moneycontrol = MoneyControl()

    index_name = INDEX_NAME_MAP.get(index_ticker, index_ticker)

    # 1) All indices + this index row
    all_indices_df = _safe_df(tech.get_all_indices)

    index_row = pd.DataFrame()
    if not all_indices_df.empty:
        cols_lower = [c.lower() for c in all_indices_df.columns]
        name_col = None
        for candidate in ["indexsymbol", "indexname", "symbol", "name"]:
            if candidate in cols_lower:
                name_col = all_indices_df.columns[cols_lower.index(candidate)]
                break
        if name_col is not None:
            index_row = all_indices_df[all_indices_df[name_col] == index_name]

    # 2) India VIX (Moneycontrol) – daily if possible, else 1-min
    try:
        india_vix_df = moneycontrol.get_india_vix(interval="1d")
    except Exception as e:
        print(f"[WARN] MoneyControl.get_india_vix('1d') failed: {e}")
        india_vix_df = _safe_df(moneycontrol.get_india_vix, interval="1")

    # 3) Option chain & PCR for nearest expiry
    try:
        expiry_raw = deriv.get_options_expiry(ticker=index_ticker, is_index=True)
        expiry_dt = _normalize_expiry(expiry_raw)
    except Exception as e:
        print(f"[WARN] Could not get/normalize expiry for {index_ticker}: {e}")
        expiry_dt = None

    if expiry_dt is not None:
        option_chain_df = _safe_df(
            deriv.get_option_chain,
            ticker=index_ticker,
            expiry=expiry_dt,
            is_index=True,
        )

        pcr_oi = _safe_scalar(
            deriv.get_pcr,
            ticker=index_ticker,
            is_index=True,
            on_field="OI",
            expiry=expiry_dt,
        )

        pcr_volume = _safe_scalar(
            deriv.get_pcr,
            ticker=index_ticker,
            is_index=True,
            on_field="Volume",
            expiry=expiry_dt,
        )
    else:
        option_chain_df = pd.DataFrame()
        pcr_oi = None
        pcr_volume = None

    # 4) Block deals (optional flavour; may be empty)
    block_deals_df = _safe_df(tech.get_all_today_block_deals)

    return {
        "index_ticker": index_ticker,
        "index_name": index_name,
        "all_indices": all_indices_df,
        "index_row": index_row,
        "india_vix": india_vix_df,
        "option_chain": option_chain_df,
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_volume,
        "block_deals": block_deals_df,
    }


# -----------------------------------------------------------------------------
# 3) Combine into a single “market state snapshot”
# -----------------------------------------------------------------------------

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


def compute_criticality_score(state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Compute a first-pass Criticality Score C(t) in [0, 1] for NIFTY
    using the existing state dict from build_nifty_market_state().

    Components:
      - dryness_score   : low RV vs its own history
      - ivrv_score      : IV much richer than RV
      - crowding_score  : OI crowded near ATM
      - pcr_score       : PCR far from 1.0 (one-sided book)
    """
    feats = state["state_features"]
    rv_feats = state["rv_features"]
    der_state = state["derivatives_state"]

    # ------------------------------------------------------
    # 1) Dryness score (low 20d RV vs its own distribution)
    # ------------------------------------------------------
    try:
        rv20_series = rv_feats["rv_series"][20].dropna()
        if len(rv20_series) >= 30:
            rv20_last = float(rv20_series.iloc[-1])
            mu = float(rv20_series.mean())
            sigma = float(rv20_series.std(ddof=0))

            if sigma > 0:
                # lower RV20 than usual => positive z => higher dryness_score
                z = (mu - rv20_last) / sigma
                dryness_score = 1.0 / (1.0 + np.exp(-z))  # sigmoid(z)
            else:
                dryness_score = np.nan
        else:
            dryness_score = np.nan
    except Exception as e:
        print(f"[WARN] dryness_score computation failed: {e}")
        dryness_score = np.nan

    # ------------------------------------------------------
    # 2) IV/RV score: how rich is IV vs 20d RV
    # ------------------------------------------------------
    ivrv = feats.get("iv_rv_ratio_20d", np.nan)
    ivrv_score = _norm_01(ivrv, 0.8, 1.5) if ivrv is not None else np.nan

    # ------------------------------------------------------
    # 3) OI crowding score: share of OI within ±2% of spot
    # ------------------------------------------------------
    crowding_score = np.nan
    try:
        oc = der_state["option_chain"]
        last_close = feats["nifty_last_close"]
        if isinstance(oc, pd.DataFrame) and not oc.empty and last_close and last_close > 0:
            cols = oc.columns
            ce_col = next((c for c in cols if "CE_openInterest" in c), None)
            pe_col = next((c for c in cols if "PE_openInterest" in c), None)

            if ce_col and pe_col:
                oc_local = oc.copy()
                oc_local["total_oi"] = oc_local[ce_col].fillna(0) + oc_local[pe_col].fillna(0)
                total_oi = oc_local["total_oi"].sum()

                if total_oi > 0:
                    # strikes: use explicit strikePrice column if present, else index
                    if "strikePrice" in oc_local.columns:
                        strikes = oc_local["strikePrice"]
                    else:
                        strikes = oc_local.index.to_series()

                    lower = last_close * 0.98
                    upper = last_close * 1.02
                    mask = (strikes >= lower) & (strikes <= upper)
                    atm_oi = oc_local.loc[mask, "total_oi"].sum()

                    ratio = atm_oi / total_oi
                    # map ratio in [0.2, 0.6] -> [0, 1]
                    crowding_score = _norm_01(ratio, 0.2, 0.6)
    except Exception as e:
        print(f"[WARN] crowding_score computation failed: {e}")
        crowding_score = np.nan

    # ------------------------------------------------------
    # 4) PCR score: how far is PCR(OI) from 1.0?
    # ------------------------------------------------------
    pcr_oi = feats.get("pcr_oi", None)
    if pcr_oi is not None:
        try:
            if not np.isnan(pcr_oi):
                dev = abs(pcr_oi - 1.0)  # distance from balanced
                pcr_score = _norm_01(dev, 0.0, 0.5)  # deviation 0→0, 0.5→1
            else:
                pcr_score = np.nan
        except TypeError:
            pcr_score = np.nan
    else:
        pcr_score = np.nan

    components = {
        "dryness_score": dryness_score,
        "ivrv_score": ivrv_score,
        "crowding_score": crowding_score,
        "pcr_score": pcr_score,
    }

    # ------------------------------------------------------
    # Weighted combination into C(t)
    # ------------------------------------------------------
    weights = {
        "dryness_score": 0.35,
        "ivrv_score": 0.25,
        "crowding_score": 0.25,
        "pcr_score": 0.15,
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
                # s isn't numeric; skip
                continue

    C = num / den if den > 0 else np.nan
    return C, components


def compute_simple_state_features(
    nifty_hist: pd.DataFrame,
    rv_feats: Dict[str, Any],
    der_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a first-pass feature dict for NIFTY "state of the system".

    This is NOT your final criticality score, just a clean bundle of
    ingredients that you can later transform into C(t).

    Features:
    - Latest close
    - Realized vol (10d/20d/60d)
    - Latest India VIX close
    - Simple IV/RV ratio (VIX / RV_20d)
    - PCR (OI & Volume)
    """
    # Latest NIFTY close
    if "Adj Close" in nifty_hist.columns:
        last_close = float(nifty_hist["Adj Close"].iloc[-1])
    else:
        last_close = float(nifty_hist["Close"].iloc[-1])

    # Realized vols
    rv_latest = rv_feats["rv_latest"]
    rv_10 = rv_latest.get(10, np.nan)
    rv_20 = rv_latest.get(20, np.nan)
    rv_60 = rv_latest.get(60, np.nan)

    # India VIX: use last 'c' from daily data (index value, in percent)
    india_vix_df = der_state["india_vix"]
    if not india_vix_df.empty and "c" in india_vix_df.columns:
        vix_last = float(india_vix_df["c"].iloc[-1])  # e.g. 11.61 (%)
    else:
        vix_last = np.nan

    # Convert VIX to decimal before comparing to RV (which is already decimal)
    if vix_last and not np.isnan(vix_last):
        vix_dec = vix_last / 100.0  # 11.61 -> 0.1161
    else:
        vix_dec = np.nan

    # Simple IV/RV ratio using 20d RV
    if rv_20 and not np.isnan(rv_20) and vix_dec and not np.isnan(vix_dec):
        iv_rv_ratio_20 = vix_dec / rv_20  # ~1.4 if IV > RV
    else:
        iv_rv_ratio_20 = np.nan

    # PCRs
    pcr_oi = der_state.get("pcr_oi", None)
    pcr_vol = der_state.get("pcr_volume", None)

    return {
        "nifty_last_close": last_close,
        "rv_10d": rv_10,
        "rv_20d": rv_20,
        "rv_60d": rv_60,
        "india_vix_last": vix_last,
        "iv_rv_ratio_20d": iv_rv_ratio_20,
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
    }


def build_nifty_market_state(years_hist: int = 3) -> Dict[str, Any]:
    """
    High-level function:
    - Fetch NIFTY history via yfinance
    - Compute realized vol features
    - Fetch derivatives state via Bharat-sm-data
    - Compute simple combined state features

    Returns:
        {
          "as_of": <timestamp>,
          "nifty_history": <DataFrame>,
          "rv_features": {...},
          "derivatives_state": {...},
          "state_features": {...},
        }
    """
    # 1) NIFTY history & RV
    nifty_hist = get_nifty_history_yf(years=years_hist)
    rv_feats = compute_realized_vol_features(nifty_hist, windows=(10, 20, 60))

    # 2) Derivatives state (VIX + option chain + PCR)
    der_state = fetch_nifty_derivatives_state(index_ticker="NIFTY")

    # 3) Simple combined feature dict
    state_feats = compute_simple_state_features(nifty_hist, rv_feats, der_state)

    return {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "nifty_history": nifty_hist,
        "rv_features": rv_feats,
        "derivatives_state": der_state,
        "state_features": state_feats,
    }

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


def categorize_criticality(C: float, components: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict that holds both the numeric value and the bucket/category
    for C(t) and each component.

    Returns:
        {
          "C": {"value": <float>, "category": <str>},
          "dryness_score": {"value": <float>, "category": <str>},
          "ivrv_score": {...},
          "crowding_score": {...},
          "pcr_score": {...},
        }
    """
    result = {}

    # Overall criticality
    result["C"] = {
        "value": C,
        "category": _bucket_score(C),
    }

    # Sub-components
    for name, val in components.items():
        result[name] = {
            "value": val,
            "category": _bucket_score(val),
        }

    return result

def _category_explanation(metric: str, category: str) -> str:
    """
    Return a one-line explanation for a given metric/category combo.

    metric: "C", "dryness_score", "ivrv_score", "crowding_score", "pcr_score"
    category: "Low", "Moderate", "Elevated", "Critical", or "N/A"
    """

    if category == "N/A":
        return "Not enough data to interpret this metric."

    # Overall criticality C(t)
    if metric == "C":
        if category == "Low":
            return "System appears calm with limited build-up of fragility."
        if category == "Moderate":
            return "System is in a normal, watchful regime—neither very calm nor very fragile."
        if category == "Elevated":
            return "System shows elevated fragility; modest shocks can cause outsized moves."
        if category == "Critical":
            return "System is highly fragile; a meaningful trigger could cause large, disorderly moves."

    # Dryness: realized vol vs its own history
    if metric == "dryness_score":
        if category == "Low":
            return "Realized volatility is high vs history—market is already choppy, less scope for quiet risk build-up."
        if category == "Moderate":
            return "Realized volatility is near its historical norm."
        if category == "Elevated":
            return "Realized volatility is below average—conditions are calm enough for risk to quietly build."
        if category == "Critical":
            return "Realized volatility is extremely low—markets look unusually quiet and complacent."

    # IV/RV: implied vs realized vol
    if metric == "ivrv_score":
        if category == "Low":
            return "Implied volatility is in line with or below realized—options are not pricing much extra risk."
        if category == "Moderate":
            return "Implied volatility is slightly richer than realized, consistent with a normal risk premium."
        if category == "Elevated":
            return "Implied volatility is meaningfully richer than realized—options market is pricing in extra risk or events."
        if category == "Critical":
            return "Implied volatility is extremely rich vs realized—strong tension or hedging pressure in the options market."

    # Crowding: OI near ATM
    if metric == "crowding_score":
        if category == "Low":
            return "Open interest is well-distributed across strikes—little crowding around the current level."
        if category == "Moderate":
            return "Some concentration of open interest near spot, but not extreme."
        if category == "Elevated":
            return "Open interest is heavily concentrated near spot—moves can be amplified by hedging flows."
        if category == "Critical":
            return "Open interest is extremely concentrated near spot—very high gamma/crowding risk around the current level."

    # PCR: one-sidedness of positioning
    if metric == "pcr_score":
        if category == "Low":
            return "Put–call open interest is near balanced—no strong one-sided positioning."
        if category == "Moderate":
            return "Put–call open interest shows a mild tilt to one side."
        if category == "Elevated":
            return "Put–call open interest is clearly one-sided—vulnerable to squeezes or sharp adjustments."
        if category == "Critical":
            return "Put–call open interest is extremely one-sided—very vulnerable to forced unwinds or violent reversals."

    # Fallback (shouldn’t hit often)
    return "Category indicates relative risk level for this metric."

def categorize_criticality(C: float, components: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict that holds numeric value, category, and a one-line explanation
    for C(t) and each component.

    Returns:
        {
          "C": {
            "value": <float>,
            "category": <str>,
            "explanation": <str>,
          },
          "dryness_score": {...},
          "ivrv_score": {...},
          "crowding_score": {...},
          "pcr_score": {...},
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Overall C
    C_cat = _bucket_score(C)
    result["C"] = {
        "value": C,
        "category": C_cat,
        "explanation": _category_explanation("C", C_cat),
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

# -----------------------------------------------------------------------------
# 4) Manual test / demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    state = build_nifty_market_state(years_hist=3)

    print("Snapshot as of:", state["as_of"])

    feats = state["state_features"]
    print("\n=== NIFTY Market State (summary) ===")
    print(f"NIFTY last close        : {feats['nifty_last_close']:.2f}")
    print(f"RV 10d (ann.)           : {feats['rv_10d']:.4f}")
    print(f"RV 20d (ann.)           : {feats['rv_20d']:.4f}")
    print(f"RV 60d (ann.)           : {feats['rv_60d']:.4f}")
    print(f"India VIX (last close)  : {feats['india_vix_last']:.2f}")
    print(f"IV/RV 20d ratio         : {feats['iv_rv_ratio_20d']:.2f}")
    print(f"PCR (OI)                : {feats['pcr_oi']}")
    print(f"PCR (Volume)            : {feats['pcr_volume']}")

    C, comp = compute_criticality_score(state)
    crit_summary = categorize_criticality(C, comp)

    print("\n=== Criticality Score C(t) ===")
    C_info = crit_summary["C"]
    C_val = C_info["value"]
    C_cat = C_info["category"]
    C_expl = C_info["explanation"]

    if C_val is not None and not np.isnan(C_val):
        print(f"C(t): {C_val:.2f}  [{C_cat}]  (0 = calm, 1 = highly fragile)")
        print(f"      -> {C_expl}")
    else:
        print("C(t): NaN  [N/A]")
        print(f"      -> {C_expl}")

    print("\nComponents:")
    for name in ["dryness_score", "ivrv_score", "crowding_score", "pcr_score"]:
        info = crit_summary.get(name, {"value": np.nan, "category": "N/A", "explanation": "N/A"})
        val = info["value"]
        cat = info["category"]
        expl = info["explanation"]

        if val is not None and not np.isnan(val):
            print(f"  {name}: {val:.2f}  [{cat}]")
            print(f"      -> {expl}")
        else:
            print(f"  {name}: NaN  [{cat}]")
            print(f"      -> {expl}")