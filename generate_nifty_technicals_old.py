"""
generate_nifty_technicals.py  (v3 – handles MultiIndex columns + NaN -> null)

Fetch Nifty 50 intraday data from Yahoo Finance, compute:
- Bollinger Bands (20, 2)
- RSI (14)
- Awesome Oscillator (5/34 median price)

and write them to docs/data/nifty_technicals.json

Run this from the repo root:

    python generate_nifty_technicals.py

Requirements:
    pip install yfinance pandas
"""

from pathlib import Path
import json

import pandas as pd
import yfinance as yf


SYMBOL = "^NSEI"
PERIOD = "5d"       # last 5 days
INTERVAL = "5m"     # 5-minute candles

OUTPUT_REL_PATH = Path("docs") / "data" / "nifty_technicals.json"


def _normalize_ohlc_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance output to have plain columns:
    Open, High, Low, Close, Adj Close, Volume

    Handles:
    - Single-level columns (already fine)
    - Columns like 'Open/^NSEI'
    - MultiIndex columns like ('Open', '^NSEI')
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Try taking the slice for our symbol on the last level
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        # Or on the first level
        elif symbol in df.columns.get_level_values(0):
            df = df[symbol]

    # If still not standard, flatten 'Open/^NSEI' -> 'Open'
    flat_cols = []
    for col in df.columns:
        c = str(col)
        if "/" in c:
            c = c.split("/", 1)[0]
        flat_cols.append(c)
    df.columns = flat_cols

    expected = ["Open", "High", "Low", "Close"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing OHLC columns after normalization: {missing}. Got: {list(df.columns)}")

    return df


def fetch_nifty_data(symbol: str = SYMBOL, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """Download Nifty data from Yahoo Finance and normalize columns."""
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check symbol/period/interval or your network.")

    df = _normalize_ohlc_columns(df, symbol)

    # Drop timezone info for simplicity
    if df.index.tz is not None:
        df = df.tz_convert(None)
    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands columns: bb_mid, bb_upper, bb_lower."""
    close = df["Close"]
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column using Wilder's smoothing."""
    close = df["Close"]
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (EMA with alpha = 1/period)
    roll_up = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def compute_awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.DataFrame:
    """Add Awesome Oscillator (AO) column.
    AO = SMA(fast, median price) - SMA(slow, median price)
    """
    median_price = (df["High"] + df["Low"]) / 2.0
    sma_fast = median_price.rolling(window=fast).mean()
    sma_slow = median_price.rolling(window=slow).mean()
    df["ao"] = sma_fast - sma_slow
    return df


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'time' column as UNIX timestamp (seconds)."""
    df = df.copy()
    df["time"] = (df.index.view("int64") // 10**9).astype(int)
    return df


def build_output_records(df: pd.DataFrame):
    """Convert DataFrame to a list of dict records suitable for JSON.
    Uses pandas.to_json so NaN -> null, which is valid JSON.
    """
    # Ensure indicator columns exist even if upstream computation failed
    for col in ["bb_mid", "bb_upper", "bb_lower", "rsi", "ao"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Drop rows where we don't have proper price data
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    cols = [
        "time",
        "Open",
        "High",
        "Low",
        "Close",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "rsi",
        "ao",
    ]
    cols = [c for c in cols if c in df.columns]

    df_out = df[cols].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    )

    # Use pandas' JSON serializer to convert NaN -> null
    json_str = df_out.to_json(orient="records")
    records = json.loads(json_str)
    return records


def main():
    df = fetch_nifty_data()
    df = compute_bollinger_bands(df)
    df = compute_rsi(df)
    df = compute_awesome_oscillator(df)
    df = add_time_column(df)

    records = build_output_records(df)

    output_path = (Path(__file__).resolve().parent / OUTPUT_REL_PATH).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} data points to {output_path}")


if __name__ == "__main__":
    main()
