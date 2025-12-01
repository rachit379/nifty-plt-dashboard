"""
generate_nifty_technicals.py

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


def fetch_nifty_data(symbol: str = SYMBOL, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """Download Nifty data from Yahoo Finance."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check symbol/period/interval or your network.")
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
    """Add a 'time' column as UNIX timestamp (seconds).
    This is a convenient format for Lightweight Charts.
    """
    # DatetimeIndex in ns -> convert to seconds
    df = df.copy()
    df["time"] = (df.index.view("int64") // 10**9).astype(int)
    return df


def build_output_records(df: pd.DataFrame):
    """Convert DataFrame to a list of dict records suitable for JSON."""
    # Keep only rows where we have all indicator values
    df = df.dropna(subset=["bb_mid", "bb_upper", "bb_lower", "rsi", "ao"])

    # Select and round columns
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
    df_out = df[cols].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    )

    df_out = df_out.round(4)
    return df_out.to_dict(orient="records")


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
