"""
Microstructure indicators: VWAP, order flow imbalance, 1-min RSI.

These provide higher-conviction signals by looking at price/volume
structure at a finer granularity than the primary strategy timeframe.
"""
from __future__ import annotations

import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price.

    df must have columns: high, low, close, volume.
    Returns a running VWAP series (expands from the start of the dataframe).
    """
    hlc = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]
    vwap_series = (volume * hlc).cumsum() / volume.cumsum()
    return vwap_series


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling VWAP over a fixed window (resets each window).
    Useful when you want VWAP per 'session' or per N-bar window
    rather than the expanding cumulative VWAP.
    """
    hlc = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]
    pv = (hlc * volume).rolling(window=window).sum()
    v = volume.rolling(window=window).sum()
    return pv / v


def order_flow_imbalance(
    df: pd.DataFrame,
    window: int = 10,
    use_ticker: bool = False,
) -> pd.Series:
    """
    Order flow imbalance: measures buying vs selling pressure.

    If 'bid_size' and 'ask_size' columns exist (level-1 data), uses them
    directly. Otherwise approximates from bar data using the (close - low)
    vs (high - close) ratio as a proxy for buy/sell volume.

    Returns a series in [-1, 1] where:
      - positive = buying pressure
      - negative = selling pressure

    Parameters
    ----------
    df : DataFrame with high, low, close, volume (or bid_size, ask_size)
    window : rolling average window
    use_ticker : if True, treat each bar as a discrete tick imbalance
                 (no rolling). Useful for tick-level data.
    """
    if "bid_size" in df.columns and "ask_size" in df.columns:
        total = df["bid_size"] + df["ask_size"]
        # Buying pressure = ask_size / total (liquidity absorption at ask)
        # Positive = more buying interest at the ask
        imbalance = (df["ask_size"] - df["bid_size"]) / total.where(total > 0, 1)
    else:
        # Bar-level proxy: buying = close - low, selling = high - close
        buy_pressure = (df["close"] - df["low"])
        sell_pressure = (df["high"] - df["close"])
        total = buy_pressure + sell_pressure
        imbalance = (buy_pressure - sell_pressure) / total.where(total > 0, 1)

    if use_ticker:
        return imbalance
    return imbalance.rolling(window=window).mean()


def rsi_fast(series: pd.Series, period: int = 5) -> pd.Series:
    """Fast RSI on 1-min bars (period=5 for ~5-minute momentum)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def price_momentum_confidence(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20) -> float:
    """
    Momentum confidence score (0.0-1.0) from price action.
    Combines short-term momentum direction with longer-term trend
    alignment.

    Returns a float where:
      - >0.5 = bullish setup
      - <0.5 = bearish setup
      - ~0.5 = neutral
    """
    if len(df) < slow_period:
        return 0.5

    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
    rsi_val = rsi_fast(df["close"], period=5).iloc[-1]

    # Trend alignment
    if pd.isna(fast_ema.iloc[-1]) or pd.isna(slow_ema.iloc[-1]):
        return 0.5
    trend_score = 0.5 + (0.15 if fast_ema.iloc[-1] > slow_ema.iloc[-1] else -0.15)

    # RSI contribution
    if not pd.isna(rsi_val):
        rsi_score = 0.5 + (0.1 * (rsi_val - 50) / 50)  # scale -0.1 to +0.1
    else:
        rsi_score = 0.0

    return max(0.0, min(1.0, trend_score + rsi_score))
