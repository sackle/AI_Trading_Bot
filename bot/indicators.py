"""
Shared technical indicator calculations used across strategies and the risk
manager. Pure pandas/numpy, no Alpaca dependency -- this makes them easy to
unit test offline with synthetic data (see tests/ for examples).
"""

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def stdev(series: pd.Series, period: int) -> pd.Series:
    """Rolling (population-style, pandas default ddof=1) standard deviation."""
    return series.rolling(window=period).std()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing (an EMA with alpha=1/period).
    `df` must have columns: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def donchian_channel(df: pd.DataFrame, period: int):
    """Returns (rolling_high, rolling_low) of the high/low columns over `period` bars."""
    rolling_high = df["high"].rolling(window=period).max()
    rolling_low = df["low"].rolling(window=period).min()
    return rolling_high, rolling_low


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).
    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent of a time series using the R/S method.
    H < 0.5: mean reverting, H = 0.5: random walk, H > 0.5: trending.
    Returns NaN if the series is too short.
    """
    import numpy as np

    ts = series.dropna().values
    if len(ts) < max_lag * 2:
        return float('nan')

    lags = range(2, max_lag + 1)
    rs_values = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, len(ts) - lag + 1, lag)]
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 2:
        return float('nan')

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    hurst, _ = np.polyfit(log_lags, log_rs, 1)
    return float(hurst)


def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.Series:
    """
    Volume-at-price distribution. Returns a Series indexed by price bin
    mid-points with total volume at each level.
    `df` must have columns: close, volume.
    """
    import numpy as np

    prices = df["close"]
    volumes = df["volume"]
    price_bins = np.linspace(prices.min(), prices.max(), bins + 1)
    indices = np.digitize(prices, price_bins) - 1
    indices = np.clip(indices, 0, bins - 1)

    midpoints = (price_bins[:-1] + price_bins[1:]) / 2
    profile = pd.Series(0.0, index=midpoints, name="volume_profile")
    for idx, vol in zip(indices, volumes):
        profile.iloc[idx] += vol
    return profile
