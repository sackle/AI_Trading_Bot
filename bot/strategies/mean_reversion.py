"""
Strategy 1: Mean Reversion (SPY, QQQ), 15-minute candles.

20-period SMA + stdev of closes. Enters counter-trend when price stretches
`std_multiplier` standard deviations away from the mean, on the bet that it
reverts; exits when price returns to the mean. No trailing stop here -- the
universal hard ATR stop (risk_manager.py) is the only downside protection
while a reversion trade is open.
"""

from typing import Optional

import pandas as pd

from bot.indicators import sma, stdev, rsi

class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, symbol: str, std_multiplier: float, lookback: int = 20):
        self.symbol = symbol
        self.std_multiplier = std_multiplier
        self.lookback = lookback
        self.min_bars_required = max(lookback + 2, 15)  # need at least 15 for RSI 14

    def evaluate_entry(self, bars: pd.DataFrame) -> Optional[str]:
        if len(bars) < self.min_bars_required:
            return None

        closes = bars["close"]
        mean = sma(closes, self.lookback).iloc[-1]
        std = stdev(closes, self.lookback).iloc[-1]
        price = closes.iloc[-1]
        
        rsi_series = rsi(closes, period=14)
        current_rsi = rsi_series.iloc[-1]

        if pd.isna(mean) or pd.isna(std) or std == 0 or pd.isna(current_rsi):
            return None

        upper_band = mean + self.std_multiplier * std
        lower_band = mean - self.std_multiplier * std

        if price < lower_band and current_rsi < 35:
            return "long"  # oversold -> expect reversion up
        if price > upper_band and current_rsi > 65:
            return "short"  # overbought -> expect reversion down
        return None

    def evaluate_exit(self, bars: pd.DataFrame, side: str) -> bool:
        """Exit when price has reverted back to the moving average."""
        if len(bars) < self.lookback:
            return False

        mean = sma(bars["close"], self.lookback).iloc[-1]
        price = bars["close"].iloc[-1]

        if pd.isna(mean):
            return False

        return bool(price >= mean) if side == "long" else bool(price <= mean)
