"""
Strategy 3: Trend Following (GLD, USO), 4-hour candles.

Classic 50/200 EMA crossover. A golden cross (50 EMA crosses above 200 EMA)
opens a long. A death cross (50 EMA crosses below 200 EMA) exits an open
long and, if `allow_short` is enabled, opens a fresh short (mirrored the
other way for an open short). Going short GLD/USO requires a margin
account with those shares marginable/shortable -- see README.

Exit/risk management: an initial hard ATR stop plus a trailing stop of
`trail_atr_multiple` ATRs, both handled by risk_manager.py / portfolio.py.
"""

from typing import Optional, Tuple

import pandas as pd

from bot.indicators import ema, macd, volume_profile


class TrendFollowingStrategy:
    name = "trend_following"
    min_bars_required = 50

    def __init__(
        self,
        symbol: str,
        fast_period: int = 50,
        slow_period: int = 200,
        allow_short: bool = True,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.allow_short = allow_short

    def _crosses(self, bars: pd.DataFrame) -> Tuple[Optional[bool], Optional[bool]]:
        closes = bars["close"]
        fast = ema(closes, self.fast_period)
        slow = ema(closes, self.slow_period)

        curr_fast, prev_fast = fast.iloc[-1], fast.iloc[-2]
        curr_slow, prev_slow = slow.iloc[-1], slow.iloc[-2]

        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
            return None, None

        golden_cross = bool(prev_fast <= prev_slow and curr_fast > curr_slow)
        death_cross = bool(prev_fast >= prev_slow and curr_fast < curr_slow)
        return golden_cross, death_cross

    def evaluate_entry(self, bars: pd.DataFrame) -> Optional[str]:
        if len(bars) < self.min_bars_required:
            return None

        golden_cross, death_cross = self._crosses(bars)
        if not golden_cross and not death_cross:
            return None

        # MACD Confirmation
        macd_line, signal_line, hist = macd(bars["close"])
        if len(hist) < 2 or pd.isna(hist.iloc[-1]):
            return None
            
        # A fresh EMA crossover can lead the slower MACD histogram by one or
        # two candles.  Accept either a histogram already on the matching
        # side of zero or a histogram moving decisively in that direction;
        # otherwise valid crossovers were silently discarded.
        macd_bullish = hist.iloc[-1] > 0 or hist.iloc[-1] > hist.iloc[-2]
        macd_bearish = hist.iloc[-1] < 0 or hist.iloc[-1] < hist.iloc[-2]
        
        # Volume Confirmation
        vol_sma = bars["volume"].rolling(20).mean()
        curr_vol = bars["volume"].iloc[-1]
        vol_surge = curr_vol > vol_sma.iloc[-1] * 1.2  # 20% above average volume

        if golden_cross and macd_bullish and vol_surge:
            return "long"
        if death_cross and self.allow_short and macd_bearish and vol_surge:
            return "short"
        return None

    def evaluate_exit(self, bars: pd.DataFrame, side: str) -> bool:
        if len(bars) < self.min_bars_required:
            return False

        golden_cross, death_cross = self._crosses(bars)
        if golden_cross is None:
            return False

        return death_cross if side == "long" else golden_cross
