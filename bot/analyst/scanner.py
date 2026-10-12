from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ScanResult:
    symbol: str
    rsi: float
    rsi_signal: str
    macd_signal: str
    macd_histogram: float
    volume_ratio: float
    volume_signal: str
    timestamp: datetime


class MarketScanner:
    def scan(self, symbol: str, bars: pd.DataFrame) -> ScanResult:
        now = datetime.now()
        if bars.empty or len(bars) < 35:
            return ScanResult(symbol, 50.0, "neutral", "neutral", 0.0, 1.0, "normal", now)
            
        close = bars['close']
        volume = bars['volume']
        
        rsi = self._compute_rsi(close)
        macd_line, signal_line, hist = self._compute_macd(close)
        
        current_rsi = rsi.iloc[-1]
        rsi_sig = "neutral"
        if pd.notna(current_rsi):
            if current_rsi > 70:
                rsi_sig = "overbought"
            elif current_rsi < 30:
                rsi_sig = "oversold"
                
        current_hist = hist.iloc[-1]
        macd_sig = "neutral"
        if pd.notna(current_hist):
            if current_hist > 0 and hist.iloc[-2] <= 0:
                macd_sig = "bullish"
            elif current_hist < 0 and hist.iloc[-2] >= 0:
                macd_sig = "bearish"
                
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / vol_avg if pd.notna(vol_avg) and vol_avg > 0 else 1.0
        
        vol_sig = "normal"
        if vol_ratio > 2.0:
            vol_sig = "high"
        elif vol_ratio < 0.5:
            vol_sig = "low"
            
        return ScanResult(
            symbol=symbol,
            rsi=float(current_rsi) if pd.notna(current_rsi) else 50.0,
            rsi_signal=rsi_sig,
            macd_signal=macd_sig,
            macd_histogram=float(current_hist) if pd.notna(current_hist) else 0.0,
            volume_ratio=float(vol_ratio),
            volume_signal=vol_sig,
            timestamp=now
        )

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
