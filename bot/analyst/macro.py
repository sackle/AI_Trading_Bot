from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class MacroResult:
    vix: float
    vix_signal: str
    yield_spread: float
    yield_signal: str
    macro_environment: str
    timestamp: datetime


class MacroMonitor:
    def __init__(self):
        self.cache: MacroResult | None = None
        self.cache_ttl = timedelta(minutes=5)  # CHANGED from 15: refresh macro data 3x faster

    def check(self) -> MacroResult:
        now = datetime.now()
        if self.cache and now - self.cache.timestamp < self.cache_ttl:
            return self.cache
            
        if not YFINANCE_AVAILABLE:
            return self._neutral_result(now)
            
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_hist = vix_ticker.history(period="5d")
            vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
            
            tnx_ticker = yf.Ticker('^TNX')
            tnx_hist = tnx_ticker.history(period="5d")
            tnx = tnx_hist['Close'].iloc[-1] if not tnx_hist.empty else 4.0
            
            irx_ticker = yf.Ticker('^IRX')
            irx_hist = irx_ticker.history(period="5d")
            irx = irx_hist['Close'].iloc[-1] if not irx_hist.empty else 4.0
            
            spread = tnx - irx
            
            vix_sig = "moderate"
            if vix < 15:
                vix_sig = "low_fear"
            elif vix >= 25 and vix <= 35:
                vix_sig = "high_fear"
            elif vix > 35:
                vix_sig = "extreme_fear"
                
            yield_sig = "normal"
            if spread < 0:
                yield_sig = "inverted"
            elif spread < 0.5:
                yield_sig = "flattening"
                
            env = "NEUTRAL"
            if vix_sig in ("high_fear", "extreme_fear") or yield_sig == "inverted":
                env = "RISK_OFF"
            elif vix_sig == "low_fear" and yield_sig == "normal":
                env = "RISK_ON"
                
            res = MacroResult(
                vix=float(vix),
                vix_signal=vix_sig,
                yield_spread=float(spread),
                yield_signal=yield_sig,
                macro_environment=env,
                timestamp=now
            )
            self.cache = res
            return res
        except Exception as e:
            logging.warning(f"Macro check failed: {e}")
            return self._neutral_result(now)

    def _neutral_result(self, now: datetime) -> MacroResult:
        return MacroResult(
            vix=20.0,
            vix_signal="moderate",
            yield_spread=1.0,
            yield_signal="normal",
            macro_environment="NEUTRAL",
            timestamp=now
        )
