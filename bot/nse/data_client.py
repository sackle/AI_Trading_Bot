"""
NSE Data Client — unified wrapper around NSEPython, nsepy, and yfinance.

Provides the same interface as the Alpaca Broker class so existing
strategy code can run unchanged on NSE data:

    client = NSEDataClient()
    bars = client.get_bars("RELIANCE", timeframe="5min", limit=100)
    quote = client.get_quote("RELIANCE")
    chain = client.get_fno_chain("NIFTY")

Data sources (in priority order):
1. nsepy  — stable historical OHLCV (official NSE API)
2. nsepython — live quotes + F&O chain (scrapes nseindia.com)
3. yfinance — fallback for any symbol (.NS suffix added automatically)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("aegis.nse.data")

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if a library is missing
# ---------------------------------------------------------------------------
try:
    from nsepython import nse_eq, nse_fno, equity_history
    _NSEPYTHON_OK = True
except Exception:
    _NSEPYTHON_OK = False
    logger.warning("nsepython not available — live NSE quotes will fall back to yfinance")

try:
    from nsepy import get_history
    from nsepy.symbols import get_symbol_list
    _NSEPY_OK = True
except Exception:
    _NSEPY_OK = False
    logger.warning("nsepy not available — historical NSE data will use yfinance")

try:
    import yfinance as yf
    _YFINANCE_OK = True
except Exception:
    _YFINANCE_OK = False
    logger.warning("yfinance not available")

# Intraday interval map for yfinance
_YF_INTERVAL = {
    "1min": "1m",
    "2min": "2m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "1d": "1d",
}

# How many calendar days back to look for each timeframe to get `limit` bars
_LOOKBACK_DAYS = {
    "1min": 5,
    "2min": 5,
    "5min": 7,
    "15min": 14,
    "30min": 14,
    "1h": 30,
    "1d": 365,
}


def _nse_to_yf(symbol: str) -> str:
    """Add .NS suffix for yfinance if not already present."""
    if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        mapping = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^NSMIDCP"}
        return mapping.get(symbol, f"{symbol}.NS")
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


class NSEDataClient:
    """Unified NSE market data client.

    Falls back gracefully: nsepy → nsepython → yfinance.
    All methods return plain DataFrames / dicts so strategy code is portable.
    """

    def __init__(self):
        self._bar_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._quote_cache: dict[str, tuple[dict, datetime]] = {}
        self._cache_ttl = timedelta(seconds=60)

    # ------------------------------------------------------------------
    # OHLCV bars — primary method used by strategy classes
    # ------------------------------------------------------------------
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Return the most recent `limit` OHLCV bars for `symbol`.

        Returns a DataFrame indexed by UTC timestamp, columns:
        open, high, low, close, volume. Empty DataFrame on failure.
        """
        cache_key = f"{symbol}:{timeframe}:{limit}"
        if cache_key in self._bar_cache:
            df, ts = self._bar_cache[cache_key]
            if datetime.now() - ts < self._cache_ttl:
                return df

        df = pd.DataFrame()

        # --- Try nsepy for daily / swing data ---
        if _NSEPY_OK and timeframe == "1d":
            df = self._bars_nsepy(symbol, limit)

        # --- Try yfinance for intraday or as fallback ---
        if df.empty and _YFINANCE_OK:
            df = self._bars_yfinance(symbol, timeframe, limit)

        if not df.empty:
            self._bar_cache[cache_key] = (df, datetime.now())

        return df

    def _bars_nsepy(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            end = date.today()
            start = end - timedelta(days=max(limit * 2, 365))  # generous window
            df = get_history(symbol=symbol, start=start, end=end)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["open", "high", "low", "close", "volume"]].tail(limit)
        except Exception as exc:
            logger.debug("nsepy failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def _bars_yfinance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            yf_sym = _nse_to_yf(symbol)
            interval = _YF_INTERVAL.get(timeframe, "1d")
            lookback = _LOOKBACK_DAYS.get(timeframe, 60)
            period = f"{lookback}d" if lookback <= 59 else None
            start = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d") if lookback > 59 else None

            ticker = yf.Ticker(yf_sym)
            if period:
                df = ticker.history(period=period, interval=interval)
            else:
                df = ticker.history(start=start, interval=interval)

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["open", "high", "low", "close", "volume"]].tail(limit)
        except Exception as exc:
            logger.debug("yfinance failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Live quote
    # ------------------------------------------------------------------
    def get_quote(self, symbol: str) -> dict | None:
        """Return live quote dict: {symbol, ltp, open, high, low, close, volume, change_pct}."""
        if symbol in self._quote_cache:
            q, ts = self._quote_cache[symbol]
            if datetime.now() - ts < self._cache_ttl:
                return q

        quote = None

        if _NSEPYTHON_OK:
            quote = self._quote_nsepython(symbol)

        if quote is None and _YFINANCE_OK:
            quote = self._quote_yfinance(symbol)

        if quote:
            self._quote_cache[symbol] = (quote, datetime.now())

        return quote

    def _quote_nsepython(self, symbol: str) -> dict | None:
        try:
            data = nse_eq(symbol)
            if not data:
                return None
            pd_data = data.get("priceInfo", {})
            return {
                "symbol": symbol,
                "ltp": float(pd_data.get("lastPrice", 0)),
                "open": float(pd_data.get("open", 0)),
                "high": float(pd_data.get("intraDayHighLow", {}).get("max", 0)),
                "low": float(pd_data.get("intraDayHighLow", {}).get("min", 0)),
                "close": float(pd_data.get("previousClose", 0)),
                "change_pct": float(pd_data.get("pChange", 0)),
                "volume": 0,
                "source": "nsepython",
            }
        except Exception as exc:
            logger.debug("nsepython quote failed for %s: %s", symbol, exc)
            return None

    def _quote_yfinance(self, symbol: str) -> dict | None:
        try:
            yf_sym = _nse_to_yf(symbol)
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            ltp = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if ltp is None:
                return None
            prev_close = getattr(info, "previous_close", ltp)
            return {
                "symbol": symbol,
                "ltp": float(ltp),
                "open": float(getattr(info, "open", ltp)),
                "high": float(getattr(info, "day_high", ltp)),
                "low": float(getattr(info, "day_low", ltp)),
                "close": float(prev_close),
                "change_pct": round((ltp - prev_close) / prev_close * 100, 2) if prev_close else 0.0,
                "volume": int(getattr(info, "three_month_average_volume", 0)),
                "source": "yfinance",
            }
        except Exception as exc:
            logger.debug("yfinance quote failed for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # F&O option chain
    # ------------------------------------------------------------------
    def get_fno_chain(self, symbol: str) -> dict | None:
        """Return the full F&O options chain for `symbol` (index or stock).

        Returns dict with keys:
          expiry_dates, chain (DataFrame with strike, CE_OI, PE_OI, CE_LTP, PE_LTP, CE_IV, PE_IV)
        """
        if not _NSEPYTHON_OK:
            logger.warning("nsepython required for F&O chain data")
            return None
        try:
            from nsepython import option_chain_new  # preferred over nse_fno()
            data = option_chain_new(symbol)
            if not data:
                return None

            records = data.get("records", {})
            expiry_dates = records.get("expiryDates", [])
            raw = records.get("data", [])

            rows = []
            for item in raw:
                ce = item.get("CE", {})
                pe = item.get("PE", {})
                rows.append({
                    "expiry": item.get("expiryDate", ""),
                    "strike": float(item.get("strikePrice", 0)),
                    "CE_OI":    int(ce.get("openInterest", 0)),
                    "CE_chgOI": int(ce.get("changeinOpenInterest", 0)),
                    "CE_LTP":   float(ce.get("lastPrice", 0)),
                    "CE_IV":    float(ce.get("impliedVolatility", 0)),
                    "PE_OI":    int(pe.get("openInterest", 0)),
                    "PE_chgOI": int(pe.get("changeinOpenInterest", 0)),
                    "PE_LTP":   float(pe.get("lastPrice", 0)),
                    "PE_IV":    float(pe.get("impliedVolatility", 0)),
                })

            chain_df = pd.DataFrame(rows)
            return {
                "symbol": symbol,
                "expiry_dates": expiry_dates,
                "chain": chain_df,
                "fetched_at": datetime.now(IST).isoformat(),
            }
        except Exception as exc:
            logger.error("F&O chain fetch failed for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Market hours check
    # ------------------------------------------------------------------
    @staticmethod
    def is_market_open() -> bool:
        """True during NSE regular session (9:15 AM – 3:30 PM IST, Mon–Fri)."""
        now_ist = datetime.now(IST)
        if now_ist.weekday() >= 5:  # Saturday / Sunday
            return False
        session_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        session_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        return session_start <= now_ist <= session_end
