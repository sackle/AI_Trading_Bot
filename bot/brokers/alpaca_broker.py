"""
Thin wrapper around Alpaca's alpaca-py SDK.

Every alpaca-py-specific detail (client construction, request objects,
crypto vs equity quirks, exception types) lives in this one file, so the
rest of the bot only ever sees plain Python types (floats, dicts,
DataFrames) and never imports `alpaca` directly.

NOTE: alpaca-trade-api (the library originally requested) was deprecated
by Alpaca in favor of alpaca-py, which is what this module uses instead.
See README "Design notes" for details.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.enums import DataFeed
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    _ALPACA_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # keeps offline tests/dashboard importable without alpaca-py
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    MarketOrderRequest = None
    StopOrderRequest = None
    StockHistoricalDataClient = None
    CryptoHistoricalDataClient = None
    StockBarsRequest = None
    CryptoBarsRequest = None
    DataFeed = None
    TimeFrame = None
    TimeFrameUnit = None

    class APIError(Exception):
        pass

    _ALPACA_IMPORT_ERROR = exc

import config
from bot.resilience import CircuitBreaker, retry_on_transient
from bot.brokers.base import BaseBroker
from bot.network import configure_client_timeout

_UNIT_MAP = (
    {
        "Minute": TimeFrameUnit.Minute,
        "Hour": TimeFrameUnit.Hour,
        "Day": TimeFrameUnit.Day,
    }
    if TimeFrameUnit is not None
    else {}
)
_MINUTES_PER_UNIT = {"Minute": 1, "Hour": 60, "Day": 1440}
_TIMEFRAME_ALIASES = {
    "min": "Minute", "mins": "Minute", "minute": "Minute", "minutes": "Minute",
    "m": "Minute", "hour": "Hour", "hours": "Hour", "hr": "Hour", "hrs": "Hour",
    "h": "Hour", "day": "Day", "days": "Day", "d": "Day",
}

_CRYPTO_SYMBOLS = set(config.ALL_CRYPTO_SYMBOLS)
_SESSION_HOURS_PER_DAY = 6.5  # regular US equity session, for lookback sizing only


def _is_crypto(symbol: str) -> bool:
    return symbol in _CRYPTO_SYMBOLS or "/" in symbol


def _int_or_zero(value) -> int:
    """Alpaca may return None for optional counters like daytrade_count."""
    return 0 if value is None else int(value)


def normalize_timeframe_unit(timeframe_unit: str) -> str:
    """Return Alpaca's canonical timeframe spelling or a useful error."""
    raw = str(timeframe_unit).strip()
    if raw in _UNIT_MAP:
        return raw
    normalized = _TIMEFRAME_ALIASES.get(raw.lower())
    if normalized is None:
        supported = ", ".join(_MINUTES_PER_UNIT)
        raise ValueError(f"Unsupported timeframe unit {timeframe_unit!r}; use one of: {supported}.")
    return normalized


def _lookback_days(timeframe_amount: int, timeframe_unit: str, bar_count: int, is_crypto: bool) -> int:
    """
    How many *calendar* days back `start` needs to reach so that the
    window [start, now] contains at least `bar_count` bars.

    This is deliberately generous rather than exact -- get_bars() below
    never trims from the broker's response using a request-side `limit`
    (which would truncate from the *oldest* end of the window, silently
    handing back stale data instead of the most recent bars). Instead it
    fetches everything in this window and trims locally with `.tail()`,
    so erring on the wide side here just means a bit more data transferred,
    never wrong data.
    """
    minutes_per_bar = timeframe_amount * _MINUTES_PER_UNIT[timeframe_unit]
    total_minutes = minutes_per_bar * bar_count

    if is_crypto:  # trades 24/7, so elapsed time ~= bar count * bar length
        days = total_minutes / 1440
        return max(5, int(days * 1.3) + 1)

    # Equities only accrue bars during ~6.5h sessions, ~5 days/week.
    trading_hours_needed = total_minutes / 60
    trading_days_needed = trading_hours_needed / _SESSION_HOURS_PER_DAY
    calendar_days_needed = trading_days_needed * (7 / 5)
    return max(10, int(calendar_days_needed * 1.5) + 5)  # extra buffer for holidays


class AlpacaBroker(BaseBroker):
    """Encapsulates all Alpaca API interactions with transient retry resilience."""

    def __init__(self):
        if TradingClient is None:
            raise RuntimeError(
                "alpaca-py is not installed in this Python environment. "
                "Install requirements.txt or activate the project virtualenv before using the live broker."
            ) from _ALPACA_IMPORT_ERROR
        self.trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )
        self.stock_data_client = StockHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        # Crypto data doesn't require keys, but providing them raises the
        # rate limit, so we pass them anyway.
        self.crypto_data_client = CryptoHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        for client in (self.trading_client, self.stock_data_client, self.crypto_data_client):
            if not configure_client_timeout(client, config.BROKER_REQUEST_TIMEOUT_SECONDS):
                raise RuntimeError(
                    "Could not configure bounded network timeouts for the Alpaca SDK client."
                )

        # Shared circuit breaker — tracks consecutive failures across all
        # broker calls so the main loop can enter degraded mode when the
        # API is persistently unreachable.
        self.circuit_breaker = CircuitBreaker()
        self._bars_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._clock_cache: tuple[float, dict] | None = None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_bars(
        self,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        limit: int = 300,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of the most recent `limit` bars, indexed by
        UTC timestamp ascending (oldest first), columns: open, high, low,
        close, volume. Empty DataFrame if nothing is available yet.
        """
        return self.get_bars_batch(
            [symbol], timeframe_amount, timeframe_unit, limit
        ).get(symbol, pd.DataFrame())

    @retry_on_transient(label="Broker.get_bars_batch")
    def get_bars_batch(
        self,
        symbols: List[str],
        timeframe_amount: int,
        timeframe_unit: str,
        limit: int = 300,
    ) -> dict[str, pd.DataFrame]:
        """Fetch uncached symbols in one Alpaca request per asset class."""
        canonical_unit = normalize_timeframe_unit(timeframe_unit)
        unique_symbols = list(dict.fromkeys(symbols))
        now_monotonic = time.monotonic()
        ttl = max(0.0, config.MARKET_DATA_CACHE_TTL_SECONDS)
        results: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for symbol in unique_symbols:
            cache_key = (symbol, timeframe_amount, canonical_unit, limit)
            cached = self._bars_cache.get(cache_key)
            if cached and now_monotonic - cached[0] <= ttl:
                results[symbol] = cached[1].copy()
            else:
                missing.append(symbol)

        for is_crypto in (False, True):
            group = [symbol for symbol in missing if _is_crypto(symbol) == is_crypto]
            if not group:
                continue
            tf = TimeFrame(timeframe_amount, _UNIT_MAP[canonical_unit])
            start = datetime.now(timezone.utc) - timedelta(
                days=_lookback_days(timeframe_amount, canonical_unit, limit, is_crypto)
            )
            if is_crypto:
                request = CryptoBarsRequest(symbol_or_symbols=group, timeframe=tf, start=start)
                bar_set = self.crypto_data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=group,
                    timeframe=tf,
                    start=start,
                    feed=DataFeed(config.ALPACA_DATA_FEED),
                )
                bar_set = self.stock_data_client.get_stock_bars(request)

            raw = bar_set.df
            for symbol in group:
                if raw.empty:
                    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                else:
                    try:
                        frame = raw.loc[symbol] if isinstance(raw.index, pd.MultiIndex) else raw
                    except KeyError:
                        frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                    else:
                        frame = frame[["open", "high", "low", "close", "volume"]].tail(limit)
                cache_key = (symbol, timeframe_amount, canonical_unit, limit)
                self._bars_cache[cache_key] = (now_monotonic, frame.copy())
                results[symbol] = frame.copy()

        return results

    # ------------------------------------------------------------------
    # Account / positions / clock
    # ------------------------------------------------------------------
    @retry_on_transient(label="Broker.get_equity")
    def get_equity(self) -> float:
        account = self.trading_client.get_account()
        return float(account.equity)

    @retry_on_transient(label="Broker.get_account_info")
    def get_account_info(self) -> dict:
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "pattern_day_trader": bool(account.pattern_day_trader),
            "daytrade_count": _int_or_zero(account.daytrade_count),
        }

    @retry_on_transient(label="Broker.get_positions")
    def get_positions(self) -> List[dict]:
        positions = self.trading_client.get_all_positions()
        normalized = []
        for p in positions:
            side_attr = getattr(p, "side", None)
            side_str = str(side_attr).lower() if side_attr is not None else ""
            qty = float(p.qty)

            if "short" in side_str:
                side = "short"
            elif "long" in side_str:
                side = "long"
            else:
                side = "short" if qty < 0 else "long"  # fallback if side is missing

            normalized.append(
                {
                    "symbol": p.symbol,
                    "side": side,
                    "qty": abs(qty),
                    "avg_entry_price": float(p.avg_entry_price),
                }
            )
        return normalized

    @retry_on_transient(label="Broker.get_market_clock")
    def get_market_clock(self) -> dict:
        now_monotonic = time.monotonic()
        ttl = max(0.0, config.MARKET_CLOCK_CACHE_TTL_SECONDS)
        if self._clock_cache and now_monotonic - self._clock_cache[0] <= ttl:
            return dict(self._clock_cache[1])
        clock = self.trading_client.get_clock()
        result = {
            "is_open": bool(clock.is_open),
            "timestamp": clock.timestamp,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }
        self._clock_cache = (now_monotonic, result)
        return dict(result)

    def is_market_open(self, symbol: str) -> bool:
        if _is_crypto(symbol):
            return True  # crypto trades 24/7 on Alpaca
        return bool(self.get_market_clock()["is_open"])

    def set_streamer(self, streamer):
        self._streamer = streamer

    @retry_on_transient(label="Broker.get_latest_price")
    def get_latest_price(self, symbol: str) -> float | None:
        if hasattr(self, "_streamer") and self._streamer is not None:
            live_price = self._streamer.get_latest_price(symbol)
            if live_price is not None:
                return live_price

        bars = self.get_bars(symbol, timeframe_amount=1, timeframe_unit="Minute", limit=1)
        if bars.empty:
            return None
        return float(bars["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    @retry_on_transient(label="Broker.submit_market_order")
    def submit_market_order(self, symbol: str, qty: float, side: str) -> str:
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        tif = TimeInForce.GTC if _is_crypto(symbol) else TimeInForce.DAY

        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6) if _is_crypto(symbol) else float(int(qty)),
            side=order_side,
            time_in_force=tif,
        )
        order = self.trading_client.submit_order(order_data=request)
        return str(order.id)
        
    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        from alpaca.trading.requests import LimitOrderRequest
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        tif = TimeInForce.GTC if _is_crypto(symbol) else TimeInForce.DAY

        request = LimitOrderRequest(
            symbol=symbol,
            qty=round(qty, 6) if _is_crypto(symbol) else float(int(qty)),
            side=order_side,
            time_in_force=tif,
            limit_price=limit_price,
        )
        order = self.trading_client.submit_order(order_data=request)
        return str(order.id)

    @retry_on_transient(label="Broker.submit_stop_order")
    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        """
        `side` is the position's side (long/short); the order submitted is
        the opposite (closing) side -- SELL to protect a long, BUY to cover
        a short.
        """
        order_side = OrderSide.SELL if side == "long" else OrderSide.BUY

        request = StopOrderRequest(
            symbol=symbol,
            qty=round(qty, 6) if _is_crypto(symbol) else float(int(qty)),
            side=order_side,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
        )
        order = self.trading_client.submit_order(order_data=request)
        return str(order.id)

    def get_fill_price(self, order_id: str, fallback: float, attempts: int = 5, delay: float = 1.0) -> float:
        """Polls a just-submitted order briefly for its actual fill price."""
        for _ in range(attempts):
            try:
                order = self.trading_client.get_order_by_id(order_id)
            except APIError:
                break
            if order.filled_avg_price is not None:
                return float(order.filled_avg_price)
            time.sleep(delay)
        return fallback

    def get_order_snapshot(self, order_id: str) -> dict | None:
        try:
            order = self.trading_client.get_order_by_id(order_id)
        except APIError:
            return None
        return {
            "status": str(getattr(order, "status", "")),
            "filled_qty": (
                float(order.filled_qty)
                if getattr(order, "filled_qty", None) is not None
                else 0.0
            ),
            "filled_avg_price": (
                float(order.filled_avg_price)
                if getattr(order, "filled_avg_price", None) is not None
                else None
            ),
        }

    def cancel_order(self, order_id: str):
        try:
            self.trading_client.cancel_order_by_id(order_id)
        except APIError:
            pass  # already filled or cancelled -- not fatal

    def close_position_market(self, symbol: str):
        try:
            self.trading_client.close_position(symbol)
        except APIError:
            pass  # already flat -- not fatal
