import threading
import asyncio
from datetime import datetime, timezone
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
import config

class MarketStreamer:
    """
    Background WebSocket engine that maintains zero-latency live prices and bars.
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.live_prices = {}
        self.order_book_imbalance = {}
        self._lock = threading.RLock()
        self._events = {symbol: {} for symbol in symbols}
        self._last_error = ""
        self.thread = None
        self.stream = StockDataStream(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            feed=DataFeed(config.ALPACA_DATA_FEED),
        )

    def _record_event(self, kind, event):
        received_at = datetime.now(timezone.utc)
        event_time = getattr(event, "timestamp", None)
        if event_time is not None:
            try:
                event_time = event_time.isoformat()
            except AttributeError:
                event_time = str(event_time)
        with self._lock:
            symbol = getattr(event, "symbol", "")
            if symbol:
                self._events.setdefault(symbol, {})[kind] = {
                    "received_at": received_at.isoformat(),
                    "event_at": event_time,
                }
        
    def start(self):
        async def quote_callback(q):
            total_size = q.bid_size + q.ask_size
            if total_size > 0:
                # OBI ranges from -1.0 (heavy sell pressure) to +1.0 (heavy buy pressure)
                with self._lock:
                    self.order_book_imbalance[q.symbol] = (q.bid_size - q.ask_size) / total_size
            self._record_event("quote", q)
            
        async def trade_callback(t):
            with self._lock:
                self.live_prices[t.symbol] = t.price
            self._record_event("trade", t)
            
        async def bar_callback(b):
            with self._lock:
                self.live_prices[b.symbol] = b.close
            self._record_event("bar", b)
            
        try:
            # Subscribe to quotes for L2 Order Book Imbalance.
            self.stream.subscribe_quotes(quote_callback, *self.symbols)
            self.stream.subscribe_trades(trade_callback, *self.symbols)
            self.stream.subscribe_bars(bar_callback, *self.symbols)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        
        def _run_loop():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.stream.run()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            
        self.thread = threading.Thread(target=_run_loop, daemon=True, name="Alpaca-WS")
        self.thread.start()

    def get_latest_price(self, symbol: str) -> float:
        with self._lock:
            return self.live_prices.get(symbol)

    def get_order_book_imbalance(self, symbol: str) -> float:
        with self._lock:
            return self.order_book_imbalance.get(symbol, 0.0)

    def get_status(self) -> dict:
        """Return safe, serializable feed diagnostics for the dashboard."""
        with self._lock:
            events = {symbol: dict(values) for symbol, values in self._events.items()}
            error = self._last_error
        alive = bool(self.thread and self.thread.is_alive())
        last_event_at = max(
            (
                item.get("received_at")
                for symbol_events in events.values()
                for item in symbol_events.values()
                if item.get("received_at")
            ),
            default=None,
        )
        event_age_seconds = None
        if last_event_at:
            event_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - datetime.fromisoformat(last_event_at)).total_seconds(),
            )
        status = (
            "error" if error else
            "stopped" if not alive else
            "waiting" if event_age_seconds is None else
            "receiving" if event_age_seconds <= 120 else
            "stale"
        )
        return {
            "feed": config.ALPACA_DATA_FEED.upper(),
            "thread_alive": alive,
            "status": status,
            "last_event_at": last_event_at,
            "event_age_seconds": round(event_age_seconds, 1) if event_age_seconds is not None else None,
            "last_error": error,
            "symbols": events,
        }
