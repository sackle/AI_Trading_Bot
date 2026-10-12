import logging
import uuid
import pandas as pd
from typing import List, Dict, Optional
from bot.brokers.base import BaseBroker
from bot.resilience import CircuitBreaker
import config

logger = logging.getLogger("broker.paper")

class PaperBroker(BaseBroker):
    """
    Virtual paper trading broker for TradingView simulation.
    Maintains a virtual ledger and simulates fills.
    """
    def __init__(self, starting_cash: float = 100000.0, data_provider: BaseBroker | None = None):
        self._cash = starting_cash
        # positions format: { symbol: {"side": "long", "qty": 100, "avg_entry_price": 150.0} }
        self._positions: Dict[str, Dict] = {}
        # We need a data provider for get_bars if we are going to use it
        # For a pure TV-signal paper bot, we might not need to fetch bars, but we should fulfill the interface.
        self._latest_prices = {}
        self._orders: Dict[str, Dict] = {}
        self._data_provider = data_provider
        self.circuit_breaker = CircuitBreaker()

    def get_bars(self, symbol: str, timeframe_amount: int, timeframe_unit: str, limit: int = 300) -> pd.DataFrame:
        if self._data_provider is None:
            logger.error("PaperBroker has no market-data provider configured for %s.", symbol)
            return pd.DataFrame()
        bars = self._data_provider.get_bars(symbol, timeframe_amount, timeframe_unit, limit)
        if not bars.empty:
            self._update_price(symbol, float(bars["close"].iloc[-1]))
        return bars

    def get_bars_batch(self, symbols, timeframe_amount: int, timeframe_unit: str, limit: int = 300):
        if self._data_provider is None:
            return {symbol: pd.DataFrame() for symbol in dict.fromkeys(symbols)}
        frames = self._data_provider.get_bars_batch(symbols, timeframe_amount, timeframe_unit, limit)
        for symbol, bars in frames.items():
            if not bars.empty:
                self._update_price(symbol, float(bars["close"].iloc[-1]))
        return frames

    def get_equity(self) -> float:
        equity = self._cash
        for sym, pos in list(self._positions.items()):
            current_price = self.get_latest_price(sym) or pos["avg_entry_price"]
            # Updating a price can trigger a resting paper stop and remove
            # this position while we are valuing the account.
            if self._positions.get(sym) is not pos:
                continue
            if pos["side"] == "long":
                equity += current_price * pos["qty"]
            else:
                # Short-sale proceeds are already in cash; subtract the
                # current buy-to-cover liability.
                equity -= current_price * pos["qty"]
        return equity

    def get_account_info(self) -> dict:
        return {
            "equity": self.get_equity(),
            "cash": self._cash,
            "buying_power": self._cash * 2,
            "pattern_day_trader": False,
            "daytrade_count": 0,
        }

    def get_positions(self) -> List[dict]:
        positions_list = []
        for sym, pos in self._positions.items():
            positions_list.append({
                "symbol": sym,
                "side": pos["side"],
                "qty": pos["qty"],
                "avg_entry_price": pos["avg_entry_price"]
            })
        return positions_list

    def get_market_clock(self) -> dict:
        if self._data_provider is not None:
            return self._data_provider.get_market_clock()
        import datetime
        return {
            "is_open": True,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "next_open": datetime.datetime.now(datetime.timezone.utc),
            "next_close": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
        }

    def is_market_open(self, symbol: str) -> bool:
        if self._data_provider is not None:
            return self._data_provider.is_market_open(symbol)
        return True
        
    def set_streamer(self, streamer):
        if self._data_provider is not None:
            self._data_provider.set_streamer(streamer)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
        if self._data_provider is not None:
            price = self._data_provider.get_latest_price(symbol)
            if price is not None:
                self._update_price(symbol, float(price))
            return price
        return None

    def _update_price(self, symbol: str, price: float):
        self._latest_prices[symbol] = float(price)
        self._trigger_stop_orders(symbol, float(price))

    def _close_at_price(self, symbol: str, price: float) -> bool:
        position = self._positions.pop(symbol, None)
        if position is None:
            return False
        value = float(price) * float(position["qty"])
        if position["side"] == "long":
            self._cash += value
        else:
            self._cash -= value
        return True

    def _cancel_open_stops(self, symbol: str):
        for order in self._orders.values():
            if order.get("symbol") == symbol and order.get("status") == "accepted":
                order["status"] = "canceled"

    def _trigger_stop_orders(self, symbol: str, price: float):
        """Fill resting protective stops when a newly observed price crosses."""
        for order_id, order in list(self._orders.items()):
            if order.get("symbol") != symbol or order.get("status") != "accepted":
                continue
            position = self._positions.get(symbol)
            if position is None or position["side"] != order.get("position_side"):
                order["status"] = "canceled"
                continue
            hit = (
                position["side"] == "long" and price <= order["stop_price"]
            ) or (
                position["side"] == "short" and price >= order["stop_price"]
            )
            if hit and self._close_at_price(symbol, price):
                order["status"] = "filled"
                order["filled_avg_price"] = float(price)
                self._cancel_open_stops(symbol)
                # Restore the triggered order's terminal state after the
                # helper cancels any other resting stops.
                order["status"] = "filled"
                logger.info("Paper stop %s filled for %s at %.4f", order_id, symbol, price)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> str:
        price = self.get_latest_price(symbol)
        if price is None:
            raise RuntimeError(f"Cannot paper-trade {symbol}: no current market price is available.")
        if symbol in self._positions:
            raise RuntimeError(f"PaperBroker supports one position per symbol; {symbol} is already open.")

        qty = float(qty)
        if qty <= 0:
            raise ValueError("Order quantity must be positive.")
        side = side.lower()
        if side not in {"long", "short"}:
            raise ValueError("Order side must be 'long' or 'short'.")

        fill_price = float(price)
        self._positions[symbol] = {"side": side, "qty": qty, "avg_entry_price": fill_price}
        notional = fill_price * qty
        self._cash += -notional if side == "long" else notional
        order_id = f"paper_{uuid.uuid4().hex[:8]}"
        self._orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "qty": qty,
            "filled_qty": qty,
            "side": side,
            "status": "filled",
            "filled_avg_price": fill_price,
        }
        return order_id

    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        return self.submit_market_order(symbol, qty, side)

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        position = self._positions.get(symbol)
        if position is None:
            raise RuntimeError(f"Cannot protect {symbol}: there is no open paper position.")
        if position["side"] != side:
            raise ValueError("Stop side must match the open position side.")
        if float(qty) <= 0:
            raise ValueError("Order quantity must be positive.")
        order_id = f"paper_stop_{uuid.uuid4().hex[:8]}"
        self._orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "qty": float(qty),
            "position_side": side,
            "stop_price": float(stop_price),
            "status": "accepted",
        }
        return order_id

    def get_fill_price(self, order_id: str, fallback: float, attempts: int = 5, delay: float = 1.0) -> float:
        snapshot = self._orders.get(order_id, {})
        return float(snapshot.get("filled_avg_price", fallback))

    def get_order_snapshot(self, order_id: str) -> Optional[dict]:
        snapshot = self._orders.get(order_id)
        return dict(snapshot) if snapshot is not None else None

    def cancel_order(self, order_id: str):
        order = self._orders.get(order_id)
        if order and order.get("status") == "accepted":
            order["status"] = "canceled"

    def close_position_market(self, symbol: str):
        position = self._positions.get(symbol)
        if position is None:
            return
        price = self.get_latest_price(symbol) or position["avg_entry_price"]
        if self._close_at_price(symbol, float(price)):
            self._cancel_open_stops(symbol)
