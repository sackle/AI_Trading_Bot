"""
Tracks locally-known open positions and exposes the state the strategies
and risk manager need: is a given symbol currently long/short, its resting
stop order id, its high/low water mark for trailing-stop purposes, etc.

Alpaca remains the source of truth for qty/fills; this class is a
convenience cache plus the extra bookkeeping (stop order id, water mark,
which strategy opened it) that Alpaca's own Position object doesn't carry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bot import logger as trade_logger
import config


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


@dataclass
class OpenPosition:
    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    stop_price: float
    stop_order_id: Optional[str] = None
    extreme_price: float = field(default=0.0)  # running high (long) / low (short) water mark
    strategy: str = ""
    risk_amount: float = 0.0
    notional: float = 0.0
    asset_class: str = ""
    entry_signal: str = ""
    entry_order_id: Optional[str] = None
    adopted: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpenPosition":
        return cls(
            symbol=payload["symbol"],
            side=payload["side"],
            qty=float(payload["qty"]),
            entry_price=float(payload["entry_price"]),
            entry_time=datetime.fromisoformat(payload["entry_time"]),
            atr_at_entry=float(payload.get("atr_at_entry", 0.0)),
            stop_price=float(payload["stop_price"]),
            stop_order_id=payload.get("stop_order_id"),
            extreme_price=float(payload.get("extreme_price", payload["entry_price"])),
            strategy=payload.get("strategy", ""),
            risk_amount=float(payload.get("risk_amount", 0.0)),
            notional=float(payload.get("notional", 0.0)),
            asset_class=payload.get("asset_class", ""),
            entry_signal=payload.get("entry_signal", ""),
            entry_order_id=payload.get("entry_order_id"),
            adopted=bool(payload.get("adopted", False)),
            metadata=payload.get("metadata", {}),
        )

    def current_heat(self) -> float:
        return abs(self.entry_price - self.stop_price) * self.qty


class Portfolio:
    def __init__(self, broker):
        self.broker = broker
        self.positions: Dict[str, OpenPosition] = {}

    # ------------------------------------------------------------------
    def is_long(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "long"

    def is_short(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "short"

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get(self, symbol: str) -> Optional[OpenPosition]:
        return self.positions.get(symbol)

    # ------------------------------------------------------------------
    def open_position(self, position: OpenPosition):
        self.positions[position.symbol] = position

    def record_closed_position(
        self,
        position: OpenPosition,
        exit_price: float,
        reason: str,
        closed_at: datetime | None = None,
    ) -> dict[str, Any]:
        direction = 1 if position.side == "long" else -1
        pnl = (exit_price - position.entry_price) * direction * position.qty

        trade_logger.log_trade(
            timestamp=closed_at or datetime.now(timezone.utc),
            instrument=position.symbol,
            direction=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            position_size=position.qty,
            strategy=position.strategy,
            reason=reason,
        )
        return {"pnl": pnl, "position": position}

    def close_position(self, symbol: str, exit_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        return self.record_closed_position(pos, exit_price=exit_price, reason=reason)

    def update_extreme_price(self, symbol: str, price: float):
        pos = self.positions.get(symbol)
        if pos is None or price is None:
            return
        if pos.side == "long":
            pos.extreme_price = max(pos.extreme_price, price)
        else:
            pos.extreme_price = min(pos.extreme_price, price)

    def position_count(self, strategy: str | None = None, asset_class: str | None = None, side: str | None = None) -> int:
        count = 0
        for position in self.positions.values():
            if strategy and position.strategy != strategy:
                continue
            if asset_class and position.asset_class != asset_class:
                continue
            if side and position.side != side:
                continue
            count += 1
        return count

    def gross_exposure(self) -> float:
        return sum(abs(position.notional or (position.entry_price * position.qty)) for position in self.positions.values())

    def strategy_exposure(self, strategy: str) -> float:
        return sum(
            abs(position.notional or (position.entry_price * position.qty))
            for position in self.positions.values()
            if position.strategy == strategy
        )

    def asset_class_exposure(self, asset_class: str) -> float:
        return sum(
            abs(position.notional or (position.entry_price * position.qty))
            for position in self.positions.values()
            if position.asset_class == asset_class
        )

    def portfolio_heat(self) -> float:
        return sum(position.current_heat() for position in self.positions.values())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {symbol: position.to_dict() for symbol, position in self.positions.items()}

    def load_positions(self, payload: dict[str, dict[str, Any]] | None) -> None:
        self.positions = {}
        for symbol, position_payload in (payload or {}).items():
            self.positions[symbol] = OpenPosition.from_dict(position_payload)

    # ------------------------------------------------------------------
    def sync_from_broker(self):
        """
        Reconciles the local cache against Alpaca's actual open positions.

        Handles the bot restarting with pre-existing positions via the
        ``ADOPTED_POSITION_POLICY`` config knob:

        - ``observe_only`` (default): track for P&L visibility and risk
          exposure, but don't submit exits from strategy signals.  The
          position's ``strategy`` is set to ``"adopted_on_restart"`` so
          ``_process_symbol`` won't try to manage it.
        - ``manage``: adopt and assign a placeholder strategy so the bot
          can evaluate exits normally.
        - ``flatten``: mark the position for immediate closing by the
          caller (returned in the ``"flatten"`` list).

        Positions that closed at the broker while the bot wasn't looking
        (e.g. a resting stop filled) are returned in ``"removed"`` so the
        caller can record the close.
        """
        broker_positions = self.broker.get_positions()
        broker_symbols = set()
        adopted = []
        removed = []
        flatten = []
        policy = config.ADOPTED_POSITION_POLICY

        for p in broker_positions:
            broker_symbols.add(p["symbol"])

            if p["symbol"] in self.positions:
                # Already tracked — update qty / price, skip re-adoption.
                current = self.positions[p["symbol"]]
                current.qty = p["qty"]
                current.entry_price = p["avg_entry_price"]
                current.notional = abs(current.qty * current.entry_price)
                continue

            # New position found at broker but not in our cache.
            position = OpenPosition(
                symbol=p["symbol"],
                side=p["side"],
                qty=p["qty"],
                entry_price=p["avg_entry_price"],
                entry_time=datetime.now(timezone.utc),
                atr_at_entry=0.0,
                stop_price=p["avg_entry_price"],
                extreme_price=p["avg_entry_price"],
                strategy="adopted_on_restart",
                notional=abs(p["qty"] * p["avg_entry_price"]),
                adopted=True,
            )

            if policy == "flatten":
                # Caller will close these immediately.
                flatten.append(position)
                # Still track locally so close_position can find it.
                self.positions[p["symbol"]] = position
            elif policy == "manage":
                # Adopt and mark so strategy signals can manage exits.
                position.strategy = "adopted_managed"
                self.positions[p["symbol"]] = position
                adopted.append(position)
            else:
                # observe_only (default): track but don't manage.
                self.positions[p["symbol"]] = position
                adopted.append(position)

        for symbol in list(self.positions.keys()):
            if symbol not in broker_symbols:
                removed.append(self.positions.pop(symbol))

        return {"adopted": adopted, "removed": removed, "flatten": flatten}

    def get_equity(self) -> float:
        return self.broker.get_equity()
