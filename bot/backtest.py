"""
Historical backtest and replay engine for the current Alpaca strategy set.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from bot.artifacts import save_latest_backtest_summary
from bot.indicators import atr as calc_atr
from bot.market_filters import MarketFilters
from bot.portfolio import OpenPosition, Portfolio
from bot.risk_manager import RiskManager
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy


def build_strategies() -> dict:
    from bot.strategies.red_green_breakout import RedGreenBreakoutStrategy
    return {
        "^NSEBANK": {
            "strategy": RedGreenBreakoutStrategy("^NSEBANK"),
            "timeframe_amount": 5,
            "timeframe_unit": "Min",
            "trail_atr_multiple": 2.0
        }
    }


def _parse_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value)
    

def _timeframe_label(spec: dict) -> str:
    return f"{spec['timeframe_amount']} {spec['timeframe_unit']}"

def load_market_data(path, default_symbol=None):
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    if default_symbol:
        return {default_symbol: df}
    return {"UNKNOWN": df}

class SimulatedPortfolio(Portfolio):
    def __init__(self, initial_equity=100000.0):
        super().__init__(broker=None)
        self.equity = initial_equity
        self.cash = initial_equity
        self.open_positions = {}
        
    def get_equity(self):
        return self.equity
        
@dataclass
class BacktestResult:
    summary: dict
    trades: list
    replay_lines: list
    output_dir: str

class BacktestRunner:
    def __init__(
        self,
        *,
        data_path: str | Path,
        default_symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        initial_equity: float = config.BACKTEST_INITIAL_EQUITY,
        replay: bool = False,
    ):
        self.data_path = Path(data_path)
        self.default_symbol = default_symbol
        self.start = _parse_optional_timestamp(start)
        self.end = _parse_optional_timestamp(end)
        self.initial_equity = float(initial_equity)
        self.replay = replay

        self.market_filters = MarketFilters()
        self.risk_manager = RiskManager()
        self.portfolio = SimulatedPortfolio()
        self.strategies = build_strategies()
        self.market_data = load_market_data(self.data_path, default_symbol=default_symbol)
        filtered_market_data = {}
        for symbol, frame in self.market_data.items():
            filtered = frame
            if self.start is not None:
                filtered = filtered[filtered.index >= self.start]
            if self.end is not None:
                filtered = filtered[filtered.index <= self.end]
            filtered_market_data[symbol] = filtered
        self.market_data = filtered_market_data
        self.market_data = {symbol: frame for symbol, frame in self.market_data.items() if not frame.empty}

        self.current_date: str | None = None
        self.day_start_equity = self.initial_equity
        self.daily_realized_pnl = 0.0
        self.cumulative_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.wins = 0
        self.losses = 0
        self.new_entries_allowed = True
        self.entry_halt_reason = ""
        self.latest_prices: dict[str, float] = {}
        self.trades: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.replay_lines: list[str] = []
        self.equity_curve: list[dict[str, Any]] = []

    def current_equity(self) -> float:
        unrealized = 0.0
        for position in self.portfolio.positions.values():
            latest_price = self.latest_prices.get(position.symbol, position.entry_price)
            direction = 1 if position.side == "long" else -1
            unrealized += (latest_price - position.entry_price) * direction * position.qty
        return self.initial_equity + self.cumulative_realized_pnl + unrealized

    def run(self) -> BacktestResult:
        timeline = []
        for symbol, frame in self.market_data.items():
            if symbol not in self.strategies:
                continue
            for index, timestamp in enumerate(frame.index):
                timeline.append((timestamp, symbol, index))
        timeline.sort(key=lambda item: (item[0], item[1]))

        for timestamp, symbol, index in timeline:
            spec = self.strategies[symbol]
            frame = self.market_data[symbol]
            bars = frame.iloc[: index + 1]
            latest_bar = bars.iloc[-1]
            self.latest_prices[symbol] = float(latest_bar["close"])
            self._roll_day_if_needed(timestamp)
            self._refresh_entry_gate(timestamp)
            self._process_symbol_bar(symbol, spec, bars, latest_bar, timestamp)
            self.equity_curve.append({"timestamp": timestamp.isoformat(), "equity": round(self.current_equity(), 4)})

        summary = self._build_summary()
        output_dir = self._persist_results(summary)
        return BacktestResult(summary=summary, trades=self.trades, replay_lines=self.replay_lines, output_dir=output_dir)

    def _roll_day_if_needed(self, timestamp: pd.Timestamp):
        date_value = timestamp.date().isoformat()
        if date_value == self.current_date:
            return
        self.current_date = date_value
        self.day_start_equity = self.current_equity()
        self.daily_realized_pnl = 0.0
        self.new_entries_allowed = True
        self.entry_halt_reason = ""
        self._record_event(timestamp, "day_roll", f"Started backtest day {date_value}.")

    def _refresh_entry_gate(self, timestamp: pd.Timestamp):
        reasons = []
        equity_now = self.current_equity()
        if self.day_start_equity:
            drawdown = equity_now - self.day_start_equity
            daily_limit = self.day_start_equity * config.MAX_DAILY_LOSS_PCT_OF_EQUITY
            if drawdown <= -daily_limit:
                reasons.append(f"Daily loss cap reached ({drawdown:.2f}).")
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            reasons.append("Consecutive loss cap reached.")

        allowed = not reasons
        reason = " ".join(reasons)
        if allowed != self.new_entries_allowed or (not allowed and reason != self.entry_halt_reason):
            self.new_entries_allowed = allowed
            self.entry_halt_reason = reason
            self._record_event(timestamp, "entry_gate_changed", reason or "Entry gate reopened.")

    def _process_symbol_bar(self, symbol: str, spec: dict, bars: pd.DataFrame, latest_bar, timestamp: pd.Timestamp):
        strategy = spec["strategy"]
        if len(bars) < strategy.min_bars_required:
            return

        position = self.portfolio.get(symbol)
        if position is not None:
            self._update_trailing_stop(position, latest_bar, spec["trail_atr_multiple"])
            stop_hit, stop_price = self._stop_hit(position, latest_bar)
            if stop_hit:
                self._close_position(position, stop_price, "hard_stop", timestamp)
                position = None
            elif strategy.evaluate_exit(bars, position.side):
                self._close_position(position, float(latest_bar["close"]), "strategy_exit", timestamp)
                position = None

        if position is None:
            entry_side = strategy.evaluate_entry(bars)
            if entry_side in ("long", "short"):
                self._try_open_position(symbol, spec, bars, entry_side, timestamp)

    def _update_trailing_stop(self, position: OpenPosition, latest_bar, trail_atr_multiple: float | None):
        extreme = float(latest_bar["high"] if position.side == "long" else latest_bar["low"])
        self.portfolio.update_extreme_price(position.symbol, extreme)
        if not trail_atr_multiple:
            return
        position.stop_price = self.risk_manager.update_trailing_stop(
            current_stop=position.stop_price,
            extreme_price=position.extreme_price,
            atr_value=position.atr_at_entry,
            trail_multiple=trail_atr_multiple,
            side=position.side,
        )

    def _stop_hit(self, position: OpenPosition, latest_bar) -> tuple[bool, float]:
        if position.side == "long" and float(latest_bar["low"]) <= position.stop_price:
            return True, float(position.stop_price)
        if position.side == "short" and float(latest_bar["high"]) >= position.stop_price:
            return True, float(position.stop_price)
        return False, position.stop_price

    def _slipped_fill(self, price: float, side: str) -> float:
        bps = config.BACKTEST_DEFAULT_SLIPPAGE_BPS / 10000.0
        return price * (1 + bps) if side == "long" else price * (1 - bps)

    def _try_open_position(self, symbol: str, spec: dict, bars: pd.DataFrame, side: str, timestamp: pd.Timestamp):
        strategy_name = spec["strategy"].name
        atr_value = calc_atr(bars, period=config.ATR_PERIOD).iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            return

        raw_price = float(bars["close"].iloc[-1])
        equity = self.current_equity()
        sizing = self.risk_manager.size_position(
            equity=equity,
            atr_value=float(atr_value),
            price=raw_price,
            side=side,
            fractionable="/" in symbol,
        )
        if sizing is None:
            return

        filter_decision = self.market_filters.evaluate(
            symbol=symbol,
            strategy_name=strategy_name,
            side=side,
            bars=bars,
            atr_value=float(atr_value),
        )
        if not filter_decision.allowed:
            self._record_replay(timestamp, f"{symbol} {side} filtered: {' '.join(filter_decision.reasons)}")
            return

        guardrails = self.risk_manager.evaluate_entry_guardrails(
            symbol=symbol,
            strategy=strategy_name,
            side=side,
            equity=equity,
            candidate_notional=sizing.notional,
            candidate_risk=sizing.risk_amount,
            asset_class=config.symbol_asset_class(symbol),
            portfolio=self.portfolio,
            daily_realized_pnl=self.daily_realized_pnl,
            day_start_equity=self.day_start_equity,
            consecutive_losses=self.consecutive_losses,
            new_entries_allowed=self.new_entries_allowed,
            entry_halt_reason=self.entry_halt_reason,
        )
        if not guardrails.allowed:
            self._record_replay(timestamp, f"{symbol} {side} blocked: {' '.join(guardrails.reasons)}")
            return

        fill_price = self._slipped_fill(raw_price, side)
        stop_price = self.risk_manager.initial_stop_price(fill_price, float(atr_value), side)
        position = OpenPosition(
            symbol=symbol,
            side=side,
            qty=sizing.qty,
            entry_price=fill_price,
            entry_time=timestamp.to_pydatetime(),
            atr_at_entry=float(atr_value),
            stop_price=stop_price,
            extreme_price=fill_price,
            strategy=strategy_name,
            risk_amount=sizing.risk_amount,
            notional=float(fill_price * sizing.qty),
            asset_class=config.symbol_asset_class(symbol),
            entry_signal=f"{strategy_name}:{side}",
        )
        self.portfolio.open_position(position)
        self._record_event(timestamp, "position_opened", f"{symbol} {side} opened at {fill_price:.2f}.")
        self._record_replay(timestamp, f"{symbol} {side} ENTRY @ {fill_price:.2f} via {strategy_name}")

    def _close_position(self, position: OpenPosition, exit_price: float, reason: str, timestamp: pd.Timestamp):
        result = self.portfolio.close_position(position.symbol, exit_price, reason)
        if result is None:
            return
        pnl = float(result["pnl"])
        self.cumulative_realized_pnl += pnl
        self.daily_realized_pnl += pnl
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1

        trade = {
            "timestamp": timestamp.isoformat(),
            "instrument": position.symbol,
            "direction": position.side,
            "entry_price": round(position.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "profit_loss": round(pnl, 2),
            "position_size": position.qty,
            "strategy": position.strategy,
            "reason": reason,
        }
        self.trades.append(trade)
        self._record_event(timestamp, "position_closed", f"{position.symbol} closed via {reason} for {pnl:+.2f}.")
        self._record_replay(timestamp, f"{position.symbol} EXIT @ {exit_price:.2f} ({reason}) P/L {pnl:+.2f}")

    def _record_event(self, timestamp: pd.Timestamp, event_type: str, message: str):
        self.events.append({"timestamp": timestamp.isoformat(), "type": event_type, "message": message})

    def _record_replay(self, timestamp: pd.Timestamp, message: str):
        if self.replay:
            self.replay_lines.append(f"{timestamp.isoformat()} {message}")

    def _build_summary(self) -> dict[str, Any]:
        pnl_values = [trade["profit_loss"] for trade in self.trades]
        total_trades = len(self.trades)
        win_rate = (self.wins / total_trades * 100.0) if total_trades else 0.0
        equity_points = [row["equity"] for row in self.equity_curve] or [self.initial_equity]

        running_peak = equity_points[0]
        max_drawdown = 0.0
        for equity in equity_points:
            running_peak = max(running_peak, equity)
            max_drawdown = min(max_drawdown, equity - running_peak)

        summary = {
            "data_path": str(self.data_path),
            "start": self.equity_curve[0]["timestamp"] if self.equity_curve else "",
            "end": self.equity_curve[-1]["timestamp"] if self.equity_curve else "",
            "starting_equity": round(self.initial_equity, 2),
            "ending_equity": round(self.current_equity(), 2),
            "realized_pnl": round(self.cumulative_realized_pnl, 2),
            "total_trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(win_rate, 1),
            "best_trade": round(max(pnl_values), 2) if pnl_values else 0.0,
            "worst_trade": round(min(pnl_values), 2) if pnl_values else 0.0,
            "max_drawdown": round(max_drawdown, 2),
            "open_positions_left": self.portfolio.position_count(),
            "symbols_tested": sorted(symbol for symbol in self.market_data if symbol in self.strategies),
        }
        return summary

    def _persist_results(self, summary: dict[str, Any]) -> Path:
        config.ensure_runtime_directories()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.runtime_path(config.BACKTESTS_DIR)) / f"run_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        trades_path = output_dir / "trades.csv"
        with open(trades_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "instrument",
                    "direction",
                    "entry_price",
                    "exit_price",
                    "profit_loss",
                    "position_size",
                    "strategy",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(self.trades)

        events_path = output_dir / "events.jsonl"
        with open(events_path, "w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

        equity_path = output_dir / "equity_curve.csv"
        with open(equity_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "equity"])
            writer.writeheader()
            writer.writerows(self.equity_curve[-config.BACKTEST_MAX_ROWS_IN_SUMMARY :])

        save_latest_backtest_summary(summary)
        return output_dir
