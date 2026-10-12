"""
NSE Trading Loop — mirrors the US TradingBot but runs on Indian market hours.

Runs under the shared runtime supervisor in ``run_bot.py``.  Its INR paper
ledger remains intentionally separate from the USD Alpaca account, but worker
health and the operator entry gate are coordinated with the US loop.

Features:
- Intraday strategies (5-min bars) during 9:15 AM – 3:30 PM IST
- Swing strategies (daily bars) with end-of-day signal evaluation
- F&O scanner signals used as entry confirmation layer
- Paper-trade execution via NSEPaperBroker
- Logs signals/trades to runtime/nse_trades.csv + runtime/nse_signals.csv
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config
from bot.nse.data_client import NSEDataClient
from bot.nse.broker_nse import NSEPaperBroker
from bot.nse.fno_scanner import FnOScanner
from bot.indicators import atr as calc_atr
from bot.risk_manager import RiskManager
from bot.portfolio import Portfolio, OpenPosition
from bot.runtime_coordinator import RuntimeCoordinator
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.artifacts import log_event

logger = logging.getLogger("aegis.nse.loop")
IST = ZoneInfo("Asia/Kolkata")

NSE_SIGNALS_LOG = os.path.join(config.ROOT_DIR, "runtime", "nse_signals.csv")


def _ensure_signals_log():
    if not Path(NSE_SIGNALS_LOG).exists():
        with open(NSE_SIGNALS_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "symbol", "strategy", "side",
                "status", "price", "reason", "timeframe"
            ])


def _log_nse_signal(symbol, strategy, side, status, price, reason, timeframe):
    _ensure_signals_log()
    with open(NSE_SIGNALS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol, strategy, side, status,
            round(float(price), 4), reason, timeframe
        ])


def _build_nse_strategies() -> dict:
    """Build strategy instances for all configured NSE symbols."""
    strategies = {}

    # Intraday mean reversion on NSE equities (5-min bars)
    for symbol in getattr(config, "NSE_EQUITY_SYMBOLS", []):
        params = config.MEAN_REVERSION_PARAMS.get(symbol, {"std_multiplier": 1.5})
        strategies[symbol] = {
            "strategy": MeanReversionStrategy(
                symbol=symbol,
                std_multiplier=params["std_multiplier"],
                lookback=config.MEAN_REVERSION_LOOKBACK,
            ),
            "timeframe": getattr(config, "NSE_INTRADAY_TIMEFRAME", "5min"),
            "trail_atr_multiple": 1.5,
            "instrument_type": "equity",
        }

    # Swing trend following on NSE equities (daily bars)
    for symbol in getattr(config, "NSE_SWING_SYMBOLS", []):
        strategies[f"{symbol}_swing"] = {
            "strategy": TrendFollowingStrategy(
                symbol=symbol,
                fast_period=config.TREND_FAST_EMA,
                slow_period=config.TREND_SLOW_EMA,
                allow_short=False,  # no shorting in NSE cash without SLB
            ),
            "timeframe": "1d",
            "trail_atr_multiple": config.TREND_TRAIL_ATR_MULTIPLIER,
            "instrument_type": "equity",
            "_real_symbol": symbol,
        }

    return strategies


class NSETradingBot:
    """Restart-safe NSE paper loop managed by the common runtime supervisor."""

    def __init__(
        self,
        coordinator: RuntimeCoordinator | None = None,
        *,
        data_client: NSEDataClient | None = None,
        broker: NSEPaperBroker | None = None,
    ):
        self.coordinator = coordinator
        self.data_client = data_client or NSEDataClient()
        self.broker = broker or NSEPaperBroker()
        self.risk_manager = RiskManager()
        self.fno_scanner = FnOScanner()
        self.portfolio = Portfolio(broker=self.broker)
        # The persistent broker ledger is canonical across a restart.  Keep
        # Portfolio as the strategy convenience cache only.
        self.portfolio.load_positions(
            {
                position.symbol: position.to_dict()
                for position in self.broker.get_open_position_models()
            }
        )
        self.strategies = _build_nse_strategies()
        self.last_processed_bar: dict[str, pd.Timestamp | None] = {
            k: None for k in self.strategies
        }
        _ensure_signals_log()
        logger.info(
            "NSE Trading Bot initialised — %d strategies (%d symbols)",
            len(self.strategies),
            len(set(
                spec.get("_real_symbol", sym.replace("_swing", ""))
                for sym, spec in self.strategies.items()
            )),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _heartbeat(self, status: str) -> None:
        if self.coordinator is None:
            return
        account = self.broker.get_account_info()
        self.coordinator.heartbeat(
            "nse",
            status=status,
            metrics={
                "currency": "INR",
                "equity": round(float(account["equity"]), 2),
                "cash": round(float(account["cash"]), 2),
                "open_positions": self.portfolio.position_count(),
            },
        )

    def run_forever(self, stop_event: threading.Event | None = None):
        """Blocking loop with cooperative shutdown for the runtime supervisor."""
        stop_event = stop_event or threading.Event()
        mode_tag = "PAPER"
        logger.info("NSE bot starting (%s mode)...", mode_tag)
        log_event("nse_startup", f"NSE bot started in {mode_tag} mode.")

        while not stop_event.is_set():
            try:
                if self.data_client.is_market_open():
                    self._tick()
                    self._heartbeat("running")
                else:
                    self._heartbeat("idle")
                    logger.debug("NSE market closed — sleeping 60s")
                    stop_event.wait(min(60, getattr(config, "NSE_LOOP_INTERVAL_SECONDS", 60)))
                    continue
            except Exception as exc:
                detail = traceback.format_exc()
                logger.error("Unexpected error in NSE loop:\n%s", detail)
                if self.coordinator is not None:
                    self.coordinator.heartbeat(
                        "nse",
                        status="degraded",
                        metrics={"last_error": str(exc), "open_positions": self.portfolio.position_count()},
                    )

            sleep_secs = getattr(config, "NSE_LOOP_INTERVAL_SECONDS", 60)
            stop_event.wait(sleep_secs)

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------
    def _tick(self):
        self.broker.ensure_session()
        requests = {}
        for key, spec in self.strategies.items():
            real_symbol = spec.get("_real_symbol", key.replace("_swing", ""))
            requests[(real_symbol, spec["timeframe"])] = None

        bars_by_request = {}
        # NSE quotes are retrieved from external public data sources. Fetch
        # independent symbol/timeframe requests concurrently to avoid a
        # serial 30-request tick, while bounding pressure on those services.
        with ThreadPoolExecutor(
            max_workers=min(config.NSE_DATA_WORKERS, max(1, len(requests))),
            thread_name_prefix="nse-bars",
        ) as pool:
            futures = {
                pool.submit(
                    self.data_client.get_bars,
                    symbol,
                    timeframe=timeframe,
                    limit=config.BAR_FETCH_LIMIT,
                ): (symbol, timeframe)
                for symbol, timeframe in requests
            }
            for future in as_completed(futures):
                request = futures[future]
                try:
                    bars_by_request[request] = future.result()
                except Exception as exc:
                    logger.warning("NSE data fetch failed for %s %s: %s", *request, exc)
                    bars_by_request[request] = pd.DataFrame()

        failures = []
        data_successes = 0
        for key, spec in self.strategies.items():
            try:
                real_symbol = spec.get("_real_symbol", key.replace("_swing", ""))
                bars = bars_by_request.get((real_symbol, spec["timeframe"]), pd.DataFrame())
                if self._process_symbol(key, real_symbol, spec, bars=bars):
                    data_successes += 1
            except Exception as exc:
                failures.append((key, exc))
                logger.error("Error processing NSE symbol %s:\n%s", key, traceback.format_exc())
        if failures and len(failures) == len(self.strategies):
            symbols = ", ".join(key for key, _ in failures)
            raise RuntimeError(f"All NSE strategy evaluations failed: {symbols}")
        if self.strategies and data_successes == 0:
            raise RuntimeError("No fresh NSE market data was available for any configured symbol.")

    def _process_symbol(
        self,
        key: str,
        symbol: str,
        spec: dict,
        *,
        bars: pd.DataFrame | None = None,
    ):
        timeframe = spec["timeframe"]
        strategy = spec["strategy"]

        if bars is None:
            bars = self.data_client.get_bars(symbol, timeframe=timeframe, limit=config.BAR_FETCH_LIMIT)
        if bars.empty or len(bars) < strategy.min_bars_required:
            return False

        latest_bar = bars.iloc[-1]
        latest_bar_time = bars.index[-1]

        # Dedup — skip if we already processed this bar
        if latest_bar_time == self.last_processed_bar.get(key):
            return

        price = float(latest_bar["close"])
        self.broker.mark_price(symbol, price)
        position = self.portfolio.get(symbol)

        # A cash symbol can have intraday and swing strategy specs. There is
        # still only one position per instrument, so only the strategy that
        # opened it may manage its stop or exit.
        if position is not None and position.strategy != strategy.name:
            self.last_processed_bar[key] = latest_bar_time
            return True

        # --- Exit check ---
        if position is not None:
            # A protective stop must work even when the strategy itself has
            # no exit signal. Use the adverse edge of the candle to avoid
            # optimistic paper fills through a price gap.
            stop_hit = (
                (position.side == "long" and float(latest_bar["low"]) <= position.stop_price)
                or (position.side == "short" and float(latest_bar["high"]) >= position.stop_price)
            )
            if stop_hit:
                stop_fill = (
                    min(float(latest_bar["low"]), position.stop_price)
                    if position.side == "long"
                    else max(float(latest_bar["high"]), position.stop_price)
                )
                self._close_position(symbol, position, stop_fill, "stop_loss", spec)
                self.last_processed_bar[key] = latest_bar_time
                return
            self._update_trailing_stop(symbol, position, latest_bar, spec["trail_atr_multiple"])
            if strategy.evaluate_exit(bars, position.side):
                self._close_position(symbol, position, price, "strategy_exit", spec)
                position = None

        # --- Entry check (only on new bar) ---
        if position is None:
            entry_side = strategy.evaluate_entry(bars)
            if entry_side not in ("long", "short"):
                self.last_processed_bar[key] = latest_bar_time
                return True

            # F&O confirmation for index symbols
            fno_ok = self._check_fno_confirmation(symbol, entry_side)
            if not fno_ok:
                _log_nse_signal(symbol, strategy.name, entry_side,
                                "fno_blocked", price, "F&O signal opposes entry", timeframe)
                self.last_processed_bar[key] = latest_bar_time
                return True

            self._try_open_position(symbol, spec, bars, entry_side, timeframe)

        self.last_processed_bar[key] = latest_bar_time
        return True

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _try_open_position(self, symbol: str, spec: dict, bars: pd.DataFrame, side: str, timeframe: str):
        strategy = spec["strategy"]
        if self.coordinator is not None:
            entries_allowed, reason = self.coordinator.entries_allowed()
            if not entries_allowed:
                _log_nse_signal(symbol, strategy.name, side, "blocked", float(bars.iloc[-1]["close"]), reason, timeframe)
                return
        atr_value = calc_atr(bars, period=config.ATR_PERIOD).iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            return

        price = float(bars.iloc[-1]["close"])
        account = self.broker.get_account_info()
        equity = account["equity"]

        sizing = self.risk_manager.size_position(
            equity=equity,
            atr_value=atr_value,
            price=price,
            side=side,
            fractionable=False,
        )
        if sizing is None:
            return

        # Guard: check max open positions
        if self.portfolio.position_count() >= config.MAX_OPEN_POSITIONS:
            _log_nse_signal(symbol, strategy.name, side, "blocked",
                            price, "Max open positions reached", timeframe)
            return

        risk_state = self.broker.account_risk_state()
        decision = self.risk_manager.evaluate_entry_guardrails(
            symbol=symbol,
            strategy=strategy.name,
            side=side,
            equity=equity,
            candidate_notional=sizing.notional,
            candidate_risk=sizing.risk_amount,
            asset_class="equity",
            portfolio=self.portfolio,
            daily_realized_pnl=float(risk_state["daily_realized_pnl"]),
            day_start_equity=float(risk_state["day_start_equity"]),
            consecutive_losses=int(risk_state["consecutive_losses"]),
            new_entries_allowed=True,
            entry_halt_reason="",
        )
        if not decision.allowed:
            reason = " ".join(decision.reasons)
            _log_nse_signal(symbol, strategy.name, side, "blocked", price, reason, timeframe)
            log_event(
                "nse_entry_blocked",
                reason,
                payload={"symbol": symbol, "metrics": decision.metrics},
                severity="warn",
            )
            return

        # Submit paper order
        order_id = self.broker.submit_market_order(symbol, sizing.qty, side)
        fill_price = self.broker.get_fill_price(order_id, fallback=price, side=side, symbol=symbol)
        stop_price = self.risk_manager.initial_stop_price(fill_price, atr_value, side)
        stop_order_id = self.broker.submit_stop_order(symbol, sizing.qty, side, stop_price)

        position = OpenPosition(
            symbol=symbol,
            side=side,
            qty=sizing.qty,
            entry_price=fill_price,
            entry_time=datetime.now(timezone.utc),
            atr_at_entry=float(atr_value),
            stop_price=stop_price,
            stop_order_id=stop_order_id,
            extreme_price=fill_price,
            strategy=strategy.name,
            risk_amount=sizing.risk_amount,
            notional=float(sizing.qty * fill_price),
            asset_class=spec.get("instrument_type", "equity"),
            entry_signal=f"{strategy.name}:{side}",
            entry_order_id=order_id,
        )
        # Persist the ledger before exposing the position to strategy code.
        # A process crash after this point therefore resumes with an accurate
        # cash balance and protective-stop metadata.
        self.broker.register_position(position)
        self.portfolio.open_position(position)

        _log_nse_signal(symbol, strategy.name, side, "entered",
                        fill_price, "", timeframe)
        msg = (
            f"[NSE {symbol}] Opened {side} {sizing.qty} @ ₹{fill_price:.2f} "
            f"(ATR={atr_value:.2f}, stop=₹{stop_price:.2f}, risk=₹{sizing.risk_amount:.2f})"
        )
        logger.info(msg)
        log_event("nse_position_opened", msg, payload=position.to_dict())

    def _close_position(self, symbol: str, position: OpenPosition, exit_price: float, reason: str, spec: dict):
        settlement = self.broker.close_position(symbol, exit_price)
        if settlement is None:
            raise RuntimeError(f"NSE ledger has no open position for {symbol}; refusing duplicate close.")
        settled_position, pnl = settlement
        self.portfolio.positions.pop(symbol, None)
        self.broker.log_trade(
            symbol=symbol,
            side=settled_position.side,
            qty=settled_position.qty,
            entry_price=settled_position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            strategy=settled_position.strategy,
            reason=reason,
            instrument_type=spec.get("instrument_type", "equity"),
        )

        _log_nse_signal(symbol, settled_position.strategy, settled_position.side,
                        "exited", exit_price, reason, spec.get("timeframe", ""))
        msg = (
            f"[NSE {symbol}] Closed {settled_position.side} @ ₹{exit_price:.2f} "
            f"P&L=₹{pnl:.2f} ({reason})"
        )
        logger.info(msg)
        log_event("nse_position_closed", msg)

    def _update_trailing_stop(self, symbol: str, position: OpenPosition, latest_bar, trail_multiple: float):
        if not trail_multiple:
            return
        extreme = float(latest_bar["high"] if position.side == "long" else latest_bar["low"])
        self.portfolio.update_extreme_price(symbol, extreme)
        new_stop = self.risk_manager.update_trailing_stop(
            current_stop=position.stop_price,
            extreme_price=position.extreme_price,
            atr_value=position.atr_at_entry,
            trail_multiple=trail_multiple,
            side=position.side,
        )
        if new_stop != position.stop_price:
            position.stop_price = new_stop
            self.broker.update_position(position)
            logger.info("[NSE %s] Trailing stop → ₹%.2f", symbol, new_stop)

    # ------------------------------------------------------------------
    # F&O confirmation gate
    # ------------------------------------------------------------------
    def _check_fno_confirmation(self, symbol: str, side: str) -> bool:
        """For F&O symbols, only enter if F&O signal agrees. Equities: always pass."""
        fno_symbols = getattr(config, "NSE_FNO_SYMBOLS", [])
        if symbol not in fno_symbols:
            return True  # no F&O gate for pure equity symbols

        try:
            sig = self.fno_scanner.scan(symbol)
            if sig is None:
                return True  # can't fetch → don't block
            if side == "long" and sig.direction == "bearish":
                return False
            if side == "short" and sig.direction == "bullish":
                return False
        except Exception as exc:
            logger.debug("F&O scan failed for %s: %s", symbol, exc)
        return True
