"""Main orchestration loop.

Alpaca v3 adds (on top of v2):
- Market Analyst Bot integration: regime detection, sentiment analysis,
  macro monitoring, and multi-factor signal scoring
- Watchdog for autonomous 24/7 operation: API health tracking, graduated
  response levels, automatic exposure reduction
- Regime-aware strategy gating: disable mismatched strategies
- Confidence-based and regime-scaled position sizing
- Adaptive loop cadence tied to volatility regime

Previous v2 features:
- persistent runtime state and restart-safe resume
- portfolio-level guardrails
- structured signal / event / guardrail artifacts
- periodic broker reconciliation for externally closed positions
- market filters between raw strategy signals and order submission
- adaptive loop cadence when positions are open or the broker is flaky
"""

from __future__ import annotations
import os
import math
import pandas as pd

import sys
import time
import traceback
import threading
from collections import Counter
from datetime import datetime, timezone


import config
from bot import logger as trade_logger
from bot.artifacts import log_event, log_guardrail, log_signal, generate_session_performance_report
from bot.brokers import create_broker
from bot.stress_test import StressTestEngine
from bot.garch_model import VolatilityForecaster
from bot.indicators import atr as calc_atr
from bot.market_filters import MarketFilters
from bot.portfolio import OpenPosition, Portfolio
from bot.risk_manager import RiskManager
from bot.tradingview.signal_bridge import drain_signals
from bot.runtime_state import RuntimeState
from bot.runtime_coordinator import RuntimeCoordinator
from bot.state_store import RuntimeStateStore
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.strategies.red_green_breakout import RedGreenBreakoutStrategy
from bot.watchdog import Watchdog
from bot.optimizer import StrategyOptimizer
from bot.guardian import Guardian
from bot.drift_monitor import DriftMonitor
from bot.agents.post_mortem import PostMortemAnalyzer
from bot.shadow_trader import ShadowTrader

try:
    from bot.analyst import MarketAnalystBot, MarketSignal
    ANALYST_AVAILABLE = True
except ImportError:
    ANALYST_AVAILABLE = False
    MarketAnalystBot = None  # type: ignore[assignment,misc]
    MarketSignal = None  # type: ignore[assignment,misc]

from alpaca.common.exceptions import APIError


def build_strategies() -> dict:
    """Real broker symbol -> strategy specification.

    A strategy key is always a tradable symbol. Keeping internal labels out
    of this registry prevents accidental requests/orders for fake symbols
    such as ``SPY_rg``.
    """
    strategies = {}

    for symbol in config.MEAN_REVERSION_SYMBOLS:
        params = config.MEAN_REVERSION_PARAMS.get(symbol, {"std_multiplier": 1.5})
        strategies[symbol] = {
            "symbol": symbol,
            "strategy": MeanReversionStrategy(
                symbol=symbol,
                std_multiplier=params["std_multiplier"],
                lookback=config.MEAN_REVERSION_LOOKBACK,
            ),
            "timeframe_amount": config.MEAN_REVERSION_TIMEFRAME_AMOUNT,
            "timeframe_unit": config.MEAN_REVERSION_TIMEFRAME_UNIT,
            "trail_atr_multiple": 1.5,  # CHANGED from None: add trailing stop to lock in profits and protect against trend continuation
        }

    for symbol in config.MOMENTUM_SYMBOLS:
        strategies[symbol] = {
            "symbol": symbol,
            "strategy": MomentumBreakoutStrategy(
                symbol=symbol,
                lookback=config.MOMENTUM_LOOKBACK,
                volume_multiplier=config.MOMENTUM_VOLUME_MULTIPLIER,
            ),
            "timeframe_amount": config.MOMENTUM_TIMEFRAME_AMOUNT,
            "timeframe_unit": config.MOMENTUM_TIMEFRAME_UNIT,
            "trail_atr_multiple": config.MOMENTUM_TRAIL_ATR_MULTIPLIER,
        }

    for symbol in config.TREND_SYMBOLS:
        strategies[symbol] = {
            "symbol": symbol,
            "strategy": TrendFollowingStrategy(
                symbol=symbol,
                fast_period=config.TREND_FAST_EMA,
                slow_period=config.TREND_SLOW_EMA,
                allow_short=config.TREND_ALLOW_SHORT,
            ),
            "timeframe_amount": config.TREND_TIMEFRAME_AMOUNT,
            "timeframe_unit": config.TREND_TIMEFRAME_UNIT,
            "trail_atr_multiple": config.TREND_TRAIL_ATR_MULTIPLIER,
        }

    if config.RED_GREEN_BREAKOUT_ENABLED:
        if not config.RED_GREEN_BREAKOUT_SYMBOLS:
            print(
                "[RedGreen] Enabled but no symbols are assigned; "
                "the strategy will remain idle."
            )
        duplicate_symbols = set(config.RED_GREEN_BREAKOUT_SYMBOLS) & set(strategies)
        if duplicate_symbols:
            names = ", ".join(sorted(duplicate_symbols))
            raise ValueError(
                "RED_GREEN_BREAKOUT_SYMBOLS overlaps the primary strategy universe "
                f"({names}). Assign only one strategy to an underlying symbol."
            )
        for symbol in config.RED_GREEN_BREAKOUT_SYMBOLS:
            strategies[symbol] = {
                "symbol": symbol,
                "strategy": RedGreenBreakoutStrategy(symbol=symbol),
                "timeframe_amount": 5,
                "timeframe_unit": "Minute",
                "trail_atr_multiple": 2.0,
            }
    
    return strategies


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unauthorized" in message or "authorization required" in message or "401" in message


def _alpaca_auth_help() -> str:
    expected_mode = "PAPER" if config.ALPACA_PAPER else "LIVE"
    return (
        "Alpaca rejected the current API credentials.\n"
        f"Expected account mode: {expected_mode}\n"
        f"Update {config.ROOT_DIR}/.env with a matching API key / secret pair and "
        f"confirm ALPACA_PAPER is set to {'true' if config.ALPACA_PAPER else 'false'}.\n"
        "Then restart with:\n"
        f"  cd {config.ROOT_DIR}\n"
        f"  {config.BOT_LAUNCH_COMMAND}"
    )


def _run_tradingview_webhook() -> None:
    """Serve TradingView alerts without blocking the trading worker."""
    try:
        from bot.tradingview.webhook_server import start_webhook_server

        start_webhook_server()
    except Exception as exc:
        print(f"[TradingView] Webhook server failed to start: {exc}")
        log_event("tradingview_webhook_start_failed", str(exc), severity="error")


def _timestamp_or_none(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value)


def _timeframe_label(spec: dict) -> str:
    return f"{spec['timeframe_amount']} {spec['timeframe_unit']}"


class TradingBot:
    def __init__(self, coordinator: RuntimeCoordinator | None = None):
        from bot.agents.desk_manager import DeskManager
        from bot.agents.news_terminal import NewsTerminal
        from bot.agents.committee import InvestmentCommittee
        from bot.agents.virtual_brain import VirtualHumanBrain
        config.ensure_runtime_directories()
        self.coordinator = coordinator
        # --- ML Signal Filter (MUST BE INITIALIZED BEFORE THREADS ON MAC) ---
        self._ml_filter = None
        if getattr(config, "ML_FILTER_ENABLED", False):
            try:
                from bot.ml_filter import MLSignalFilter
                self._ml_filter = MLSignalFilter()
                print("[ML] XGBoost Signal Filter initialised.")
            except Exception as exc:
                print(f"[ML] Failed to initialise ML Filter: {exc}")

        self.broker = create_broker()
        self.portfolio = Portfolio(self.broker)
        self.risk_manager = RiskManager()
        self._stress_tester = StressTestEngine(self.portfolio)
        self._garch_forecaster = VolatilityForecaster()
        self.market_filters = MarketFilters()
        self.guardian = Guardian()
        self._drift_monitor = DriftMonitor()
        self.strategies = build_strategies()
        self._recent_closes = {}
        self._market_data_snapshot: dict = {
            "source": "alpaca_bars_rest",
            "feed": config.ALPACA_DATA_FEED.upper(),
            "status": "waiting",
            "symbols": {},
        }
        
        self.state_store = RuntimeStateStore()
        self.desk = DeskManager()
        self._post_mortem = PostMortemAnalyzer()
        self._shadow_trader = ShadowTrader()
        self._shadow_trader.load_results()
        self.committee = None
        self.virtual_brain = None
        if config.AI_COMMITTEE_ENABLED:
            try:
                self.committee = InvestmentCommittee()
                print("[AI] Investment Committee initialized (local Ollama).")
            except Exception as exc:
                log_event("committee_init_failed", str(exc), severity="warn")
        if config.AI_VIRTUAL_BRAIN_ENABLED:
            try:
                self.virtual_brain = VirtualHumanBrain()
                print("[AI] Virtual Human Brain initialized (local Ollama).")
            except Exception as exc:
                log_event("virtual_brain_init_failed", str(exc), severity="warn")
        try:
            self.news_terminal = NewsTerminal(self.desk)
            self.news_terminal.start()
        except Exception:
            pass
        now = datetime.now(timezone.utc)
        


        # --- Alpaca WebSocket Streamer ---
        self._streamer = None
        try:
            from bot.market_streamer import MarketStreamer
            stream_symbols = sorted({
                spec["symbol"]
                for spec in self.strategies.values()
                if "/" not in spec["symbol"]
            })
            if stream_symbols:
                self._streamer = MarketStreamer(stream_symbols)
                self._streamer.start()
                self.broker.set_streamer(self._streamer)
                print(f"[WebSockets] Alpaca Live Streamer initialized for {len(stream_symbols)} symbols.")
        except Exception as e:
            print(f"[WebSockets] Failed to init streamer: {e}")

        # --- Market Analyst Bot ---
        self._analyst = None
        self._analyst_signals: dict[str, MarketSignal] = {}  # type: ignore[type-arg]
        self._last_analyst_run: datetime | None = None
        self._last_analyst_success: datetime | None = None
        self._analyst_lock = threading.Lock()
        self._analyst_running = False
        self._analyst_started_at: datetime | None = None
        self._analyst_timed_out = False
        self._analyst_generation = 0
        self._pending_analyst_signals = None
        self._pending_analyst_error = None
        if config.ANALYST_ENABLED and ANALYST_AVAILABLE:
            try:
                self._analyst = MarketAnalystBot(broker=self.broker)
                print("[Analyst] Market Analyst Bot initialised.")
                log_event("analyst_init", "Market Analyst Bot initialised successfully.")
            except Exception as exc:
                print(f"[Analyst] Failed to initialise: {exc} — running without analyst.")
                log_event("analyst_init_failed", str(exc), severity="warn")
        elif config.ANALYST_ENABLED and not ANALYST_AVAILABLE:
            print("[Analyst] Analyst dependencies not installed — running without analyst.")

        # --- Watchdog ---
        self._watchdog = Watchdog() if config.WATCHDOG_ENABLED else None

        # --- Bayesian strategy optimizer ---
        self._optimizer = None
        if getattr(config, "OPTIMIZER_ENABLED", False):
            self._optimizer = StrategyOptimizer(
                refit_interval_days=getattr(config, "OPTIMIZER_REFIT_INTERVAL_DAYS", 7)
            )

        # The monolithic supervisor consumes TradingView signals locally, so
        # start the matching HTTP receiver in a daemon thread when enabled.
        self._tradingview_thread = None
        if getattr(config, "TRADINGVIEW_ENABLED", False):
            self._tradingview_thread = threading.Thread(
                target=_run_tradingview_webhook,
                name="tradingview-webhook",
                daemon=True,
            )
            self._tradingview_thread.start()
            print(
                "[TradingView] Webhook receiver starting on "
                f"port {config.TRADINGVIEW_WEBHOOK_PORT}."
            )

        # --- US Options (F&O) Scanner ---
        self._us_options_scanner = None
        if getattr(config, "US_OPTIONS_ENABLED", False):
            try:
                from bot.us_options.options_scanner import USOptionsScanner
                self._us_options_scanner = USOptionsScanner(
                    cache_ttl_seconds=getattr(config, "US_OPTIONS_CACHE_TTL_SECONDS", 300)
                )
                print("[Options] US Options Scanner initialised (PCR + max-pain gate active).")
                log_event("us_options_init", "US Options Scanner initialised.")
            except Exception as exc:
                print(f"[Options] Failed to initialise US Options Scanner: {exc}")
                log_event("us_options_init_failed", str(exc), severity="warn")

        # --- Telegram Notifier ---
        from bot.notifications import TelegramNotifier
        self._notifier = TelegramNotifier(command_handler=self.handle_telegram_command)

        # --- Earnings Calendar Filter ---
        self._earnings_calendar = None
        if getattr(config, "EARNINGS_ENABLED", True):
            try:
                from bot.earnings import EarningsCalendar
                self._earnings_calendar = EarningsCalendar()
                # Prefetch all equity symbols in the background
                all_symbols = sorted({spec["symbol"] for spec in self.strategies.values()})
                self._earnings_calendar.prefetch(all_symbols)
                print(f"[Earnings] Earnings Filter initialised (gating {len(all_symbols)} symbols).")
                log_event("earnings_init", "Earnings Calendar initialised.")
            except Exception as exc:
                print(f"[Earnings] Failed to initialise Earnings Calendar: {exc}")
                log_event("earnings_init_failed", str(exc), severity="warn")
            


        
        self.state = self.state_store.load() or RuntimeState(
            mode="PAPER" if config.ALPACA_PAPER else "LIVE",
            started_at=now,
            runtime_status="starting",
            last_processed_bar={symbol: None for symbol in self.strategies},
        )
        self.state.started_at = self.state.started_at or now
        self.portfolio.load_positions(self.state.open_positions)

        self.last_processed_bar = {
            symbol: _timestamp_or_none(self.state.last_processed_bar.get(symbol))
            for symbol in self.strategies
        }
        for symbol in self.strategies:
            self.state.last_processed_bar.setdefault(symbol, None)
            self.last_processed_bar.setdefault(symbol, None)

        self.day_start_equity = self.state.day_start_equity
        self.current_date = self.state.current_date or None
        self._auth_failure_reported = False
        self._recent_closes = {}
        self._manual_entry_pause = self.state.entry_halt_reason == "Manually paused via Telegram."
        

    def handle_telegram_command(self, text: str) -> str:
        text = text.lower().strip()
        
        if text == "/status":
            try:
                equity = self.portfolio.get_equity()
            except:
                equity = self.state.last_equity or 0.0
            positions = len(self.portfolio.positions)
            mode = "PAPER" if config.ALPACA_PAPER else "LIVE"
            pnl = self.state.daily_realized_pnl
            return (
                f"🤖 *Bot Status ({mode})*\n"
                f"Equity: ${equity:,.2f}\n"
                f"Daily P&L: ${pnl:,.2f}\n"
                f"Open Positions: {positions}\n"
                f"Entries Allowed: {'Yes' if self.state.new_entries_allowed else 'No'}"
            )
            
        elif text == "/pause":
            self._manual_entry_pause = True
            self.state.new_entries_allowed = False
            self.state.entry_halt_reason = "Manually paused via Telegram."
            if self.coordinator is not None:
                self.coordinator.set_operator_pause(True, self.state.entry_halt_reason)
            self._persist_state()
            return "⏸️ Bot paused. No new entries will be taken."
            
        elif text == "/resume":
            self._manual_entry_pause = False
            if self.coordinator is not None:
                self.coordinator.set_operator_pause(False)
                entries_allowed, pause_reason = self.coordinator.entries_allowed()
                self.state.new_entries_allowed = entries_allowed
                self.state.entry_halt_reason = pause_reason
            else:
                self.state.new_entries_allowed = True
                self.state.entry_halt_reason = ""
            self._persist_state()
            if self.state.new_entries_allowed:
                return "▶️ Bot resumed. Entries are now allowed."
            return f"⏸️ Operator pause cleared, but entries remain halted: {self.state.entry_halt_reason}"
            
        elif text == "/positions":
            if not self.portfolio.positions:
                return "📂 No open positions."
            lines = ["📊 *Open Positions*"]
            total_unrealized = 0.0
            for sym, pos in self.portfolio.positions.items():
                try:
                    curr_price = self.broker.get_latest_price(sym)
                    if curr_price is None:
                        raise RuntimeError("no current price")
                    curr_price = float(curr_price)
                    unrealized = (curr_price - pos.entry_price) * pos.qty * (1 if pos.side == "long" else -1)
                    total_unrealized += unrealized
                    sign = "+" if unrealized >= 0 else ""
                    lines.append(f"• `{sym}` ({pos.side}): {sign}${unrealized:,.2f}")
                except:
                    lines.append(f"• `{sym}` ({pos.side})")
            lines.append(f"\nTotal Unrealized: *${total_unrealized:,.2f}*")
            return "\n".join(lines)
            
        elif text == "/logs":
            # Load last 5 closed trades
            trades = []
            if os.path.exists(config.TRADES_LOG_PATH):
                import csv
                with open(config.TRADES_LOG_PATH, "r") as f:
                    reader = list(csv.DictReader(f))
                    trades = reader[-5:]
            if not trades:
                return "📭 No recent trades found."
            lines = ["📝 *Recent Trades (Last 5)*"]
            for t in reversed(trades):
                sym = t.get("instrument", "?")
                pnl = float(t.get("profit_loss", 0))
                sign = "🟢 +" if pnl > 0 else "🔴 "
                lines.append(f"{sign}${pnl:,.2f} on `{sym}`")
            return "\n".join(lines)
            
        elif text.startswith("/close"):
            parts = text.split()
            if len(parts) < 2:
                return "❌ Usage: /close <SYMBOL>"
            symbol = parts[1].upper()
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                try:
                    exit_price = self.broker.get_latest_price(symbol)
                    if exit_price is None:
                        raise RuntimeError("no current price")
                    exit_price = float(exit_price)
                except:
                    exit_price = pos.entry_price
                self._close_position(symbol, pos, exit_price, reason="manual_telegram_close")
                return f"✅ Closing {symbol}..."
            else:
                return f"⚠️ {symbol} is not currently open."
                
        elif text == "/help":
            return (
                "🛠️ *Available Commands*\n"
                "/status - View live P&L and equity\n"
                "/pause - Stop taking new trades\n"
                "/resume - Allow new trades\n"
                "/close SYMBOL - Manually close a position\n"
            )
            
        return "❓ Unknown command. Type /help for options."

    def run_forever(self, stop_event: threading.Event | None = None):
        """Run until the shared supervisor asks this worker to stop."""
        stop_event = stop_event or threading.Event()
        mode = "PAPER" if config.ALPACA_PAPER else "LIVE"
        analyst_tag = " + Market Analyst" if self._analyst else ""
        watchdog_tag = " + Watchdog" if self._watchdog else ""
        print(f"Starting multi-strategy Alpaca bot v3 ({mode} trading{analyst_tag}{watchdog_tag})...")
        log_event("startup", f"Launching Alpaca bot v3 in {mode} mode.{analyst_tag}{watchdog_tag}")
        self._startup_sync()

        # Telegram startup alert
        try:
            equity = self.portfolio.get_equity()
            self._notifier.send_startup(
                mode=mode,
                equity=equity,
                nse_enabled=getattr(config, "NSE_ENABLED", False),
            )
        except Exception:
            pass

        while not stop_event.is_set():
            cb = self.broker.circuit_breaker

            if not cb.allow_request():
                # Circuit breaker is OPEN — skip trading until cooldown elapses.
                if self.state.runtime_status != "degraded":
                    self._mark_runtime_degraded(
                        f"Circuit breaker OPEN after {cb.consecutive_failures} consecutive failures. "
                        f"Cooling down for {cb.cooldown_seconds}s."
                    )
                stop_event.wait(self._sleep_interval())
                continue

            try:
                self._tick()
                cb.record_success()
                if self._watchdog:
                    self._watchdog.record_api_success()
            except APIError as exc:
                cb.record_failure()
                if self._watchdog:
                    self._watchdog.record_api_error(str(exc))
                if _is_auth_error(exc):
                    self._handle_auth_failure("broker sync failed during the main loop", exc)
                else:
                    self._mark_runtime_degraded(f"APIError in main loop: {exc}", auth_error=False)
                    print(f"[APIError] {exc}")
            except Exception:
                cb.record_failure()
                if self._watchdog:
                    self._watchdog.record_api_error("unexpected_exception")
                detail = traceback.format_exc()
                self._mark_runtime_degraded("Unexpected error in main loop.", detail)
                print("[Unexpected error in main loop]")
                traceback.print_exc()

            # Watchdog periodic health check and alerting.
            if self._watchdog:
                self._watchdog.check_and_alert()

            stop_event.wait(self._sleep_interval())

        self.state.runtime_status = "stopped"
        self.state.broker_connected = False
        self.state.note("US trading worker stopped by runtime supervisor.")
        self._persist_state()

    def _sleep_interval(self) -> int:
        if self.state.runtime_status == "degraded":
            return config.ERROR_RETRY_INTERVAL_SECONDS
        if self.portfolio.positions:
            base = config.OPEN_POSITION_LOOP_INTERVAL_SECONDS
        else:
            base = config.LOOP_INTERVAL_SECONDS

        # Regime-aware: speed up in HIGH_VOLATILITY, slow down in quiet markets.
        if getattr(config, "ADAPTIVE_LOOP_CADENCE", True):
            dominant_regime = self._get_dominant_regime()
            if dominant_regime == "HIGH_VOLATILITY":
                base = max(5, base // 2)  # twice as fast, min 5s
            elif dominant_regime == "MEAN_REVERTING":
                base = int(base * 1.5)  # 50% slower in quiet markets
        return base

    def _get_dominant_regime(self) -> str:
        """Most common analyst regime across symbols, or empty string."""
        if not self._analyst_signals:
            return ""
        regimes = [s.regime for s in self._analyst_signals.values() if getattr(s, "regime", None)]
        if not regimes:
            return ""
        return Counter(regimes).most_common(1)[0][0]

    def _startup_sync(self):
        for attempt in range(5):
            try:
                self._reconcile_portfolio_with_broker(startup=True)
                info = self.broker.get_account_info()
                print(
                    f"Connected. Equity=${info['equity']:.2f} "
                    f"buying_power=${info['buying_power']:.2f}"
                )
                if info["pattern_day_trader"] and info["equity"] < 25000:
                    print(
                        "NOTE: this account is flagged as a pattern day "
                        "trader with equity under $25,000. Alpaca will "
                        "reject equity day trades beyond the PDT limit."
                    )
                self.state.runtime_status = "running"
                self.state.broker_connected = True
                self.state.auth_error = False
                self.state.last_broker_error = ""
                self._auth_failure_reported = False
                self._persist_state()
                return
            except APIError as exc:
                if _is_auth_error(exc):
                    raise RuntimeError(_alpaca_auth_help()) from exc
                wait = 2 ** attempt
                print(f"Startup sync failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)
            except Exception as exc:
                if _is_auth_error(exc):
                    raise RuntimeError(_alpaca_auth_help()) from exc
                wait = 2 ** attempt
                print(f"Startup sync failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)

        print(
            "Could not reach Alpaca after several attempts at startup; "
            "will keep retrying inside the main loop."
        )
        self._mark_runtime_degraded("Broker startup sync failed after retries.")

    def _mark_runtime_degraded(self, message: str, detail: str | None = None, auth_error: bool = False):
        if self.state.runtime_status != "degraded":
            self.state.degraded_since = datetime.now(timezone.utc)
        self.state.runtime_status = "degraded"
        self.state.broker_connected = False
        self.state.auth_error = auth_error
        self.state.last_broker_error = detail or message
        self.state.note(message)
        log_event("runtime_degraded", message, payload={"detail": detail or "", "auth_error": auth_error}, severity="warn")
        self._persist_state()

    def _handle_auth_failure(self, context: str, exc: Exception):
        detail = str(exc)
        if not self.state.auth_error:
            self._mark_runtime_degraded(
                f"Alpaca authentication failed while {context}.",
                detail=detail,
                auth_error=True,
            )
        else:
            self.state.runtime_status = "degraded"
            self.state.broker_connected = False
            self.state.auth_error = True
            self.state.last_broker_error = detail
            self._persist_state()

        if not self._auth_failure_reported:
            heading = context[:1].upper() + context[1:]
            print(f"\n[AUTH ERROR] {heading}.\n{_alpaca_auth_help()}\n")
            self._auth_failure_reported = True

    def _tick(self):
        if self._optimizer:
            try:
                updated_symbols = self._optimizer.apply_pending_updates(self.strategies)
                if updated_symbols:
                    log_event(
                        "optimizer_params_applied",
                        "Applied completed optimizer results between trading ticks.",
                        payload={"symbols": updated_symbols},
                    )
            except Exception as exc:
                log_event("optimizer_apply_error", str(exc), severity="warn")

        self._reconcile_portfolio_with_broker(startup=False)
        self._roll_day_if_needed()

        try:
            equity_now = self.portfolio.get_equity()
        except Exception as exc:
            self._mark_runtime_degraded("Could not fetch account equity.", str(exc))
            raise RuntimeError("Could not fetch account equity; skipping trading tick.") from exc

        self.state.last_equity = float(equity_now)
        self._refresh_entry_gate(equity_now)
        if self._watchdog:
            self._watchdog.record_pnl_snapshot(float(equity_now))

        bars_by_symbol = self._collect_strategy_bars()

        # --- Run Market Analyst Bot (throttled) ---
        analyst_bars = bars_by_symbol
        if getattr(self, "_market_data_snapshot", {}).get("status") == "market_closed" and self._analyst_due():
            try:
                analyst_bars = self._collect_analyst_bars()
            except Exception as exc:
                analyst_bars = {}
                log_event(
                    "analyst_market_data_error",
                    "Could not refresh closed-session bars for market analysis.",
                    payload={"error_type": type(exc).__name__},
                    severity="warn",
                )
        self._maybe_run_analyst(analyst_bars)

        # --- Bayesian parameter retune (throttled to every N days) ---
        if self._optimizer:
            try:
                self._optimizer.maybe_optimize(self.strategies, self.broker)
            except Exception as exc:
                print(f"[Optimizer] Error during maybe_optimize: {exc}")
                log_event("optimizer_error", str(exc), severity="warn")

        # --- Watchdog: block new entries if health is critical ---
        if self._watchdog and self._watchdog.should_block_new_entries:
            if self.state.new_entries_allowed:
                reason = (
                    f"Watchdog blocking new entries — health score "
                    f"{self._watchdog.health_score:.0f} ({self._watchdog.health_status})."
                )
                self.state.new_entries_allowed = False
                self.state.entry_halt_reason = reason
                log_event("watchdog_block", reason, severity="warn")

        symbol_failures = []
        for symbol, spec in self.strategies.items():
            try:
                bars = bars_by_symbol.get(symbol)
                if bars is not None:
                    self._process_symbol(symbol, spec, bars=bars)
            except APIError as exc:
                if _is_auth_error(exc):
                    self._handle_auth_failure(f"loading market data for {symbol}", exc)
                    raise

                print(f"[{symbol}] APIError: {exc}")
                symbol_failures.append((symbol, exc))
                if self._watchdog:
                    self._watchdog.record_api_error(str(exc))
                log_event("symbol_api_error", f"{symbol} raised an API error.", payload={"error": str(exc), "symbol": symbol}, severity="warn")
            except Exception:
                print(f"[{symbol}] Unexpected error:")
                traceback.print_exc()
                symbol_failures.append((symbol, sys.exc_info()[1]))
                log_event(
                    "symbol_error",
                    f"{symbol} raised an unexpected error.",
                    payload={"symbol": symbol, "traceback": traceback.format_exc()},
                    severity="warn",
                )

        if symbol_failures and len(symbol_failures) == len(bars_by_symbol):
            failed_symbols = ", ".join(symbol for symbol, _ in symbol_failures)
            raise RuntimeError(f"All strategy evaluations failed this tick: {failed_symbols}")

        if getattr(config, "TRADINGVIEW_ENABLED", False):
            self._process_tradingview_signals()

        self._log_running_daily_pnl()

        # Transition from degraded → running only after a complete healthy tick.
        was_degraded = self.state.runtime_status == "degraded"
        self.state.runtime_status = "running"
        self.state.broker_connected = True
        self.state.auth_error = False
        self.state.last_broker_error = ""
        self._auth_failure_reported = False
        self._recent_closes = {}
        

        if was_degraded:
            self.state.recovery_count += 1
            duration = ""
            if self.state.degraded_since:
                elapsed = (datetime.now(timezone.utc) - self.state.degraded_since).total_seconds()
                duration = f" after {elapsed:.0f}s"
            self.state.degraded_since = None
            self.state.note(f"Recovered from degraded mode{duration}.")
            log_event(
                "runtime_recovered",
                f"Broker connection recovered{duration}; resuming normal operation.",
                payload={"recovery_count": self.state.recovery_count},
                severity="info",
            )

        self._persist_state()

    # ------------------------------------------------------------------
    # Market Analyst Bot
    # ------------------------------------------------------------------
    def _collect_strategy_bars(self) -> dict[str, pd.DataFrame]:
        """Fetch each live symbol/timeframe once, using the broker batch API."""
        previous_symbols = getattr(self, "_market_data_snapshot", {}).get("symbols", {})
        symbol_status = {}
        open_symbols = set()
        for symbol, spec in self.strategies.items():
            is_open = self.broker.is_market_open(symbol)
            if is_open:
                open_symbols.add(symbol)
                status = "pending"
            else:
                status = "market_closed"
            prior = previous_symbols.get(symbol, {})
            symbol_status[symbol] = {
                "strategy": getattr(spec.get("strategy"), "name", ""),
                "timeframe": _timeframe_label(spec),
                "status": status,
                "bar_count": int(prior.get("bar_count", 0)),
                "latest_bar_at": prior.get("latest_bar_at"),
                "fetched_at": prior.get("fetched_at"),
            }
        grouped: dict[tuple[int, str], list[str]] = {}
        for symbol, spec in self.strategies.items():
            if symbol in open_symbols:
                grouped.setdefault(
                    (spec["timeframe_amount"], spec["timeframe_unit"]), []
                ).append(symbol)

        bars_by_symbol: dict[str, pd.DataFrame] = {}
        for (amount, unit), symbols in grouped.items():
            try:
                frames = self.broker.get_bars_batch(
                    symbols, amount, unit, limit=config.BAR_FETCH_LIMIT
                )
            except Exception as exc:
                for symbol in symbols:
                    symbol_status[symbol].update(
                        status="error", error_type=type(exc).__name__,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                    )
                self._market_data_snapshot = {
                    "source": "alpaca_bars_rest",
                    "feed": config.ALPACA_DATA_FEED.upper(),
                    "status": "error",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "symbols": symbol_status,
                    "stream": self._stream_status(),
                }
                raise
            bars_by_symbol.update(frames)
            fetched_at = datetime.now(timezone.utc).isoformat()
            for symbol in symbols:
                frame = frames.get(symbol)
                if frame is None or frame.empty:
                    symbol_status[symbol].update(status="missing", fetched_at=fetched_at)
                    continue
                latest_bar_at = pd.Timestamp(frame.index[-1])
                if latest_bar_at.tzinfo is None:
                    latest_bar_at = latest_bar_at.tz_localize(timezone.utc)
                else:
                    latest_bar_at = latest_bar_at.tz_convert(timezone.utc)
                observed_at = datetime.now(timezone.utc)
                age_seconds = max(0.0, (observed_at - latest_bar_at.to_pydatetime()).total_seconds())
                unit_seconds = {"Minute": 60, "Hour": 3600, "Day": 86400}.get(unit, 60)
                stale_after_seconds = amount * unit_seconds * 2 + 120
                bar_status = "stale" if age_seconds > stale_after_seconds else "available"
                symbol_status[symbol].update(
                    status=bar_status,
                    bar_count=int(len(frame)),
                    latest_bar_at=latest_bar_at.isoformat(),
                    fetched_at=fetched_at,
                    age_seconds=round(age_seconds, 1),
                    stale_after_seconds=stale_after_seconds,
                )

        statuses = {row["status"] for row in symbol_status.values()}
        if not symbol_status:
            overall_status = "not_configured"
        elif "error" in statuses:
            overall_status = "error"
        elif "missing" in statuses:
            overall_status = "partial"
        elif "stale" in statuses:
            overall_status = "partial"
        elif "available" in statuses:
            overall_status = "available"
        else:
            overall_status = "market_closed"
        self._market_data_snapshot = {
            "source": "alpaca_bars_rest",
            "feed": config.ALPACA_DATA_FEED.upper(),
            "status": overall_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "configured_symbols": len(symbol_status),
            "available_symbols": sum(row["status"] == "available" for row in symbol_status.values()),
            "stale_symbols": sum(row["status"] == "stale" for row in symbol_status.values()),
            "symbols": symbol_status,
            "stream": self._stream_status(),
        }
        return bars_by_symbol

    def _stream_status(self) -> dict:
        streamer = getattr(self, "_streamer", None)
        if streamer is None:
            return {"feed": config.ALPACA_DATA_FEED.upper(), "status": "unavailable", "symbols": {}}
        try:
            return streamer.get_status()
        except Exception:
            return {"feed": config.ALPACA_DATA_FEED.upper(), "status": "unknown", "symbols": {}}

    def _analyst_due(self) -> bool:
        """Whether the Analyst needs a fresh dataset without delaying trades."""
        if not getattr(self, "_analyst", None):
            return False
        with self._analyst_lock:
            if self._analyst_running:
                return False
        last_run = self._last_analyst_run
        return (
            last_run is None
            or (datetime.now(timezone.utc) - last_run).total_seconds()
            >= config.ANALYST_INTERVAL_SECONDS
        )

    def _collect_analyst_bars(self) -> dict[str, pd.DataFrame]:
        """Fetch the latest Alpaca history for analysis only when the US session is closed."""
        symbol_status = self._market_data_snapshot.get("symbols", {})
        frames_by_symbol: dict[str, pd.DataFrame] = {}
        groups: dict[tuple[int, str], list[str]] = {}
        for symbol, spec in self.strategies.items():
            groups.setdefault(
                (spec["timeframe_amount"], spec["timeframe_unit"]), []
            ).append(symbol)

        for (amount, unit), symbols in groups.items():
            try:
                frames = self.broker.get_bars_batch(
                    symbols, amount, unit, limit=config.BAR_FETCH_LIMIT
                )
            except Exception:
                for symbol in symbols:
                    row = symbol_status.setdefault(symbol, {})
                    row.update(status="error", analysis_only=True, market_open=False)
                self._market_data_snapshot.update(
                    status="error",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    symbols=symbol_status,
                    stream=self._stream_status(),
                )
                raise

            frames_by_symbol.update(frames)
            fetched_at = datetime.now(timezone.utc).isoformat()
            for symbol in symbols:
                frame = frames.get(symbol)
                row = symbol_status.setdefault(symbol, {})
                if frame is None or frame.empty:
                    row.update(status="missing", analysis_only=True, market_open=False, fetched_at=fetched_at)
                    continue
                latest_bar_at = pd.Timestamp(frame.index[-1])
                if latest_bar_at.tzinfo is None:
                    latest_bar_at = latest_bar_at.tz_localize(timezone.utc)
                else:
                    latest_bar_at = latest_bar_at.tz_convert(timezone.utc)
                row.update(
                    status="analysis_only",
                    analysis_only=True,
                    market_open=False,
                    bar_count=int(len(frame)),
                    latest_bar_at=latest_bar_at.isoformat(),
                    fetched_at=fetched_at,
                    age_seconds=round(max(0.0, (datetime.now(timezone.utc) - latest_bar_at.to_pydatetime()).total_seconds()), 1),
                )

        self._market_data_snapshot.update(
            source="alpaca_bars_rest",
            feed=config.ALPACA_DATA_FEED.upper(),
            status="market_closed",
            analysis_only=True,
            updated_at=datetime.now(timezone.utc).isoformat(),
            configured_symbols=len(self.strategies),
            available_symbols=sum(row.get("status") == "analysis_only" for row in symbol_status.values()),
            stale_symbols=0,
            symbols=symbol_status,
            stream=self._stream_status(),
        )
        return frames_by_symbol

    def _maybe_run_analyst(self, bars_by_symbol: dict[str, pd.DataFrame]):
        """Schedule analyst I/O without ever blocking strategy execution."""
        if not self._analyst:
            return

        now = datetime.now(timezone.utc)
        with self._analyst_lock:
            completed_signals = self._pending_analyst_signals
            completed_error = self._pending_analyst_error
            self._pending_analyst_signals = None
            self._pending_analyst_error = None

        if completed_error is not None:
            print(f"[Analyst] Error running analysis: {completed_error}")
            log_event("analyst_error", str(completed_error), severity="warn")
        elif completed_signals is not None:
            self._analyst_signals = completed_signals
            self._last_analyst_success = now
            for sym, sig in self._analyst_signals.items():
                print(
                    f"[Analyst] {sym}: {sig.direction} "
                    f"(confidence={sig.confidence}, regime={sig.regime})"
                )
            log_event(
                "analyst_run",
                f"Analyst produced signals for {len(self._analyst_signals)} symbols.",
                payload={
                    sym: {"direction": s.direction, "confidence": s.confidence, "regime": s.regime}
                    for sym, s in self._analyst_signals.items()
                },
            )

        if self._analyst_running:
            if (
                self._analyst_started_at is not None
                and not self._analyst_timed_out
                and (now - self._analyst_started_at).total_seconds() > config.ANALYST_MAX_RUNTIME_SECONDS
            ):
                self._analyst_timed_out = True
                # A Python thread cannot be forcibly cancelled safely. Mark
                # this generation stale so a hung network request cannot
                # freeze analysis forever or publish stale output later.
                with self._analyst_lock:
                    self._analyst_generation += 1
                    self._analyst_running = False
                self._last_analyst_run = now
                log_event(
                    "analyst_slow",
                    "Analyst worker exceeded its deadline; using its last completed signal set.",
                    severity="warn",
                )
            return

        if (
            self._last_analyst_run is not None
            and (now - self._last_analyst_run).total_seconds() < config.ANALYST_INTERVAL_SECONDS
        ):
            return

        snapshot = {
            symbol: bars.copy(deep=True)
            for symbol, bars in bars_by_symbol.items()
            if not bars.empty
        }
        if not snapshot:
            return

        symbols = list(snapshot)
        self._last_analyst_run = now
        self._analyst_started_at = now
        self._analyst_timed_out = False
        with self._analyst_lock:
            self._analyst_generation += 1
            generation = self._analyst_generation
            self._analyst_running = True

        def run_analysis():
            try:
                result = self._analyst.analyze_all(symbols, snapshot)
                with self._analyst_lock:
                    if generation == self._analyst_generation:
                        self._pending_analyst_signals = result
            except Exception as exc:
                with self._analyst_lock:
                    if generation == self._analyst_generation:
                        self._pending_analyst_error = exc
            finally:
                with self._analyst_lock:
                    if generation == self._analyst_generation:
                        self._analyst_running = False

        threading.Thread(target=run_analysis, daemon=True, name="market-analyst").start()

    def _get_analyst_signal(self, symbol: str):
        """Return the latest MarketSignal for *symbol*, or None."""
        last_success = getattr(self, "_last_analyst_success", None)
        if (
            last_success is not None
            and (datetime.now(timezone.utc) - last_success).total_seconds()
            > config.ANALYST_SIGNAL_TTL_SECONDS
        ):
            return None
        return self._analyst_signals.get(symbol)

    def _regime_blocks_strategy(self, symbol: str, strategy_name: str) -> bool:
        """Return True if the current regime should gate this strategy."""
        if not config.REGIME_STRATEGY_GATING:
            return False
        signal = self._get_analyst_signal(symbol)
        if signal is None:
            return False
        disabled = config.REGIME_DISABLED_STRATEGIES.get(signal.regime, [])
        return strategy_name in disabled

    def _reconcile_portfolio_with_broker(self, startup: bool):
        result = self.portfolio.sync_from_broker()
        now = datetime.now(timezone.utc)
        self.state.last_sync_at = now

        for position in result["adopted"]:
            policy = config.ADOPTED_POSITION_POLICY
            message = (
                f"[SYNC] Adopted broker position: {position.symbol} {position.side} "
                f"{position.qty:g} @ {position.entry_price:.2f} "
                f"(policy={policy}). "
                "This trade was already open at Alpaca, so the bot is tracking it "
                "in resume mode."
            )
            print(message)
            self.state.note(message)
            log_event("position_adopted", message, payload={**position.to_dict(), "policy": policy}, severity="warn")

        # Flatten policy: close adopted positions immediately.
        for position in result.get("flatten", []):
            message = (
                f"[SYNC] Flattening adopted position: {position.symbol} {position.side} "
                f"{position.qty:g} @ {position.entry_price:.2f} "
                "(policy=flatten)."
            )
            print(message)
            self.state.note(message)
            log_event("position_flatten", message, payload=position.to_dict(), severity="warn")
            self.broker.close_position_market(position.symbol)
            self.portfolio.positions.pop(position.symbol, None)

        for position in result["removed"]:
            if startup:
                log_event(
                    "stale_state_cleanup",
                    f"Dropped stale cached position for {position.symbol} during startup reconciliation.",
                    payload=position.to_dict(),
                    severity="warn",
                )
                continue

            exit_price = None
            if position.stop_order_id:
                order_snapshot = self.broker.get_order_snapshot(position.stop_order_id)
                if order_snapshot:
                    exit_price = order_snapshot.get("filled_avg_price")
            if exit_price is None:
                exit_price = self.broker.get_latest_price(position.symbol) or position.entry_price

            result_payload = self.portfolio.record_closed_position(
                position,
                exit_price=float(exit_price),
                reason="broker_reconciled_close",
                closed_at=now,
            )
            self._apply_closed_position(result_payload, reason="broker_reconciled_close", approximate=True)

    def _roll_day_if_needed(self):
        # Use the market-session boundary (defaults to 4:00 PM ET) instead
        # of raw UTC midnight so daily P&L aligns with the trading day.
        session_date = config.trading_session_date()
        if session_date == self.current_date:
            return

        self.current_date = session_date
        self.state.current_date = session_date
        try:
            self.day_start_equity = self.portfolio.get_equity()
        except Exception:
            self.day_start_equity = self.day_start_equity or self.state.last_equity

        if self._notifier:
            try:
                self._notifier.send_daily_summary(
                    wins=self.state.wins,
                    losses=self.state.losses,
                    daily_pnl=self.state.daily_realized_pnl,
                    equity=self.day_start_equity,
                    market="US"
                )
            except Exception:
                pass

        self.state.day_start_equity = self.day_start_equity
        self.state.daily_realized_pnl = 0.0
        self.state.consecutive_losses = 0
        self.state.new_entries_allowed = True
        self.state.entry_halt_reason = ""
        self.state.note(f"Rolled trading day to {session_date}.")
        log_event("day_roll", f"Started new tracking day {session_date}.", payload={"day_start_equity": self.day_start_equity})

        self.guardian.roll_day(self.day_start_equity)
        
        # Check for model drift at start of each new session
        try:
            drift_report = self._drift_monitor.check_all()
            if drift_report.get('should_retrain', False):
                print('[Drift] Model drift detected! Triggering retraining...')
                if self._ml_filter and hasattr(self._ml_filter, 'retrain_if_drifted'):
                    self._ml_filter.retrain_if_drifted()
        except Exception as e:
            print(f'[Drift] Check failed: {e}')

        # Generate end-of-previous-session performance report.
        try:
            report = generate_session_performance_report()
            if report.get("total_trades", 0) > 0:
                log_event(
                    "performance_report",
                    f"Session performance report generated: {report['total_trades']} trades, "
                    f"win rate {report['win_rate']:.1%}, total P&L ${report['total_pnl']:.2f}.",
                    payload=report,
                )
        except Exception:
            pass  # best effort — don't crash the bot for report generation

        # Run post-mortem analysis on today's losing trades
        try:
            import threading
            threading.Thread(target=self._post_mortem.run_daily_postmortem, daemon=True, name='post-mortem').start()
            print('[Post-Mortem] Daily analysis started in background.')
        except Exception as e:
            print(f'[Post-Mortem] Failed: {e}')

    def _refresh_entry_gate(self, equity_now: float):
        reasons = []
        if self._manual_entry_pause:
            reasons.append("Manually paused via Telegram.")
        if self.coordinator is not None:
            coordinator_allows_entries, coordinator_reason = self.coordinator.entries_allowed()
            if not coordinator_allows_entries:
                reasons.append(coordinator_reason)
        if self.day_start_equity:
            drawdown = equity_now - self.day_start_equity
            daily_loss_limit = self.day_start_equity * config.MAX_DAILY_LOSS_PCT_OF_EQUITY
            if drawdown <= -daily_loss_limit:
                reasons.append(
                    f"Daily loss cap reached ({drawdown:.2f} <= -{daily_loss_limit:.2f})."
                )

        if self.state.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            reasons.append(
                f"Consecutive loss cap reached ({self.state.consecutive_losses} >= {config.MAX_CONSECUTIVE_LOSSES})."
            )

        allowed = not reasons
        halt_reason = " ".join(reasons)

        snapshot = {
            "open_positions": self.portfolio.position_count(),
            "gross_exposure": round(self.portfolio.gross_exposure(), 2),
            "portfolio_heat": round(self.portfolio.portfolio_heat(), 2),
            "daily_realized_pnl": round(self.state.daily_realized_pnl, 2),
            "equity_now": round(equity_now, 2),
            "day_start_equity": round(self.day_start_equity or 0.0, 2),
            "consecutive_losses": self.state.consecutive_losses,
        }
        self.state.latest_guardrail_snapshot = snapshot

        if allowed != self.state.new_entries_allowed or (not allowed and halt_reason != self.state.entry_halt_reason):
            self.state.new_entries_allowed = allowed
            self.state.entry_halt_reason = halt_reason
            status = "allowed" if allowed else "halted"
            detail = halt_reason or "All runtime guardrails are currently clear."
            log_guardrail(rule_name="runtime_entry_gate", status=status, detail=detail, metrics=snapshot)
            log_event("entry_gate_changed", detail, payload={"allowed": allowed, "snapshot": snapshot}, severity="warn" if not allowed else "info")

    def _log_running_daily_pnl(self):
        if self.day_start_equity is None:
            return
        try:
            equity_now = self.portfolio.get_equity()
        except Exception:
            return
        trade_logger.log_daily_pnl(self.current_date, self.day_start_equity, equity_now)
        self._stress_tester.print_report()
        trade_logger.log_equity_snapshot(
            timestamp=datetime.now(timezone.utc),
            equity=equity_now,
            day_start_equity=self.day_start_equity,
        )

    def _process_tradingview_signals(self):
        """Consume and act on signals from TradingView."""
        signals = drain_signals()
        for sig in signals:
            symbol = sig["symbol"]
            action = sig["action"].lower()
            
            # Use 'shadow' mode to just log it without executing
            if getattr(config, "TRADINGVIEW_SIGNAL_MODE", "supplement") == "shadow":
                print(f"[Shadow Mode] TradingView Signal: {action.upper()} {symbol}")
                log_event("shadow_signal", f"TradingView shadow signal: {action.upper()} {symbol}", payload=sig, severity="info")
                continue

            position = self.portfolio.get(symbol)
            if action in {"buy", "long"}:
                if position and position.side == "short":
                    self._close_tradingview_position(symbol, position, sig, "tradingview_cover")
                    continue
                entry_side = "long" if position is None else None
            elif action == "short":
                if position and position.side == "long":
                    self._close_tradingview_position(symbol, position, sig, "tradingview_short_exit")
                    continue
                entry_side = "short" if position is None else None
            elif action in {"sell", "cover", "close"}:
                should_close = bool(position) and (
                    action != "cover" or position.side == "short"
                )
                if should_close:
                    self._close_tradingview_position(
                        symbol, position, sig, f"tradingview_{action}"
                    )
                continue
            else:
                continue

            if entry_side:
                # A webhook price alone cannot produce a defensible ATR stop.
                # Enrich it with broker bars and use the native risk pipeline.
                bars = self.broker.get_bars(
                    symbol, 5, "Minute", limit=max(50, config.ATR_PERIOD + 2)
                )
                if bars.empty:
                    log_event(
                        "tradingview_skipped",
                        f"No market data available for TradingView signal {symbol}.",
                        payload=sig,
                        severity="warn",
                    )
                    continue
                spec = {
                    "strategy": type("TradingViewStrategy", (), {"name": sig["strategy"]})(),
                    "timeframe_amount": 5,
                    "timeframe_unit": "Minute",
                    "trail_atr_multiple": 1.5,
                }
                self._try_open_position(symbol, spec, bars, entry_side)

    def _close_tradingview_position(self, symbol: str, position, signal: dict, reason: str) -> bool:
        try:
            exit_price = self._tradingview_exit_price(symbol, signal)
        except Exception as exc:
            log_event(
                "tradingview_close_skipped",
                f"Could not safely close {symbol} from the webhook signal.",
                payload={"symbol": symbol, "reason": reason, "error": str(exc)},
                severity="warn",
            )
            return False
        self._close_position(symbol, position, exit_price, reason=reason)
        return True

    def _tradingview_exit_price(self, symbol: str, signal: dict) -> float:
        price = self.broker.get_latest_price(symbol)
        if price is None:
            price = signal.get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError(f"No valid market price is available to close {symbol}.")
        return price

    def _process_symbol(self, symbol: str, spec: dict, bars: pd.DataFrame | None = None):
        if bars is None:
            if not self.broker.is_market_open(symbol):
                return
            bars = self.broker.get_bars(
                symbol,
                spec["timeframe_amount"],
                spec["timeframe_unit"],
                limit=config.BAR_FETCH_LIMIT,
            )
        strategy = spec["strategy"]
        strategy_name = strategy.name
        if bars.empty or len(bars) < strategy.min_bars_required:
            return

        self._recent_closes[symbol] = bars['close']
        latest_bar = bars.iloc[-1]
        latest_bar_time = bars.index[-1]

        position = self.portfolio.get(symbol)

        if position is not None:
            self._update_trailing_stop(symbol, position, latest_bar, spec["trail_atr_multiple"])

            # observe_only adopted positions: update prices for visibility
            # but don't run strategy exits — the position is not ours to manage.
            is_observe_only = (
                position.adopted
                and position.strategy == "adopted_on_restart"
                and config.ADOPTED_POSITION_POLICY == "observe_only"
            )
            if not is_observe_only:
                self._try_close_position(position, spec, bars)
                if self.portfolio.get(symbol) is None:
                    position = None

        is_new_bar = latest_bar_time != self.last_processed_bar[symbol]

        if position is None and is_new_bar:
            data_status = (
                getattr(self, "_market_data_snapshot", {})
                .get("symbols", {})
                .get(symbol, {})
                .get("status", "available")
            )
            if data_status in {"stale", "missing", "error"}:
                reason = f"Entry skipped because {symbol} market data is {data_status}."
                self._record_signal_state(
                    symbol, strategy_name, "n/a", "stale_market_data",
                    float(latest_bar["close"]), latest_bar_time, reason,
                )
                log_event(
                    "entry_blocked_market_data_quality",
                    reason,
                    payload={"symbol": symbol, "status": data_status},
                    severity="warn",
                )
            # --- Regime-based strategy gating ---
            elif self._regime_blocks_strategy(symbol, strategy_name):
                signal = self._get_analyst_signal(symbol)
                regime_label = signal.regime if signal else "unknown"
                self._record_signal_state(
                    symbol, strategy_name, "n/a", "regime_gated",
                    float(latest_bar["close"]), latest_bar_time,
                    f"Strategy '{strategy_name}' gated in {regime_label} regime.",
                )
                log_event(
                    "regime_gated",
                    f"{symbol}: '{strategy_name}' skipped — regime is {regime_label}.",
                    payload={"symbol": symbol, "strategy": strategy_name, "regime": regime_label},
                )
            else:
                entry_side = strategy.evaluate_entry(bars)
                if entry_side in ("long", "short"):
                    # --- Analyst confidence gate ---
                    analyst_signal = self._get_analyst_signal(symbol)
                    if analyst_signal and analyst_signal.confidence < config.ANALYST_MIN_CONFIDENCE_FOR_ENTRY:
                        reason = (
                            f"Analyst confidence too low ({analyst_signal.confidence} "
                            f"< {config.ANALYST_MIN_CONFIDENCE_FOR_ENTRY})."
                        )
                        self._record_signal_state(
                            symbol, strategy_name, entry_side, "low_confidence",
                            float(latest_bar["close"]), latest_bar_time, reason,
                        )
                        log_signal(
                            symbol=symbol, strategy=strategy_name, side=entry_side,
                            action="entry", status="low_confidence",
                            price=float(latest_bar["close"]),
                            bar_time=latest_bar_time.to_pydatetime(),
                            timeframe=_timeframe_label(spec), reason=reason,
                        )
                    else:
                        self._try_open_position(symbol, spec, bars, entry_side)

        # Shadow-evaluate candidate strategies
        try:
            for candidate_name in list(self._shadow_trader.candidates.keys()):
                self._shadow_trader.evaluate_bar(symbol, bars, candidate_name)
        except Exception:
            pass

        self.last_processed_bar[symbol] = latest_bar_time
        self.state.last_processed_bar[symbol] = latest_bar_time.isoformat()

    def _try_close_position(self, position, spec: dict, bars: pd.DataFrame):
        strategy = spec["strategy"]
        strategy_name = strategy.name

        try:
            from bot.rl_agent import RLExitAgent
            if not hasattr(self, "_rl_exit_agent"):
                self._rl_exit_agent = RLExitAgent()
            
            current_price = float(bars.iloc[-1]["close"])
            if position.side == "long":
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - current_price) / position.entry_price
                
            from bot.indicators import atr
            atr_val = float(atr(bars, 14).iloc[-1])
            
            # The RL agent dynamically decides whether to hold or close based on PnL momentum and Volatility
            if self._rl_exit_agent.should_exit(pnl_pct, atr_val, 10):
                self._record_signal_state(position.symbol, strategy_name, position.side, "rl_exit", current_price, bars.index[-1], "RL Agent triggered exit")
                log_signal(position.symbol, strategy_name, position.side, "exit", "rl_exit", current_price, bars.index[-1].to_pydatetime(), _timeframe_label(spec), "RL Agent triggered exit")
                self._close_position(position.symbol, position, current_price, reason="rl_exit")
                return
        except Exception:
            pass # Fallback to deterministic math exits if RL agent fails

        # Fallback to standard algorithmic exit
        if strategy.evaluate_exit(bars, position.side):
            self._close_position(position.symbol, position, float(bars.iloc[-1]["close"]), reason="strategy_exit")

    def _confirm_entry_fill(self, order_id: str, requested_qty: float, fallback_price: float):
        """Confirm a broker fill and cancel any unfilled remainder.

        A submitted limit order is not a position. This prevents a fallback
        price from creating a phantom local holding and protective stop when
        the broker filled nothing. Partial fills are tracked at their actual
        quantity so the stop matches the broker position.
        """
        try:
            price_hint = self.broker.get_fill_price(order_id, fallback=fallback_price)
        except Exception:
            price_hint = fallback_price
        try:
            snapshot = self.broker.get_order_snapshot(order_id)
        except Exception:
            snapshot = None
        if not isinstance(snapshot, dict):
            try:
                self.broker.cancel_order(order_id)
                snapshot = self.broker.get_order_snapshot(order_id)
            except Exception:
                snapshot = None
            if not isinstance(snapshot, dict):
                return None, None, "unknown"

        status = str(snapshot.get("status", "")).lower().rsplit(".", 1)[-1]
        terminal_statuses = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}

        def filled_quantity(order_snapshot: dict) -> float:
            try:
                quantity = float(order_snapshot.get("filled_qty", 0.0) or 0.0)
                return quantity if math.isfinite(quantity) and quantity > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0

        filled_qty = filled_quantity(snapshot)
        if status == "filled" and filled_qty <= 0:
            # Compatible brokers may omit filled_qty on a completed order.
            filled_qty = float(requested_qty)

        if status != "filled":
            # Cancel both zero-fill orders and a partial fill's remainder,
            # then re-read to catch any fill that raced with the cancellation.
            try:
                self.broker.cancel_order(order_id)
                refreshed = self.broker.get_order_snapshot(order_id)
            except Exception:
                refreshed = None
            if isinstance(refreshed, dict):
                snapshot = refreshed
                status = str(snapshot.get("status", status)).lower().rsplit(".", 1)[-1]
                filled_qty = max(filled_qty, filled_quantity(snapshot))
            if status not in terminal_statuses:
                return None, None, "unknown"

        if filled_qty <= 0:
            return None, None, "unfilled" if isinstance(snapshot, dict) else "unknown"

        raw_price = snapshot.get("filled_avg_price")
        try:
            fill_price = float(raw_price) if raw_price is not None else float(price_hint)
        except (TypeError, ValueError):
            fill_price = float(price_hint)
        if not math.isfinite(fill_price) or fill_price <= 0:
            return None, None, "unknown"
        return min(filled_qty, float(requested_qty)), fill_price, status

    def _try_open_position(self, symbol: str, spec: dict, bars: pd.DataFrame, side: str):
        # Recheck the broker session at the shared order-entry boundary. Normal
        # strategy entries are already fed only during market hours, but other
        # sources (notably TradingView webhooks) can reach this method directly.
        # Fail closed if the clock cannot be verified; never infer an open
        # session from local time or stale signals.
        try:
            if not self.broker.is_market_open(symbol):
                log_event(
                    "entry_blocked_market_closed",
                    f"{symbol} entry blocked because Alpaca reports the market closed.",
                    payload={
                        "symbol": symbol,
                        "strategy": getattr(spec.get("strategy"), "name", "unknown"),
                    },
                    severity="info",
                )
                return
        except Exception as exc:
            log_event(
                "entry_blocked_market_clock_unavailable",
                f"{symbol} entry blocked because Alpaca's market clock could not be verified.",
                payload={"symbol": symbol, "error": type(exc).__name__},
                severity="warn",
            )
            return

        strategy_name = spec["strategy"].name
        atr_value = calc_atr(bars, period=config.ATR_PERIOD).iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            return

        price = float(bars.iloc[-1]["close"])
        equity = self.portfolio.get_equity()
        fractionable = "/" in symbol

        # --- Gather analyst + watchdog context for adaptive sizing ---
        analyst_signal = self._get_analyst_signal(symbol)
        regime = analyst_signal.regime if analyst_signal else ""
        analyst_confidence = analyst_signal.confidence if analyst_signal else 100
        watchdog_mult = self._watchdog.sizing_multiplier if self._watchdog else 1.0

        # --- L2 Order Book Imbalance (OBI) Filter ---
        if hasattr(self, "_streamer") and self._streamer is not None:
            obi = self._streamer.get_order_book_imbalance(symbol)
            if side == "long" and obi < -0.40:
                reason = f"L2 Order Book Block: Heavy resting sell pressure detected (OBI={obi:.2f})"
                self._record_signal_state(symbol, strategy_name, side, "obi_blocked", price, bars.index[-1], reason)
                log_event("obi_gate_blocked", reason, payload={"obi": obi})
                return
            elif side == "short" and obi > 0.40:
                reason = f"L2 Order Book Block: Heavy resting buy pressure detected (OBI={obi:.2f})"
                self._record_signal_state(symbol, strategy_name, side, "obi_blocked", price, bars.index[-1], reason)
                log_event("obi_gate_blocked", reason, payload={"obi": obi})
                return

        # --- ML Signal Filter (XGBoost) ---
        win_prob = None
        if self._ml_filter is not None and self._ml_filter.model is not None:
            try:
                win_prob = self._ml_filter.predict_win_probability(
                    symbol=symbol,
                    side=side,
                    strategy=strategy_name,
                    atr_value=float(atr_value),
                    analyst_signal=analyst_signal,
                )
                if win_prob < getattr(config, "ML_MIN_WIN_PROB", 0.50):
                    reason = f"ML Filter Blocked: Predicted win prob {win_prob:.1%} < {getattr(config, 'ML_MIN_WIN_PROB', 0.50):.1%}"
                    self._record_signal_state(symbol, strategy_name, side, "ml_blocked", price, bars.index[-1], reason)
                    log_signal(
                        symbol=symbol, strategy=strategy_name, side=side, action="entry",
                        status="ml_blocked", price=price, bar_time=bars.index[-1].to_pydatetime(),
                        timeframe=_timeframe_label(spec), reason=reason,
                    )
                    log_event("ml_gate_blocked", reason, payload={"win_prob": win_prob})
                    return
                else:
                    log_event("ml_gate_passed", f"ML win prob {win_prob:.1%} >= {getattr(config, 'ML_MIN_WIN_PROB', 0.50):.1%}", payload={"win_prob": win_prob})
            except Exception as exc:
                pass

        sizing = self.risk_manager.size_position(
            equity=equity,
            atr_value=atr_value,
            price=price,
            side=side,
            fractionable=fractionable,
            regime=regime,
            analyst_confidence=analyst_confidence,
            watchdog_multiplier=watchdog_mult,
            ml_win_prob=win_prob,
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
            reason = " ".join(filter_decision.reasons)
            self._record_signal_state(symbol, strategy_name, side, "filtered", price, bars.index[-1], reason)
            log_signal(
                symbol=symbol,
                strategy=strategy_name,
                side=side,
                action="entry",
                status="filtered",
                price=price,
                bar_time=bars.index[-1].to_pydatetime(),
                timeframe=_timeframe_label(spec),
                reason=reason,
            )
            log_event("signal_filtered", f"{symbol} {side} signal filtered.", payload={"reason": reason, "metrics": filter_decision.metrics})
            return

        # --- US Options (F&O) confirmation gate ---
        # For symbols in US_OPTIONS_SYMBOLS, check PCR/max-pain before entering.
        # Only blocks when the options signal *directly* opposes the equity side.
        if (
            self._us_options_scanner is not None
            and getattr(config, "US_OPTIONS_GATE_ENTRIES", True)
            and symbol in getattr(config, "US_OPTIONS_SYMBOLS", [])
        ):
            try:
                opt_signal = self._us_options_scanner.scan(symbol)
                if opt_signal is not None:
                    opposed = (
                        (side == "long"  and opt_signal.direction == "bearish") or
                        (side == "short" and opt_signal.direction == "bullish")
                    )
                    if opposed:
                        reason = (
                            f"US options gate: {opt_signal.direction} options bias "
                            f"(PCR={opt_signal.pcr:.2f}, max_pain={opt_signal.max_pain:.1f}) "
                            f"opposes {side} entry."
                        )
                        self._record_signal_state(symbol, strategy_name, side, "options_blocked", price, bars.index[-1], reason)
                        log_signal(
                            symbol=symbol,
                            strategy=strategy_name,
                            side=side,
                            action="entry",
                            status="options_blocked",
                            price=price,
                            bar_time=bars.index[-1].to_pydatetime(),
                            timeframe=_timeframe_label(spec),
                            reason=reason,
                        )
                        log_event("options_gate_blocked", f"{symbol} {side} blocked by options signal.", payload={"pcr": opt_signal.pcr, "direction": opt_signal.direction})
                        return
            except Exception as exc:
                pass  # Options gate failure is non-fatal — let the trade proceed

        # --- Earnings Calendar Filter ---
        if self._earnings_calendar is not None:
            try:
                earnings_check = self._earnings_calendar.check(symbol)
                if earnings_check.blocked:
                    reason = f"Earnings Gate: {earnings_check.reason}"
                    self._record_signal_state(symbol, strategy_name, side, "earnings_blocked", price, bars.index[-1], reason)
                    log_signal(
                        symbol=symbol,
                        strategy=strategy_name,
                        side=side,
                        action="entry",
                        status="earnings_blocked",
                        price=price,
                        bar_time=bars.index[-1].to_pydatetime(),
                        timeframe=_timeframe_label(spec),
                        reason=reason,
                    )
                    log_event("earnings_gate_blocked", reason, payload={"symbol": symbol, "earnings_date": str(earnings_check.earnings_date)})
                    try:
                        self._notifier.send_blocked(symbol, side, reason, strategy_name, market="US")
                    except Exception:
                        pass
                    return
            except Exception as exc:
                pass  # Earnings failure is non-fatal
        # --- Dynamic Correlation Guard ---
        if getattr(config, "DYNAMIC_CORRELATION_GUARD", True) and len(self.portfolio.positions) > 0:
            candidate_returns = bars['close'].pct_change().dropna()
            for open_sym, pos in self.portfolio.positions.items():
                if pos.side == side and open_sym in self._recent_closes:
                    open_series = self._recent_closes[open_sym].pct_change().dropna()
                    # Align and correlate
                    aligned = pd.concat([candidate_returns, open_series], axis=1, join='inner')
                    if len(aligned) > 20:
                        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                        if corr > 0.85:
                            reason = f"Dynamic Correlation Block: highly correlated (ρ={corr:.2f}) with open pos {open_sym}"
                            self._record_signal_state(symbol, strategy_name, side, "blocked", price, bars.index[-1], reason)
                            log_signal(
                                symbol=symbol, strategy=strategy_name, side=side, action="entry",
                                status="blocked", price=price, bar_time=bars.index[-1].to_pydatetime(),
                                timeframe=_timeframe_label(spec), reason=reason,
                            )
                            log_event(
                                "entry_blocked",
                                reason,
                                payload={"symbol": symbol, "strategy": strategy_name},
                                severity="info",
                            )
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
            daily_realized_pnl=self.state.daily_realized_pnl,
            day_start_equity=self.day_start_equity,
            consecutive_losses=self.state.consecutive_losses,
            new_entries_allowed=self.state.new_entries_allowed,
            entry_halt_reason=self.state.entry_halt_reason,
        )
        if not guardrails.allowed:
            reason = " ".join(guardrails.reasons)
            self._record_signal_state(symbol, strategy_name, side, "blocked", price, bars.index[-1], reason)
            log_signal(
                symbol=symbol,
                strategy=strategy_name,
                side=side,
                action="entry",
                status="blocked",
                price=price,
                bar_time=bars.index[-1].to_pydatetime(),
                timeframe=_timeframe_label(spec),
                reason=reason,
            )
            log_guardrail(
                rule_name="entry_guardrails",
                status="blocked",
                detail=reason,
                symbol=symbol,
                strategy=strategy_name,
                metrics=guardrails.metrics,
            )
            log_event(
                "entry_blocked",
                f"Blocked {side} on {symbol}: {reason}",
                payload={
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "metrics": guardrails.metrics,
                },
                severity="info",
            )
            return


        # --- Phase 1.5: optional Virtual Human Brain advisory ---
        if self.virtual_brain is not None:
            try:
                approved, intuition = self.virtual_brain.evaluate_price_action(symbol, bars, strategy_name)
                brain_log = f"**Virtual Brain ({symbol})**:\n{intuition}\n"
                self.desk.update_section("Virtual Brain Decisions", brain_log)
                print(f"[Virtual Brain] {symbol} {strategy_name}: {intuition.replace(chr(10), ' ')}")
                if not approved:
                    log_signal(symbol, strategy_name, side, "rejected", "virtual_brain_veto", price, bars.index[-1].to_pydatetime(), _timeframe_label(spec), intuition)
                    return
            except Exception as exc:
                log_event(
                    "virtual_brain_error",
                    str(exc),
                    payload={"symbol": symbol, "strategy": strategy_name},
                    severity="warn",
                )
                if config.AI_AGENT_FAILURE_POLICY == "block":
                    return

        # --- Multi-Agent Debate (Phase 2) ---
        if self.committee is not None:
            try:
                garch_vol = self._garch_forecaster.forecast_volatility(bars["close"])
                stress_data = self._stress_tester.run_stress_test()
                stress_str = "\n".join(
                    f"- {scenario}: {data['portfolio_pct']:+.2f}% (${data['pnl_impact']:+.2f})"
                    for scenario, data in stress_data.items()
                )
                approved, transcript = self.committee.debate_trade(
                    symbol=symbol,
                    side=side,
                    strategy=strategy_name,
                    current_price=price,
                    atr=float(atr_value),
                    garch_vol=garch_vol,
                    stress_report=stress_str,
                )
                self.desk.update_section("Active Debates", transcript)
                if not approved:
                    reason = "Multi-Agent Committee REJECTED the trade."
                    self._record_signal_state(symbol, strategy_name, side, "committee_rejected", price, bars.index[-1], reason)
                    log_event("committee_rejected", f"Committee rejected {side} on {symbol}", severity="info")
                    return
            except Exception as exc:
                log_event(
                    "committee_error",
                    str(exc),
                    payload={"symbol": symbol, "strategy": strategy_name},
                    severity="warn",
                )
                if config.AI_AGENT_FAILURE_POLICY == "block":
                    return
        
        # --- Guardian Pre-Trade Check ---
        allowed, reason = self.guardian.check_pre_trade(equity, config.RISK_PER_TRADE_PCT)
        if not allowed:
            log_signal(symbol, strategy_name, side, 'rejected', 'guardian_block', price, bars.index[-1].to_pydatetime(), _timeframe_label(spec), reason)
            return

        guardian_mult = self.guardian.get_sizing_multiplier()
        if guardian_mult <= 0:
            return  # Guardian blocked all trading
        sizing.qty = sizing.qty * guardian_mult if fractionable else int(sizing.qty * guardian_mult)
        if sizing.qty <= 0:
            return

        # Marketable Limit Order (Max 0.2% slippage allowed)
        if side == "long":
            limit_price = round(price * 1.002, 2)
        else:
            limit_price = round(price * 0.998, 2)
            
        entry_order_id = self.broker.submit_limit_order(symbol, sizing.qty, side, limit_price)
        filled = self._confirm_entry_fill(entry_order_id, sizing.qty, price)
        if filled[0] is None:
            if self._watchdog:
                self._watchdog.record_order_result(False)
            message = (
                f"Entry order {entry_order_id} for {symbol} was not confirmed as filled."
                if filled[2] == "unfilled"
                else f"Entry order {entry_order_id} for {symbol} has an unknown fill state."
            )
            log_event(
                "entry_order_unconfirmed",
                message,
                payload={"symbol": symbol, "entry_order_id": entry_order_id, "status": filled[2]},
                severity="error" if filled[2] == "unknown" else "warn",
            )
            if filled[2] == "unknown":
                self._mark_runtime_degraded(message)
                try:
                    self.broker.close_position_market(symbol)
                except Exception as exc:
                    log_event(
                        "entry_unknown_emergency_close_failed",
                        f"Could not resolve possible {symbol} exposure after unknown order state.",
                        payload={"symbol": symbol, "entry_order_id": entry_order_id, "error": str(exc)},
                        severity="critical",
                    )
            return

        filled_qty, fill_price, order_status = filled
        if self._watchdog:
            self._watchdog.record_order_result(True)
        if filled_qty < sizing.qty:
            log_event(
                "entry_order_partial_fill",
                f"{symbol} entry partially filled; tracking only confirmed shares.",
                payload={
                    "symbol": symbol,
                    "entry_order_id": entry_order_id,
                    "requested_qty": sizing.qty,
                    "filled_qty": filled_qty,
                    "status": order_status,
                },
                severity="warn",
            )
        fill_ratio = filled_qty / sizing.qty if sizing.qty else 0.0
        filled_risk_amount = sizing.risk_amount * fill_ratio
        stop_price = self.risk_manager.initial_stop_price(fill_price, atr_value, side)
        try:
            stop_order_id = self.broker.submit_stop_order(symbol, filled_qty, side, stop_price)
        except Exception as exc:
            message = f"Could not protect filled {symbol} position; requesting emergency close."
            log_event(
                "protective_stop_failed",
                message,
                payload={
                    "symbol": symbol,
                    "entry_order_id": entry_order_id,
                    "filled_qty": filled_qty,
                    "error": str(exc),
                },
                severity="error",
            )
            self._mark_runtime_degraded(message, str(exc))
            try:
                self.broker.close_position_market(symbol)
            except Exception as close_exc:
                log_event(
                    "emergency_close_failed",
                    f"Emergency close failed for {symbol}; broker reconciliation is required.",
                    payload={"symbol": symbol, "error": str(close_exc)},
                    severity="critical",
                )
            return

        position = OpenPosition(
            symbol=symbol,
            side=side,
            qty=filled_qty,
            entry_price=fill_price,
            entry_time=datetime.now(timezone.utc),
            atr_at_entry=float(atr_value),
            stop_price=stop_price,
            stop_order_id=stop_order_id,
            extreme_price=fill_price,
            strategy=strategy_name,
            risk_amount=filled_risk_amount,
            notional=float(filled_qty * fill_price),
            asset_class=config.symbol_asset_class(symbol),
            entry_signal=f"{strategy_name}:{side}",
            entry_order_id=entry_order_id,
            metadata={
                "committee_approved": self.committee is not None,
                "virtual_brain_approved": self.virtual_brain is not None,
                "ml_win_prob": win_prob if win_prob is not None else 0.0,
                "ml_features": (
                    self._ml_filter.build_feature_values(side, float(atr_value), analyst_signal)
                    if self._ml_filter is not None
                    else {}
                ),
            }
        )
        self.portfolio.open_position(position)
        self._record_signal_state(symbol, strategy_name, side, "entered", fill_price, bars.index[-1], "")
        log_signal(
            symbol=symbol,
            strategy=strategy_name,
            side=side,
            action="entry",
            status="entered",
            price=fill_price,
            bar_time=bars.index[-1].to_pydatetime(),
            timeframe=_timeframe_label(spec),
            reason="",
        )

        cap_note = "capped by max-position-pct" if sizing.capped_by_max_position else "uncapped"
        message = (
            f"[{symbol}] Opened {side} {filled_qty:g} @ {fill_price:.2f} "
            f"(ATR={atr_value:.2f}, stop={stop_price:.2f}, "
            f"risking ~${filled_risk_amount:.2f}, {cap_note})"
        )
        print(message)
        self.state.note(message)
        log_event("position_opened", message, payload=position.to_dict())
        self._notifier.send_position_opened(
            symbol=symbol,
            side=side,
            qty=filled_qty,
            entry_price=fill_price,
            stop_price=stop_price,
            risk_amount=filled_risk_amount,
            strategy=strategy_name,
            equity=equity,
            market="US",
        )

    def _sync_consensus_positions(self):
        """Writes the current open positions to the shared consensus.md memory."""
        if not self.portfolio.positions:
            self.desk.update_section("Open Positions", "None.")
            return

        lines = []
        for sym, pos in self.portfolio.positions.items():
            lines.append(f"- **{sym}** ({pos.side.upper()}): {pos.qty} shares @ ${pos.entry_price:.2f}")
        self.desk.update_section("Open Positions", "\n".join(lines))

    def _record_signal_state(
        self,
        symbol: str,
        strategy: str,
        side: str,
        status: str,
        price: float,
        bar_time,
        reason: str,
    ):
        self.state.last_signal_by_symbol[symbol] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "side": side,
            "status": status,
            "price": round(float(price), 4),
            "bar_time": bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time),
            "reason": reason,
        }

    def _update_trailing_stop(self, symbol: str, position: OpenPosition, latest_bar, trail_atr_multiple):
        extreme = float(latest_bar["high"] if position.side == "long" else latest_bar["low"])
        self.portfolio.update_extreme_price(symbol, extreme)

        if not trail_atr_multiple:
            return

        new_stop = self.risk_manager.update_trailing_stop(
            current_stop=position.stop_price,
            extreme_price=position.extreme_price,
            atr_value=position.atr_at_entry,
            trail_multiple=trail_atr_multiple,
            side=position.side,
        )

        if new_stop != position.stop_price:
            if position.stop_order_id:
                self.broker.cancel_order(position.stop_order_id)
            new_order_id = self.broker.submit_stop_order(symbol, position.qty, position.side, new_stop)
            position.stop_price = new_stop
            position.stop_order_id = new_order_id
            print(f"[{symbol}] Trailing stop moved to {new_stop:.2f}")
            log_event(
                "trailing_stop_update",
                f"{symbol} trailing stop moved to {new_stop:.2f}.",
                payload={"symbol": symbol, "new_stop": round(new_stop, 4), "side": position.side},
            )

    def _close_position(self, symbol: str, position: OpenPosition, exit_price: float, reason: str):
        if position.stop_order_id:
            self.broker.cancel_order(position.stop_order_id)
        self.broker.close_position_market(symbol)
        result = self.portfolio.close_position(symbol, exit_price, reason)
        if result is None:
            return
        self._apply_closed_position(result, reason=reason, approximate=False)
        pnl = result["pnl"]
        pnl_str = f"{pnl:+.2f}"
        print(f"[{symbol}] Closed {position.side} @ ~{exit_price:.2f} ({reason}), P/L ${pnl_str}")
        try:
            self._notifier.send_position_closed(
                symbol=symbol,
                side=position.side,
                exit_price=exit_price,
                pnl=pnl,
                strategy=position.strategy,
                equity=self.portfolio.get_equity(),
                market="US",
                reason=reason,
            )
        except Exception:
            pass
        self._sync_consensus_positions()

    def _apply_closed_position(self, result: dict, reason: str, approximate: bool):
        pnl = float(result["pnl"])
        position = result["position"]
        self.state.cumulative_realized_pnl += pnl
        self.state.daily_realized_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
            self.state.consecutive_losses = 0
        elif pnl < 0:
            self.state.losses += 1
            self.state.consecutive_losses += 1
            
        self.guardian.record_trade_outcome(pnl)

        # Feed trade outcome back to the XGBoost aggregator (no-op for weighted).
        if self._analyst is not None:
            notional = abs(position.entry_price * position.qty) if position.qty else 0.0
            if notional > 0:
                actual_return = pnl / notional
                try:
                    self._analyst.aggregator.record_outcome(position.symbol, actual_return)
                except Exception as exc:
                    log_event("aggregator_outcome_error", str(exc), severity="warn")

        regime = ""
        garch_vol = 0.0
        try:
            analyst_signal = self._get_analyst_signal(position.symbol)
            if analyst_signal:
                regime = analyst_signal.regime
            if hasattr(self, "_garch_forecaster") and self._garch_forecaster is not None and position.symbol in self._recent_closes:
                garch_vol = self._garch_forecaster.forecast_volatility(self._recent_closes[position.symbol])
        except Exception:
            pass

        payload = {
            "symbol": position.symbol,
            "side": position.side,
            "strategy": position.strategy,
            "entry_order_id": position.entry_order_id,
            "entry_time": position.entry_time.isoformat(),
            "entry_price": position.entry_price,
            "atr_at_entry": position.atr_at_entry,
            "ml_features": position.metadata.get("ml_features", {}),
            "exit_reason": reason,
            "pnl": round(pnl, 2),
            "approximate": approximate,
            "regime": regime,
            "garch_vol": float(garch_vol) if garch_vol is not None else 0.0,
            "committee_approved": position.metadata.get("committee_approved", False),
            "virtual_brain_approved": position.metadata.get("virtual_brain_approved", False),
            "ml_win_prob": float(position.metadata.get("ml_win_prob", 0.0)),
        }
        self.state.note(
            f"Closed {position.symbol} {position.side} for {pnl:+.2f} via {reason}."
        )
        log_event(
            "position_closed",
            f"{position.symbol} closed via {reason} for {pnl:+.2f}.",
            payload=payload,
            severity="warn" if approximate else "info",
        )
        if self._ml_filter:
            self._ml_filter.on_trade_closed()

    def _persist_state(self):
        self.state.last_heartbeat_at = datetime.now(timezone.utc)
        self.state.open_positions = self.portfolio.to_dict()
        for symbol, timestamp in self.last_processed_bar.items():
            self.state.last_processed_bar[symbol] = None if timestamp is None else timestamp.isoformat()

        # Attach analyst + watchdog snapshots to persisted state.
        if self._watchdog:
            self.state.latest_guardrail_snapshot["watchdog"] = self._watchdog.to_dict()
        if self._analyst_signals:
            self.state.latest_guardrail_snapshot["analyst_signals"] = {
                sym: {"direction": s.direction, "confidence": s.confidence, "regime": s.regime}
                for sym, s in self._analyst_signals.items()
            }

        market_data_snapshot = getattr(self, "_market_data_snapshot", {})
        self.state.market_data_snapshot = market_data_snapshot

        self.state_store.save(self.state)
        if self.coordinator is not None:
            self.coordinator.heartbeat(
                "us",
                status=self.state.runtime_status if self.state.runtime_status in {"degraded", "stopped"} else "running",
                metrics={
                    "currency": "USD",
                    "equity": round(float(self.state.last_equity or 0.0), 2),
                    "open_positions": self.portfolio.position_count(),
                    "entry_gate": "open" if self.state.new_entries_allowed else "halted",
                    "broker_connected": self.state.broker_connected,
                    "market_data": market_data_snapshot,
                },
            )


if __name__ == "__main__":
    try:
        config.validate_broker_backend()
        config.validate_alpaca_credentials()
        TradingBot().run_forever()
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)
