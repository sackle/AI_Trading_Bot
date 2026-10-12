"""Bayesian strategy parameter optimisation using Optuna.

Re-tunes strategy hyper-parameters on a rolling window of historical bars
every `OPTIMIZER_REFIT_INTERVAL_DAYS` days (default 7). Falls back gracefully
when `optuna` is not installed.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna

    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]

import config

logger = logging.getLogger(__name__)

_RESULTS_FILENAME = "optimizer_results.csv"


class StrategyOptimizer:
    """Auto-tunes strategy parameters on a rolling window every N days."""

    def __init__(self, refit_interval_days: int | None = None) -> None:
        self.refit_interval_days = (
            refit_interval_days
            if refit_interval_days is not None
            else getattr(config, "OPTIMIZER_REFIT_INTERVAL_DAYS", 7)
        )
        self._last_optimized: datetime | None = None
        self._warning_logged = False
        self._state_lock = threading.Lock()
        self._optimization_thread: threading.Thread | None = None
        self._pending_updates: dict[str, dict[str, Any]] = {}
        os.makedirs(config.runtime_path(config.RUNTIME_DIR), exist_ok=True)

    def maybe_optimize(self, strategies: dict, broker: Any) -> None:
        """Schedule due optimization without blocking the trading tick."""
        if not getattr(config, "OPTIMIZER_ENABLED", False):
            return

        if not OPTUNA_AVAILABLE:
            if not self._warning_logged:
                print("[Optimizer] optuna is not installed — skipping parameter tuning.")
                logger.warning("optuna is not installed. Optimizer is disabled.")
                self._warning_logged = True
            return

        now = datetime.now(timezone.utc)
        with self._state_lock:
            if self._optimization_thread and self._optimization_thread.is_alive():
                return
            if (
                self._last_optimized is not None
                and (now - self._last_optimized).days < self.refit_interval_days
            ):
                return

            # Record the attempt before starting the thread. The bot calls this
            # every tick, so recording afterward could schedule duplicate fits.
            self._last_optimized = now
            self._optimization_thread = threading.Thread(
                target=self._run_optimization_safely,
                args=(strategies, broker),
                name="strategy-optimizer",
                daemon=True,
            )
            self._optimization_thread.start()
            print("[Optimizer] Background tuning scheduled; trading tick continues.")

    def _run_optimization_safely(self, strategies: dict, broker: Any) -> None:
        try:
            self._run_optimization(strategies, broker)
        except Exception:
            logger.exception("Background strategy optimization failed")

    def apply_pending_updates(self, strategies: dict) -> list[str]:
        """Apply completed tuning results on the trading thread between ticks."""
        with self._state_lock:
            pending = self._pending_updates
            self._pending_updates = {}

        applied: list[str] = []
        for symbol, best_params in pending.items():
            strategy = strategies.get(symbol, {}).get("strategy")
            if strategy is None:
                continue
            self._apply_best_params(strategy, best_params)
            applied.append(symbol)
        return applied

    def _run_optimization(self, strategies: dict, broker: Any) -> None:
        logger.info("Running optimization for all strategies")
        max_trials = getattr(config, "OPTIMIZER_MAX_TRIALS", 50)

        for symbol, strat_info in strategies.items():
            strategy = strat_info.get("strategy")
            timeframe_amount = strat_info.get("timeframe_amount", 1)
            timeframe_unit = strat_info.get("timeframe_unit", "Day")
            if strategy is None:
                continue

            strategy_name = strategy.__class__.__name__
            try:
                bars = broker.get_bars(
                    symbol, timeframe_amount, timeframe_unit, limit=2000
                )
            except Exception as exc:
                logger.error("Failed to fetch bars for %s: %s", symbol, exc)
                continue

            if bars is None or len(bars) < 100:
                logger.warning("Not enough bars to optimize %s", symbol)
                continue

            # Use last ~180 days of bars so strategies with long EMA periods
            # (e.g. trend_following needs 200+ bars) have enough data for the
            # walk-forward backtest. Previously 90 days, but that left equity
            # strategies with only ~146 4h-bars -- below the 202-bar minimum.
            bars = self._last_n_days(bars, days=180)
            if len(bars) < 100:
                logger.warning("Not enough bars in 180-day window for %s", symbol)
                continue

            study = optuna.create_study(direction="maximize")
            try:
                study.optimize(
                    lambda trial, s=strategy, b=bars: self._objective(trial, s, b),
                    n_trials=max_trials,
                )
            except Exception as exc:
                logger.error("Optimization failed for %s: %s", symbol, exc)
                continue

            best_params = study.best_params
            best_value = study.best_value
            # Do not mutate a strategy while the trading loop may be evaluating
            # it. The next tick atomically drains and applies this result.
            with self._state_lock:
                self._pending_updates[symbol] = dict(best_params)
            self._log_results(symbol, strategy_name, best_params, best_value)
            print(f"[Optimizer] {symbol}: Applied {best_params} (Sharpe={best_value:.3f})")

    @staticmethod
    def _last_n_days(bars: pd.DataFrame, days: int = 180) -> pd.DataFrame:
        if not isinstance(bars.index, pd.DatetimeIndex) or bars.empty:
            return bars
        cutoff = bars.index[-1] - pd.Timedelta(days=days)
        return bars.loc[bars.index >= cutoff]

    def _objective(self, trial: Any, strategy: Any, bars: pd.DataFrame) -> float:
        name = strategy.name if hasattr(strategy, "name") else strategy.__class__.__name__
        params: dict[str, Any] = {}

        if name == "mean_reversion" or "MeanReversion" in strategy.__class__.__name__:
            params["std_multiplier"] = trial.suggest_float("std_multiplier", 1.0, 3.0)
            params["lookback"] = trial.suggest_int("lookback", 10, 40)
        elif name == "momentum_breakout" or "Momentum" in strategy.__class__.__name__:
            params["lookback"] = trial.suggest_int("lookback", 10, 40)
            params["volume_multiplier"] = trial.suggest_float("volume_multiplier", 1.0, 3.0)
        elif name == "trend_following" or "Trend" in strategy.__class__.__name__:
            # Trend following needs slow_period + 2 bars minimum for the walk-forward test.
            # With 180-day @ 4h bars we get ~292 bars, so slow_period max is ~280.
            # Use lower bounds to ensure we have enough data for optimization.
            params["fast_period"] = trial.suggest_int("fast_period", 10, 60)
            params["slow_period"] = trial.suggest_int("slow_period", 120, 280)
            if params["fast_period"] >= params["slow_period"]:
                return -1.0
        else:
            return 0.0

        temp = self._clone_with_params(strategy, params)
        return self._walk_forward_sharpe(temp, bars)

    @staticmethod
    def _clone_with_params(strategy: Any, params: dict[str, Any]) -> Any:
        """Build a temporary strategy instance with trial parameters."""
        cls = type(strategy)
        symbol = getattr(strategy, "symbol", "OPT")
        name = getattr(strategy, "name", cls.__name__)

        if name == "mean_reversion" or "MeanReversion" in cls.__name__:
            return cls(
                symbol=symbol,
                std_multiplier=params["std_multiplier"],
                lookback=params["lookback"],
            )
        if name == "momentum_breakout" or "Momentum" in cls.__name__:
            return cls(
                symbol=symbol,
                lookback=params["lookback"],
                volume_multiplier=params["volume_multiplier"],
            )
        if name == "trend_following" or "Trend" in cls.__name__:
            return cls(
                symbol=symbol,
                fast_period=params["fast_period"],
                slow_period=params["slow_period"],
                allow_short=getattr(strategy, "allow_short", True),
            )
        return strategy

    def _walk_forward_sharpe(self, strategy: Any, bars: pd.DataFrame) -> float:
        train_size = int(len(bars) * 0.7)
        if train_size < getattr(strategy, "min_bars_required", 50):
            return 0.0

        returns: list[float] = []
        current_position = 0
        entry_price = 0.0

        for i in range(1, len(bars) - train_size):
            current_idx = train_size + i
            window = bars.iloc[:current_idx]
            if len(window) < getattr(strategy, "min_bars_required", 0):
                continue

            close_price = float(window.iloc[-1]["close"])

            if current_position == 0:
                signal = strategy.evaluate_entry(window)
                if signal == "long":
                    current_position = 1
                    entry_price = close_price
                elif signal == "short":
                    current_position = -1
                    entry_price = close_price
            else:
                side = "long" if current_position == 1 else "short"
                if strategy.evaluate_exit(window, side):
                    if current_position == 1:
                        ret = (close_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - close_price) / entry_price
                    returns.append(ret)
                    current_position = 0

        if not returns:
            return 0.0

        ret_array = np.array(returns)
        std_ret = float(np.std(ret_array))
        if std_ret > 0:
            return float(np.mean(ret_array) / std_ret * np.sqrt(252))
        return 0.0

    def _apply_best_params(self, strategy: Any, best_params: dict) -> None:
        name = getattr(strategy, "name", strategy.__class__.__name__)
        if name == "mean_reversion" or "MeanReversion" in strategy.__class__.__name__:
            if "std_multiplier" in best_params:
                strategy.std_multiplier = best_params["std_multiplier"]
            if "lookback" in best_params:
                strategy.lookback = best_params["lookback"]
                strategy.min_bars_required = best_params["lookback"] + 2
        elif name == "momentum_breakout" or "Momentum" in strategy.__class__.__name__:
            if "lookback" in best_params:
                strategy.lookback = best_params["lookback"]
                strategy.min_bars_required = best_params["lookback"] + 2
            if "volume_multiplier" in best_params:
                strategy.volume_multiplier = best_params["volume_multiplier"]
        elif name == "trend_following" or "Trend" in strategy.__class__.__name__:
            if "fast_period" in best_params:
                strategy.fast_period = best_params["fast_period"]
            if "slow_period" in best_params:
                strategy.slow_period = best_params["slow_period"]
                strategy.min_bars_required = best_params["slow_period"] + 2

        logger.info(
            "Applied optimized params for %s (%s): %s",
            getattr(strategy, "symbol", "?"),
            name,
            best_params,
        )

    def _log_results(
        self,
        symbol: str,
        strategy_name: str,
        best_params: dict,
        best_value: float,
    ) -> None:
        filename = config.runtime_path(config.RUNTIME_DIR, _RESULTS_FILENAME)
        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline="") as csvfile:
            fieldnames = ["timestamp", "symbol", "strategy", "best_sharpe", "best_params"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "best_sharpe": f"{best_value:.4f}",
                    "best_params": str(best_params),
                }
            )
