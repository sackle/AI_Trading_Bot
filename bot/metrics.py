#!/usr/bin/env python3
"""Prometheus metrics integration for Phase 3 monitoring"""

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsRegistry:
    """Simple in-memory metrics registry for Prometheus-style metrics."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._metrics = defaultdict(dict)
                    cls._instance._counters = defaultdict(int)
                    cls._instance._gauges = {}
                    cls._instance._histograms = defaultdict(list)
        return cls._instance
    
    def counter(self, name: str, **labels: str) -> "CounterMetric":
        """Create or get a counter metric."""
        return CounterMetric(name, labels, self)
    
    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge value."""
        key = f"{name}:{tuple(sorted(labels.items()))}"
        self._gauges[key] = value
    
    def histogram(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram bucket."""
        key = f"{name}:{tuple(sorted(labels.items()))}"
        self._histograms[key].append(value)
        # Keep only last 1000 values
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]
    
    def record(self, name: str, value: float, **labels: str) -> None:
        """Record a metric (abbreviated interface)."""
        key = f"{name}:{tuple(sorted(labels.items()))}"
        self._gauges[key] = value
    
    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        """Increment a counter."""
        self._counters[f"{name}:{tuple(sorted(labels.items()))}"] += amount
    
    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = [
            "# HELP aegis_trading_bot_metrics Aegis Trading Bot metrics",
            "# TYPE aegis_trading_bot_metrics gauge",
        ]
        
        # Export gauges
        for key, value in self._gauges.items():
            parts = key.split(":")
            name = parts[0]
            labels_str = ""
            if len(parts) > 1 and parts[1]:
                labels_str = "{" + parts[1] + "}"
            lines.append(f'{name}{labels_str} {value}')
        
        # Export counters
        for key, value in self._counters.items():
            parts = key.split(":")
            name = parts[0]
            labels_str = ""
            if len(parts) > 1 and parts[1]:
                labels_str = "{" + parts[1] + "}"
            lines.append(f'{name}{labels_str} {value}')
        
        return "\n".join(lines)


class CounterMetric:
    """Counter metric implementation."""
    
    def __init__(self, name: str, labels: dict[str, str], registry: MetricsRegistry):
        self._name = name
        self._labels = labels
        self._registry = registry
    
    def inc(self, amount: int = 1) -> None:
        """Increment counter by amount."""
        label_key = tuple(sorted(self._labels.items()))
        key = f"{self._name}:{label_key}"
        self._registry._counters[key] += amount
    
    def __iadd__(self, amount: int) -> "CounterMetric":
        self.inc(amount)
        return self


# Global registry
metrics = MetricsRegistry()


class TradingBotMetrics:
    """Convenience class for common trading bot metrics."""
    
    def __init__(self):
        self.orders = metrics.counter("aegis_orders_total", service="orchestrator")
        self.trades = metrics.counter("aegis_trades_total", service="strategy")
        self.errors = metrics.counter("aegis_errors_total", service="bot")
        self.connectivity = metrics.counter("aegis_connectivity_status", service="bus")
    
    def record_order(self, order_type: str, status: str) -> None:
        self.orders.inc()
        metrics.increment("aegis_orders_by_type", 1, type=order_type, status=status)
    
    def record_trade(self, symbol: str, pnl: float) -> None:
        self.trades.inc()
        metrics.record("aegis_pnl_cumulative", pnl, symbol=symbol)
    
    def record_error(self, service: str, error_type: str) -> None:
        self.errors.inc()
        metrics.increment("aegis_errors_by_service", 1, service=service, error_type=error_type)
    
    def set_uptime(self, uptime_seconds: float) -> None:
        metrics.gauge("aegis_uptime_seconds", uptime_seconds)


# Convenience instance
bot_metrics = TradingBotMetrics()