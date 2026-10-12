"""Distributed message bus for the Aegis trading system.

Provides publish/subscribe messaging over Redis for all inter-service
communication. Each service publishes events and subscribes to channels
it cares about. This replaces the monolithic single-thread main loop.

Channels:
- bars.{symbol}      — market data bars (15min, 1h, 4h, 1min)
- signal.{symbol}    — strategy entry/exit signals
- position.{symbol}  — position open/close/update events
- risk.guardrails    — portfolio-level guardrail decisions
- analyst.signal     — market analyst regime/sentiment signals
- bot.control        — start/stop/pause commands (control plane)
- bot.heartbeat      — health/status pings from each service

Usage:
    bus = MessageBus()
    bus.publish("bars.AAPL", {"open": ..., "high": ..., ...})
    bus.subscribe("signal.*", handler_func)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

import redis

import config

logger = logging.getLogger("aegis.bus")

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PREFIX = "aegis"


class MessageBus:
    """Thin async wrapper around Redis pub/sub.

    Uses a single Redis client for publishing and a background
    subscriber thread for consuming. Subscribers register pattern
    handlers that are called with (channel: str, message: dict).
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        prefix: str = REDIS_PREFIX,
    ):
        self._prefix = prefix
        self._pubsub = None
        self._handlers: list[tuple[str, Callable]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._r: redis.Redis | None = None

        # Try to connect; fall back to fakeredis for local dev / testing.
        try:
            r = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
            r.ping()
            self._r = r
            logger.info("MessageBus connected to Redis at %s:%d", host, port)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis not available (%s); falling back to fakeredis", exc)
            try:
                import fakeredis
                self._r = fakeredis.FakeAsyncRedis()
                logger.info("MessageBus using fakeredis (in-process mode)")
            except ImportError:
                raise RuntimeError(
                    "Neither real Redis nor fakeredis is available. "
                    "Install redis: pip install redis"
                ) from exc

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(self, channel: str, message: dict[str, Any]) -> None:
        """Publish a JSON-encoded message to a channel."""
        full_channel = f"{self._prefix}:{channel}"
        data = json.dumps(message, default=str)
        self._r.publish(full_channel, data)

    def publish_raw(self, channel: str, raw_message: str) -> None:
        full_channel = f"{self._prefix}:{channel}"
        self._r.publish(full_channel, raw_message)

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------
    def subscribe(self, pattern: str, handler: Callable[[str, dict], None]) -> None:
        """Register a handler for a channel-pattern (e.g. 'signal.*')."""
        self._handlers.append((pattern, handler))

    def _start_subscriber(self) -> None:
        if self._running:
            return
        self._running = True
        self._pubsub = self._r.pubsub(ignore_subscribe_messages=True)
        for pattern, _ in self._handlers:
            self._pubsub.psubscribe(f"{self._prefix}:{pattern}")
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info("Subscriber thread started with %d handler(s)", len(self._handlers))

    def _listen(self) -> None:
        assert self._pubsub is not None
        try:
            for message in self._pubsub.listen():
                if not self._running:
                    break
                if message["type"] != "pmessage":
                    continue
                channel = message["channel"].decode().replace(f"{self._prefix}:", "")
                raw = message["data"].decode() if isinstance(message["data"], bytes) else message["data"]
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {"raw": raw}
                for pattern, handler in self._handlers:
                    # Simple wildcard match: pattern "signal.*" matches "signal.AAPL"
                    if self._matches(channel, pattern):
                        try:
                            handler(channel, payload)
                        except Exception as exc:
                            logger.exception("Handler error for %s: %s", channel, exc)
        except (redis.ConnectionError, redis.TimeoutError, ValueError):
            pass  # Expected during shutdown

    @staticmethod
    def _matches(channel: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return channel.startswith(prefix + ".")
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(channel, pattern)
        return channel == pattern

    def start(self) -> None:
        self._start_subscriber()

    def stop(self) -> None:
        self._running = False
        if self._pubsub:
            self._pubsub.close()
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Convenience: wait for a message on a channel (for testing)
    # ------------------------------------------------------------------
    def wait_for(self, channel: str, timeout: float = 5.0) -> dict | None:
        """Blocking wait for the next message on a specific channel."""
        import queue
        q: queue.Queue = queue.Queue()

        def handler(ch: str, msg: dict):
            if ch == channel:
                q.put(msg)

        self.subscribe(channel, handler)
        self._start_subscriber()
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None
