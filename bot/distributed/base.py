"""Base class for all distributed services in the Aegis trading system.

Each service:
1. Connects to the message bus
2. Registers its event handlers
3. Runs in its own process (launched via `python -m bot.distributed.<name>`)
4. Publishes heartbeats to bot.heartbeat.<service_name>
5. Handles graceful shutdown on SIGTERM/SIGINT

Usage:
    from bot.distributed.base import BaseService

    class MyService(BaseService):
        name = "my_service"

        def register_handlers(self):
            self.bus.subscribe("bars.*", self._on_bar)

        def _on_bar(self, channel, msg):
            symbol = channel.replace("bars.", "")
            # process bar...
            self.bus.publish("signal." + symbol, {"side": "long", "reason": "entered"})

    if __name__ == "__main__":
        MyService().run()
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from bot.distributed import MessageBus

logger = logging.getLogger("aegis.service")


class BaseService(ABC):
    """Abstract base for all Aegis distributed services."""

    name: str = "base_service"
    heartbeat_interval: int = 5  # seconds

    def __init__(self, bus: MessageBus | None = None):
        self.bus = bus or MessageBus()
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @abstractmethod
    def register_handlers(self) -> None:
        """Override: register bus.subscribe() calls for channels this service cares about."""

    def on_start(self) -> None:
        """Optional hook: called once after handlers are registered."""

    def on_tick(self) -> None:
        """Optional hook: called every heartbeat interval (for periodic work)."""

    def on_stop(self) -> None:
        """Optional hook: called during graceful shutdown."""

    def publish_heartbeat(self) -> None:
        self.bus.publish(
            f"bot.heartbeat.{self.name}",
            {
                "service": self.name,
                "timestamp": time.time(),
                "status": "running" if self._running else "stopped",
            },
        )

    def _heartbeat_loop(self) -> None:
        while self._running:
            self.publish_heartbeat()
            self.on_tick()
            self._stop_event.wait(self.heartbeat_interval)

    def _handle_signal(self, signum, frame) -> None:
        logger.info("[%s] Received signal %d, shutting down...", self.name, signum)
        self._running = False
        self._stop_event.set()

    def run(self) -> None:
        logger.info("[%s] Starting service", self.name)
        self._running = True
        self.register_handlers()
        self.bus.start()
        self.on_start()

        # Install signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Start heartbeat + periodic work loop
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.info("[%s] Running. Press Ctrl+C to stop.", self.name)
        self._stop_event.wait()  # block until shutdown

        # Graceful shutdown
        self.on_stop()
        self.bus.stop()
        logger.info("[%s] Stopped.", self.name)
