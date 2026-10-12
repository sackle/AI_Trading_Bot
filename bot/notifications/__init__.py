"""
Telegram Notification Service for the Aegis Trading Bot.

Sends real-time alerts to a configured Telegram chat whenever:
  - A position is opened (US or NSE)
  - A position is closed with P&L
  - An entry is blocked by options gate / guardrails
  - The bot starts up or shuts down
  - A watchdog warning fires
  - Daily summary at market close

Configuration (in .env):
    TELEGRAM_BOT_TOKEN=7123456789:AAFxxxxx...
    TELEGRAM_CHAT_ID=123456789
    TELEGRAM_ENABLED=true
    TELEGRAM_MIN_PNL_ALERT=0   # Only alert on closes with |P&L| >= this (0 = all)

Usage:
    from bot.notifications.telegram_bot import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send_position_opened(symbol, side, qty, entry, stop, risk, strategy, equity, market="US")
    notifier.send_position_closed(symbol, side, exit_price, pnl, strategy, equity, market="US")
    notifier.send_blocked(symbol, side, reason, strategy, market="US")
    notifier.send_startup(mode, equity)
    notifier.send_daily_summary(wins, losses, pnl, equity)
    notifier.send_watchdog_alert(message, level)
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from queue import Queue, Empty

import config

logger = logging.getLogger("aegis.telegram")

# ---------------------------------------------------------------------------
# Optional requests import — graceful degradation
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
    logger.warning("requests library not installed — Telegram notifications disabled")

# Emoji constants
_EMOJI = {
    "open_long":   "🟢",
    "open_short":  "🔴",
    "close_win":   "✅",
    "close_loss":  "❌",
    "blocked":     "⚠️",
    "startup":     "🚀",
    "shutdown":    "🛑",
    "watchdog":    "🐕",
    "summary":     "📊",
    "nse":         "🇮🇳",
    "us":          "🇺🇸",
    "options":     "📉",
}

_DIVIDER = "─" * 24


class TelegramNotifier:
    """
    Thread-safe Telegram notifier.

    All sends are dispatched via an internal background queue so
    the trading loop is never blocked by network latency.
    Includes a listener thread for interactive /commands.
    """

    def __init__(self, command_handler=None):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID",   "").strip()
        self.enabled = (
            os.getenv("TELEGRAM_ENABLED", "true").strip().lower() != "false"
            and bool(self.token)
            and bool(self.chat_id)
            and _REQUESTS_OK
        )
        self.min_pnl = float(os.getenv("TELEGRAM_MIN_PNL_ALERT", "0"))
        self._send_url    = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self._get_url     = f"https://api.telegram.org/bot{self.token}/getUpdates"
        self._queue: Queue[str] = Queue()
        self._command_handler = command_handler
        self._last_update_id = 0

        if self.enabled:
            self._worker = threading.Thread(target=self._dispatch_loop, daemon=True, name="telegram-notifier")
            self._worker.start()
            
            if self._command_handler:
                self._listener = threading.Thread(target=self._polling_loop, daemon=True, name="telegram-listener")
                self._listener.start()
                logger.info("Telegram interactive command listener active.")
                
            logger.info("Telegram notifier active (chat_id=%s)", self.chat_id)
        else:
            if self.token:
                logger.warning("Telegram notifier disabled — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env")
            else:
                logger.info("Telegram notifier not configured — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env")

    def _polling_loop(self):
        """Background loop that polls Telegram for new messages."""
        import time
        while True:
            try:
                params = {"timeout": 10, "offset": self._last_update_id + 1}
                resp = _requests.get(self._get_url, params=params, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        for update in data.get("result", []):
                            update_id = update.get("update_id", 0)
                            self._last_update_id = max(self._last_update_id, update_id)
                            
                            msg = update.get("message")
                            if msg and "text" in msg:
                                chat_id = str(msg["chat"]["id"])
                                if chat_id == self.chat_id:  # Only allow commands from the authorized user
                                    text = msg["text"].strip()
                                    if text.startswith("/"):
                                        response = self._command_handler(text)
                                        if response:
                                            self._send_now(response)
            except Exception as exc:
                logger.debug("Telegram polling error: %s", exc)
            time.sleep(2)

    # ------------------------------------------------------------------
    # Public notification methods
    # ------------------------------------------------------------------
    def send_position_opened(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        stop_price: float,
        risk_amount: float,
        strategy: str,
        equity: float,
        market: str = "US",
    ):
        flag  = _EMOJI["nse"] if market == "NSE" else _EMOJI["us"]
        arrow = _EMOJI["open_long"] if side == "long" else _EMOJI["open_short"]
        currency = "₹" if market == "NSE" else "$"
        msg = (
            f"{arrow} *OPENED {side.upper()}*  `{symbol}`\n"
            f"Entry:    {currency}{entry_price:,.2f}  |  Qty: {qty:g}\n"
            f"Stop:     {currency}{stop_price:,.2f}  |  Risk: {currency}{risk_amount:,.2f}\n"
            f"Strategy: `{strategy}`\n"
            f"{_DIVIDER}\n"
            f"{flag} {market} {'Paper' if config.ALPACA_PAPER else 'Live'}  |  Equity: {currency}{equity:,.2f}"
        )
        self._enqueue(msg)

    def send_position_closed(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        pnl: float,
        strategy: str,
        equity: float,
        market: str = "US",
        reason: str = "",
    ):
        if abs(pnl) < self.min_pnl:
            return
        flag     = _EMOJI["nse"] if market == "NSE" else _EMOJI["us"]
        result   = _EMOJI["close_win"] if pnl >= 0 else _EMOJI["close_loss"]
        currency = "₹" if market == "NSE" else "$"
        pnl_str  = f"+{currency}{pnl:,.2f}" if pnl >= 0 else f"-{currency}{abs(pnl):,.2f}"
        msg = (
            f"{result} *CLOSED {side.upper()}*  `{symbol}`\n"
            f"Exit:     {currency}{exit_price:,.2f}\n"
            f"P&L:      {pnl_str} {result}\n"
            f"Strategy: `{strategy}`"
            + (f"\nReason:   {reason}" if reason else "") +
            f"\n{_DIVIDER}\n"
            f"{flag} {market} Equity: {currency}{equity:,.2f}"
        )
        self._enqueue(msg)

    def send_blocked(
        self,
        symbol: str,
        side: str,
        reason: str,
        strategy: str,
        market: str = "US",
    ):
        flag = _EMOJI["nse"] if market == "NSE" else _EMOJI["us"]
        msg = (
            f"{_EMOJI['blocked']} *BLOCKED*  `{symbol}`\n"
            f"Side:     {side}\n"
            f"Strategy: `{strategy}`\n"
            f"Reason:   {reason[:200]}\n"
            f"{_DIVIDER}\n"
            f"{flag} {market}"
        )
        self._enqueue(msg)

    def send_startup(self, mode: str, equity: float, nse_enabled: bool = False):
        nse_tag = "  🇮🇳 NSE paper loop active" if nse_enabled else ""
        msg = (
            f"{_EMOJI['startup']} *Bot Started* — {mode} mode\n"
            f"Equity: ${equity:,.2f}{nse_tag}\n"
            f"{_DIVIDER}\n"
            f"_Aegis Trading Bot v3 — {datetime.now().strftime('%Y-%m-%d %H:%M')} IST_"
        )
        self._enqueue(msg)

    def send_shutdown(self, reason: str = "KeyboardInterrupt"):
        msg = f"{_EMOJI['shutdown']} *Bot Stopped*\nReason: {reason}"
        self._send_now(msg)   # blocking — process may be exiting

    def send_daily_summary(
        self,
        wins: int,
        losses: int,
        daily_pnl: float,
        equity: float,
        market: str = "US",
    ):
        flag     = _EMOJI["nse"] if market == "NSE" else _EMOJI["us"]
        currency = "₹" if market == "NSE" else "$"
        pnl_str  = f"+{currency}{daily_pnl:,.2f}" if daily_pnl >= 0 else f"-{currency}{abs(daily_pnl):,.2f}"
        wr = round(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        msg = (
            f"{_EMOJI['summary']} *Daily Summary* — {flag} {market}\n"
            f"Wins:   {wins}  |  Losses: {losses}  |  WR: {wr}%\n"
            f"P&L:    {pnl_str}\n"
            f"Equity: {currency}{equity:,.2f}\n"
            f"{_DIVIDER}\n"
            f"_{datetime.now().strftime('%Y-%m-%d')}_"
        )
        self._enqueue(msg)

    def send_watchdog_alert(self, message: str, level: str = "warn"):
        icons = {"warn": "⚠️", "critical": "🚨", "info": "ℹ️"}
        icon = icons.get(level, "⚠️")
        msg = f"{icon} *Watchdog {level.upper()}*\n{message}"
        self._enqueue(msg)

    def send_options_gate(self, symbol: str, side: str, pcr: float, direction: str, max_pain: float):
        msg = (
            f"{_EMOJI['options']} *Options Gate Blocked*  `{symbol}`\n"
            f"Intended: {side}  |  Options: {direction}\n"
            f"PCR: {pcr:.2f}  |  Max Pain: ${max_pain:,.1f}"
        )
        self._enqueue(msg)

    def test(self) -> bool:
        """Send a test message. Returns True if successful."""
        return self._send_now("🧪 *Aegis Bot* — Telegram connection test successful!")

    # ------------------------------------------------------------------
    # Internal queue + dispatch
    # ------------------------------------------------------------------
    def _enqueue(self, text: str):
        if self.enabled:
            self._queue.put(text)

    def _dispatch_loop(self):
        """Background thread — drains the queue and sends messages."""
        while True:
            try:
                text = self._queue.get(timeout=5)
                self._send_now(text)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as exc:
                logger.debug("Telegram dispatch error: %s", exc)

    def _send_now(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = _requests.post(
                self._send_url,
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:
            logger.debug("Telegram send failed: %s", exc)
            return False
