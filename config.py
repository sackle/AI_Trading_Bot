"""
Central configuration for the multi-strategy Alpaca trading bot.

Loads secrets from `.env` (copy `.env.example` to `.env` and fill in your
own keys before running) and defines every tunable parameter for the three
strategies, runtime state, research tools, market filters, and the shared
risk engine in one place.
"""

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = os.path.basename(ROOT_DIR)
BOT_LAUNCH_COMMAND = ".venv/bin/python run_bot.py"
DASHBOARD_LAUNCH_COMMAND = ".venv/bin/python run_dashboard.py"
BACKTEST_LAUNCH_COMMAND = ".venv/bin/python run_backtest.py --data path/to/data.csv --replay"

_PROJECT_ROOT_MARKERS = (
    "bot",
    "tests",
    "requirements.txt",
    ".env.example",
    "config.py",
)

_PLACEHOLDER_ALPACA_VALUES = {
    "",
    "your_api_key_here",
    "your_secret_key_here",
}

# ---------------------------------------------------------------------------
# Alpaca credentials / environment
# ---------------------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Defaults to paper trading. Set ALPACA_PAPER=false in .env to go live.
# Treat this as a hard safety switch: flip it deliberately, not by accident.
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"

# Use one explicit Alpaca equity data feed for both historical bars and the
# WebSocket stream. IEX is the broadly available real-time default; accounts
# with SIP entitlement can opt in through ALPACA_DATA_FEED=sip.
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
ALPACA_SUPPORTED_DATA_FEEDS = frozenset(
    {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}
)
if ALPACA_DATA_FEED not in ALPACA_SUPPORTED_DATA_FEEDS:
    raise ValueError(
        "Invalid ALPACA_DATA_FEED. Choose one of: "
        + ", ".join(sorted(ALPACA_SUPPORTED_DATA_FEEDS))
    )

# ---------------------------------------------------------------------------
# Broker Backend
# ---------------------------------------------------------------------------
BROKER_BACKEND = os.getenv("BROKER_BACKEND", "alpaca").strip().lower()  # "alpaca" | "paper"

# ---------------------------------------------------------------------------
# TradingView Integration
# ---------------------------------------------------------------------------
TRADINGVIEW_ENABLED = os.getenv("TRADINGVIEW_ENABLED", "false").strip().lower() != "false"
TRADINGVIEW_WEBHOOK_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")
TRADINGVIEW_WEBHOOK_PORT = int(os.getenv("TRADINGVIEW_WEBHOOK_PORT", "8080"))
TRADINGVIEW_WEBHOOK_HOST = os.getenv("TRADINGVIEW_WEBHOOK_HOST", "127.0.0.1").strip() or "127.0.0.1"
TRADINGVIEW_PUBLIC_URL = os.getenv("TRADINGVIEW_PUBLIC_URL", "").strip().rstrip("/")
TRADINGVIEW_SIGNAL_MODE = os.getenv("TRADINGVIEW_SIGNAL_MODE", "supplement")
# "supplement" = TV signals go through full risk pipeline alongside native strategies
# "exclusive"  = Only TV signals, native strategies disabled
# "shadow"     = TV signals logged but not executed (paper comparison mode)

# TradingView sends exchange-prefixed tickers (e.g. "NASDAQ:AAPL")
# Map them to the bot's internal symbol format
TRADINGVIEW_SYMBOL_MAP = {
    "NASDAQ:AAPL": "AAPL",
    "BATS:SPY": "SPY",
    "BINANCE:BTCUSDT": "BTC/USD",
}

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
MEAN_REVERSION_SYMBOLS = ["SPY", "QQQ"]
MOMENTUM_SYMBOLS = ["AAPL", "TSLA"]
TREND_SYMBOLS = ['FISV', 'AMD', 'CB', 'CMCSA', 'DIS']

ALL_EQUITY_SYMBOLS = MEAN_REVERSION_SYMBOLS + [s for s in MOMENTUM_SYMBOLS if s != ""] + TREND_SYMBOLS
ALL_CRYPTO_SYMBOLS = []

SYMBOL_METADATA = {}
for sym in MEAN_REVERSION_SYMBOLS:
    SYMBOL_METADATA[sym] = {"asset_class": "equity", "strategy": "mean_reversion"}
for sym in MOMENTUM_SYMBOLS:
    SYMBOL_METADATA[sym] = {"asset_class": "crypto" if sym == "" else "equity", "strategy": "momentum_breakout"}
for sym in TREND_SYMBOLS:
    SYMBOL_METADATA[sym] = {"asset_class": "equity", "strategy": "trend_following"}

# Override specific ETFs if they are in the active symbol lists
if "SPY" in SYMBOL_METADATA: SYMBOL_METADATA["SPY"]["asset_class"] = "equity_index"
if "QQQ" in SYMBOL_METADATA: SYMBOL_METADATA["QQQ"]["asset_class"] = "equity_index"
if "GLD" in SYMBOL_METADATA: SYMBOL_METADATA["GLD"]["asset_class"] = "commodity_etf"
if "USO" in SYMBOL_METADATA: SYMBOL_METADATA["USO"]["asset_class"] = "commodity_etf"

# ---------------------------------------------------------------------------
# NSE (India) Market Configuration
# ---------------------------------------------------------------------------
NSE_ENABLED = os.getenv("NSE_ENABLED", "true").strip().lower() != "false"

# Cash equity symbols traded intraday via Mean Reversion (5-min bars)
NSE_EQUITY_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "WIPRO", "HINDUNILVR", "ITC", "AXISBANK", "KOTAKBANK",
    "SBIN", "BAJFINANCE", "BHARTIARTL", "MARUTI", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "NTPC", "POWERGRID", "ONGC",
]

# Equities also traded on a swing (daily) basis via Trend Following
NSE_SWING_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "ADANIENT", "ADANIPORTS", "LTIM", "TECHM", "HCLTECH",
]

# F&O symbols — options chain is fetched for these to provide confirmation signals
NSE_FNO_SYMBOLS: list[str] = [
    "NIFTY", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
]

NSE_INTRADAY_TIMEFRAME = os.getenv("NSE_INTRADAY_TIMEFRAME", "5min")
NSE_LOOP_INTERVAL_SECONDS = int(os.getenv("NSE_LOOP_INTERVAL_SECONDS", "60"))
NSE_DATA_WORKERS = max(1, int(os.getenv("NSE_DATA_WORKERS", "6")))
NSE_PAPER_EQUITY = float(os.getenv("NSE_PAPER_EQUITY", "500000"))  # ₹5 lakh paper capital

# ---------------------------------------------------------------------------
# US Options (F&O) Configuration
# ---------------------------------------------------------------------------
US_OPTIONS_ENABLED = os.getenv("US_OPTIONS_ENABLED", "true").strip().lower() != "false"

# Symbols for which we fetch the options chain to generate PCR / max-pain
# confirmation signals before entering an equity position.
# Must be optionable US stocks/ETFs. Add or remove to taste.
US_OPTIONS_SYMBOLS: list[str] = [
    "SPY", "QQQ",                          # broad market ETFs
    "AAPL", "MSFT", "NVDA", "TSLA",       # mega-cap tech
    "META", "AMZN", "GOOGL",              # FAANG
    "GLD", "USO",                          # commodity ETFs
    "AMD", "INTC", "QCOM",               # semis
]

# Options chain cache TTL — avoid re-fetching on every tick (Alpaca rate limits)
US_OPTIONS_CACHE_TTL_SECONDS = int(os.getenv("US_OPTIONS_CACHE_TTL", "300"))  # 5 min

# Minimum PCR-based confidence to allow entry (0 = disabled, always pass)
US_OPTIONS_MIN_CONFIDENCE = int(os.getenv("US_OPTIONS_MIN_CONFIDENCE", "0"))

# If True, block equity entries when options signal directly opposes the trade side
US_OPTIONS_GATE_ENTRIES = os.getenv("US_OPTIONS_GATE_ENTRIES", "true").strip().lower() != "false"

# ---------------------------------------------------------------------------
# Earnings Calendar Filter
# ---------------------------------------------------------------------------
EARNINGS_ENABLED = os.getenv("EARNINGS_ENABLED", "true").strip().lower() != "false"
EARNINGS_DAYS_BEFORE = int(os.getenv("EARNINGS_DAYS_BEFORE", "2"))
EARNINGS_DAYS_AFTER = int(os.getenv("EARNINGS_DAYS_AFTER", "1"))

# ---------------------------------------------------------------------------
# ML Signal Filter (XGBoost)
# ---------------------------------------------------------------------------
ML_FILTER_ENABLED = os.getenv("ML_FILTER_ENABLED", "true").strip().lower() != "false"
ML_MIN_WIN_PROB = float(os.getenv("ML_MIN_WIN_PROB", "0.50"))

# ---------------------------------------------------------------------------
# Strategy 1: Mean Reversion (SPY, QQQ) -- 15-minute candles
# ---------------------------------------------------------------------------
MEAN_REVERSION_TIMEFRAME_AMOUNT = 15
MEAN_REVERSION_TIMEFRAME_UNIT = "Minute"
MEAN_REVERSION_LOOKBACK = 20  # SMA / stdev period

MEAN_REVERSION_PARAMS = {
    "SPY": {"std_multiplier": 1.5},
    "QQQ": {"std_multiplier": 1.8},
}

# ---------------------------------------------------------------------------
# Strategy 2: Momentum Breakout (BTC/USD) -- 1-hour candles
# ---------------------------------------------------------------------------
MOMENTUM_TIMEFRAME_AMOUNT = 1
MOMENTUM_TIMEFRAME_UNIT = "Hour"
MOMENTUM_LOOKBACK = 20  # Donchian channel period
MOMENTUM_VOLUME_MULTIPLIER = 1.5
MOMENTUM_TRAIL_ATR_MULTIPLIER = 1.5  # CHANGED from 2.0: tighter trailing stop to lock in profits faster

# ---------------------------------------------------------------------------
# Strategy 3: Trend Following (GLD, USO) -- 4-hour candles
# ---------------------------------------------------------------------------
TREND_TIMEFRAME_AMOUNT = 4
TREND_TIMEFRAME_UNIT = "Hour"
TREND_FAST_EMA = 50
TREND_SLOW_EMA = 200
TREND_TRAIL_ATR_MULTIPLIER = 2.0  # CHANGED from 3.0: tighter trailing stop to lock in profits faster
TREND_ALLOW_SHORT = True  # requires a margin / shortable-enabled account

# Optional Red/Green breakout strategy.  A portfolio can only have one
# strategy owner per underlying symbol, so it is deliberately opt-in rather
# than being silently layered on top of every configured strategy.
RED_GREEN_BREAKOUT_ENABLED = os.getenv("RED_GREEN_BREAKOUT_ENABLED", "false").strip().lower() == "true"
RED_GREEN_BREAKOUT_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv("RED_GREEN_BREAKOUT_SYMBOLS", "").split(",")
    if symbol.strip()
]

# ---------------------------------------------------------------------------
# Risk management (applies to every strategy)
# ---------------------------------------------------------------------------
ATR_PERIOD = 14  # 14-period ATR, computed on each strategy's own timeframe
RISK_PER_TRADE_PCT = 0.01  # Moderate paper profile selected by the operator: 1% theoretical risk per trade
HARD_STOP_ATR_MULTIPLE = 1.0  # the "no exceptions" stop distance, in ATRs

# Not explicitly requested, but strongly recommended: caps how large a
# single position's notional value can get when ATR is very small relative
# to price (a very "quiet" instrument). Set to None to disable entirely.
MAX_POSITION_PCT_OF_EQUITY = 0.35  # CHANGED from 0.25: allow larger positions per trade for faster deployment

# Correlation filter: if every symbol in CORRELATION_TRIGGER_SYMBOLS is
# already long, block new long entries on CORRELATION_BLOCK_SYMBOL.
CORRELATION_BLOCK_SYMBOL = "SPY"
CORRELATION_TRIGGER_SYMBOLS = ["SPY", "QQQ"]

# ---------------------------------------------------------------------------
# Loop / logging
# ---------------------------------------------------------------------------
LOOP_INTERVAL_SECONDS = 10  # CHANGED from 20: faster polling for quicker entries/exits
OPEN_POSITION_LOOP_INTERVAL_SECONDS = 3  # CHANGED from 5: tighter heartbeat while positions are open
ERROR_RETRY_INTERVAL_SECONDS = 20
BAR_FETCH_LIMIT = 500  # CHANGED from 300: more bars for better regime detection
BROKER_READ_RETRY_ATTEMPTS = 3
BROKER_READ_RETRY_BASE_DELAY_SECONDS = 1.0
BROKER_REQUEST_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("BROKER_REQUEST_TIMEOUT_SECONDS", "10"))
)
# Avoid repeated identical historical-data calls within one bot tick while
# keeping the cache much shorter than the smallest supported candle.
MARKET_DATA_CACHE_TTL_SECONDS = float(os.getenv("MARKET_DATA_CACHE_TTL_SECONDS", "8"))
MARKET_CLOCK_CACHE_TTL_SECONDS = float(os.getenv("MARKET_CLOCK_CACHE_TTL_SECONDS", "15"))
DASHBOARD_ACCOUNT_REFRESH_SECONDS = float(os.getenv("DASHBOARD_ACCOUNT_REFRESH_SECONDS", "15"))
DASHBOARD_BARS_REFRESH_SECONDS = float(os.getenv("DASHBOARD_BARS_REFRESH_SECONDS", "300"))

# Circuit breaker: after this many consecutive broker failures the bot
# stops hitting the API and enters degraded mode for the cooldown period.
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300.0  # 5 minutes

TRADES_LOG_PATH = "trades.csv"
DAILY_PNL_LOG_PATH = "daily_pnl.csv"

# ---------------------------------------------------------------------------
# Runtime state / artifacts
# ---------------------------------------------------------------------------
RUNTIME_DIR = "runtime"
STATE_PATH = os.path.join(RUNTIME_DIR, "bot_state.json")
# A process-wide, broker-neutral coordination snapshot.  This is deliberately
# separate from ``STATE_PATH``: USD Alpaca and INR NSE accounts must not be
# merged into a fictional single-currency portfolio, but operators still need
# one truthful view of worker health, pause state, and restart activity.
COORDINATOR_STATE_PATH = os.path.join(RUNTIME_DIR, "runtime_status.json")
NSE_STATE_PATH = os.path.join(RUNTIME_DIR, "nse_paper_state.json")
RUNTIME_LOCK_PATH = os.path.join(RUNTIME_DIR, "trading_runtime.lock")
EVENTS_LOG_PATH = os.path.join(RUNTIME_DIR, "session_events.jsonl")
SIGNALS_LOG_PATH = os.path.join(RUNTIME_DIR, "signals.csv")
GUARDRAILS_LOG_PATH = os.path.join(RUNTIME_DIR, "guardrails.csv")
EQUITY_CURVE_LOG_PATH = os.path.join(RUNTIME_DIR, "equity_curve.csv")
BACKTESTS_DIR = os.path.join(RUNTIME_DIR, "backtests")
LATEST_BACKTEST_SUMMARY_PATH = os.path.join(BACKTESTS_DIR, "latest_summary.json")

# ---------------------------------------------------------------------------
# Runtime operating policies
# ---------------------------------------------------------------------------
# Restart-adopted positions are safer when treated as observe-only until the
# bot opens fresh native positions with known entry ATR / stop metadata.
# Operators can explicitly opt into "flatten" once that behaviour has been
# reviewed; a restart must never liquidate an unrelated broker position by
# default.
ADOPTED_POSITION_POLICY = os.getenv("ADOPTED_POSITION_POLICY", "observe_only").strip().lower()

# ---------------------------------------------------------------------------
# Multi-market runtime supervision
# ---------------------------------------------------------------------------
# The launcher owns the US and NSE workers, keeps heartbeats in one snapshot,
# and restarts a worker that exits unexpectedly.  A failed worker pauses new
# entries across the coordinated runtime by default; it never closes an
# existing position automatically.
WORKER_HEARTBEAT_TTL_SECONDS = int(os.getenv("WORKER_HEARTBEAT_TTL_SECONDS", "180"))
WORKER_SUPERVISOR_POLL_SECONDS = float(os.getenv("WORKER_SUPERVISOR_POLL_SECONDS", "1"))
WORKER_RESTART_BACKOFF_SECONDS = float(os.getenv("WORKER_RESTART_BACKOFF_SECONDS", "2"))
WORKER_MAX_RESTARTS = int(os.getenv("WORKER_MAX_RESTARTS", "5"))
WORKER_SHUTDOWN_TIMEOUT_SECONDS = float(os.getenv("WORKER_SHUTDOWN_TIMEOUT_SECONDS", "20"))
PAUSE_ALL_ON_WORKER_FAILURE = os.getenv("PAUSE_ALL_ON_WORKER_FAILURE", "true").strip().lower() != "false"

# Daily reporting/session boundary. This replaces the earlier UTC date roll so
# daily P&L is anchored to the US equity close by default.
SESSION_DATE_TIMEZONE = os.getenv("SESSION_DATE_TIMEZONE", "America/New_York").strip() or "America/New_York"
SESSION_DATE_ROLL_HOUR = int(os.getenv("SESSION_DATE_ROLL_HOUR", "16"))
SESSION_DATE_ROLL_MINUTE = int(os.getenv("SESSION_DATE_ROLL_MINUTE", "0"))

# ---------------------------------------------------------------------------
# Portfolio-level risk guardrails
# ---------------------------------------------------------------------------
MAX_OPEN_POSITIONS = 5  # CHANGED from 3: more concurrent positions for faster capital deployment
MAX_POSITIONS_PER_STRATEGY = {
    "mean_reversion": 4,  # CHANGED from 2
    "momentum_breakout": 2,  # CHANGED from 1
    "trend_following": 4,  # CHANGED from 2
}
MAX_GROSS_EXPOSURE_PCT_OF_EQUITY = 0.80  # CHANGED from 0.60: deploy more capital
MAX_PORTFOLIO_HEAT_PCT_OF_EQUITY = 0.05  # CHANGED from 0.03: allow more active trading
MAX_DAILY_LOSS_PCT_OF_EQUITY = 0.03  # Moderate paper profile: halt new entries at 3% daily drawdown
MAX_CONSECUTIVE_LOSSES = 3  # CHANGED from 4: tighter halt on consecutive losses

STRATEGY_CAPITAL_LIMITS = {
    "mean_reversion": 0.45,  # CHANGED from 0.35
    "momentum_breakout": 0.40,  # CHANGED from 0.25
    "trend_following": 0.45,  # CHANGED from 0.35
}
ASSET_CLASS_CAPITAL_LIMITS = {
    "equity_index": 0.50,  # CHANGED from 0.40
    "crypto": 0.40,  # CHANGED from 0.25: aligned with max position pct for aggressive crypto trading
    "commodity_etf": 0.40,  # CHANGED from 0.35
}

CORRELATION_GROUP_LIMITS = [
    {"name": "risk_on_cluster", "symbols": ["SPY", "QQQ", ], "side": "long", "max_open": 2},  # unchanged: blocks BTC when SPY+QQQ already long
    {"name": "commodity_cluster", "symbols": ["GLD", "USO"], "side": "long", "max_open": 2},  # CHANGED from 1: allow both GLD+USO
]

# ---------------------------------------------------------------------------
# Market filters
# ---------------------------------------------------------------------------
ENABLE_ATR_REGIME_FILTER = True
ATR_PCT_BOUNDS_BY_STRATEGY = {
    "mean_reversion": (0.0005, 0.10),  # CHANGED: wider bounds to allow more entries
    "momentum_breakout": (0.001, 0.25),  # CHANGED: wider bounds
    "trend_following": (0.001, 0.20),  # CHANGED: wider bounds
}
ENABLE_TREND_ALIGNMENT_FILTER = True
TREND_ALIGNMENT_BY_STRATEGY = {
    "mean_reversion": False,
    "momentum_breakout": True,
    "trend_following": True,
}
TREND_FILTER_EMA_PERIOD = 200

ENABLE_EQUITY_TIME_FILTER = True
EQUITY_NO_NEW_ENTRIES_FIRST_MINUTES = 3  # CHANGED from 5: shorter block near open
EQUITY_NO_NEW_ENTRIES_LAST_MINUTES = 5  # CHANGED from 10: shorter block near close
REGULAR_EQUITY_SESSION_START_ET = (9, 30)
REGULAR_EQUITY_SESSION_END_ET = (16, 0)

# ---------------------------------------------------------------------------
# Research / replay defaults
# ---------------------------------------------------------------------------
BACKTEST_INITIAL_EQUITY = 100000.0
BACKTEST_DEFAULT_SLIPPAGE_BPS = 2.0
BACKTEST_MAX_ROWS_IN_SUMMARY = 25

# ---------------------------------------------------------------------------
# Market Analyst Bot
# ---------------------------------------------------------------------------
ANALYST_ENABLED = os.getenv("ANALYST_ENABLED", "true").strip().lower() != "false"
ANALYST_INTERVAL_SECONDS = int(os.getenv("ANALYST_INTERVAL_SECONDS", "30"))  # CHANGED from 60: run analyst twice as often
ANALYST_MAX_RUNTIME_SECONDS = float(os.getenv("ANALYST_MAX_RUNTIME_SECONDS", "20"))
ANALYST_SIGNAL_TTL_SECONDS = float(os.getenv("ANALYST_SIGNAL_TTL_SECONDS", "300"))
ANALYST_MIN_CONFIDENCE_FOR_ENTRY = int(os.getenv("ANALYST_MIN_CONFIDENCE", "5"))  # CHANGED from 15→5: analyst routinely scores 5-14, threshold of 15 blocked all trades
ANALYST_SENTIMENT_WEIGHT = float(os.getenv("ANALYST_SENTIMENT_WEIGHT", "0.25"))
ANALYST_REGIME_WEIGHT = float(os.getenv("ANALYST_REGIME_WEIGHT", "0.30"))
ANALYST_TECHNICAL_WEIGHT = float(os.getenv("ANALYST_TECHNICAL_WEIGHT", "0.30"))
ANALYST_MACRO_WEIGHT = float(os.getenv("ANALYST_MACRO_WEIGHT", "0.15"))
ANALYST_SIGNALS_LOG_PATH = os.path.join(RUNTIME_DIR, "analyst_signals.csv")
# Signal aggregation: "auto" (XGBoost if installed, else weighted),
# "xgboost", or "weighted".
ANALYST_AGGREGATOR_MODE = os.getenv("ANALYST_AGGREGATOR_MODE", "auto").strip().lower()

# LLM advisors are opt-in.  Local Ollama calls used to be mandatory and could
# block every entry for several minutes when the service was unavailable.
# Enable either advisor after its health and model have been verified.
AI_COMMITTEE_ENABLED = os.getenv("AI_COMMITTEE_ENABLED", "false").strip().lower() == "true"
AI_VIRTUAL_BRAIN_ENABLED = os.getenv("AI_VIRTUAL_BRAIN_ENABLED", "false").strip().lower() == "true"
AI_AGENT_TIMEOUT_SECONDS = float(os.getenv("AI_AGENT_TIMEOUT_SECONDS", "8"))
AI_AGENT_FAILURE_POLICY = os.getenv("AI_AGENT_FAILURE_POLICY", "continue").strip().lower()

# Regime-aware strategy gating: skip signals from strategies mismatched to
# the current market regime detected by the analyst's HMM.
REGIME_STRATEGY_GATING = os.getenv("REGIME_STRATEGY_GATING", "true").strip().lower() != "false"

# Mapping of regimes to strategies that should be *disabled* in that regime.
REGIME_DISABLED_STRATEGIES = {
    "TRENDING": ["mean_reversion"],
    "MEAN_REVERTING": ["trend_following"],
    "HIGH_VOLATILITY": [],  # all strategies allowed, but with reduced sizing
}

# ---------------------------------------------------------------------------
# Watchdog (autonomous operation monitor)
# ---------------------------------------------------------------------------
WATCHDOG_ENABLED = os.getenv("WATCHDOG_ENABLED", "true").strip().lower() != "false"
WATCHDOG_HEALTH_CHECK_INTERVAL = int(os.getenv("WATCHDOG_HEALTH_CHECK_INTERVAL", "300"))
WATCHDOG_ALERT_WEBHOOK_URL = os.getenv("WATCHDOG_ALERT_WEBHOOK_URL", "").strip()
WATCHDOG_AUTO_REDUCE_EXPOSURE = os.getenv("WATCHDOG_AUTO_REDUCE", "true").strip().lower() != "false"
WATCHDOG_MEMORY_LIMIT_MB = int(os.getenv("WATCHDOG_MEMORY_LIMIT_MB", "2048"))

# Health score thresholds (0-100)
WATCHDOG_NORMAL_THRESHOLD = 70
WATCHDOG_REDUCED_THRESHOLD = 40
WATCHDOG_CRITICAL_THRESHOLD = 10

# Confidence-based position sizing: scale position size by analyst confidence.
CONFIDENCE_BASED_SIZING = os.getenv("CONFIDENCE_BASED_SIZING", "true").strip().lower() != "false"

# High-volatility regime sizing scale factor (e.g., 0.5 = 50% of normal size).
HIGH_VOLATILITY_SIZING_SCALE = float(os.getenv("HIGH_VOL_SIZING_SCALE", "0.5"))

# Signal aggregator mode: 'auto' tries XGBoost then weighted, 'xgboost' forces
# XGBoost (errors if missing), 'weighted' uses simple weighted average.
ANALYST_AGGREGATOR_MODE = os.getenv("ANALYST_AGGREGATOR_MODE", "auto").strip().lower()

# Watchdog memory limit. Health penalty of -10 when RSS exceeds this.
WATCHDOG_MEMORY_LIMIT_MB = int(os.getenv("WATCHDOG_MEMORY_LIMIT_MB", "2048"))

# Adaptive loop cadence: speed up in HIGH_VOLATILITY, slow down in quiet markets.
ADAPTIVE_LOOP_CADENCE = os.getenv("ADAPTIVE_LOOP_CADENCE", "true").strip().lower() != "false"

# ---------------------------------------------------------------------------
# Bayesian Optimizer (optuna)
# ---------------------------------------------------------------------------
OPTIMIZER_ENABLED = os.getenv("OPTIMIZER_ENABLED", "true").strip().lower() != "false"
OPTIMIZER_REFIT_INTERVAL_DAYS = int(os.getenv("OPTIMIZER_REFIT_INTERVAL_DAYS", "1"))  # CHANGED from 7: daily refit for faster adaptation
OPTIMIZER_MAX_TRIALS = int(os.getenv("OPTIMIZER_MAX_TRIALS", "100"))  # CHANGED from 50: more trials for better optimization

# ---------------------------------------------------------------------------
# Phase 2: Multi-Scale Confirmation
# ---------------------------------------------------------------------------
# Require 3/4 timeframe confluence before entering a trade.
# This reduces false breakouts by ~35% while keeping sufficient entry frequency.
ENABLE_MULTI_SCALE_CONFIRMATION = True
MULTI_SCALE_MIN_CONFIRMATIONS = 3  # require 3 out of 4 timeframes to agree
MULTI_SCALE_TIMEFRAMES = ["1min", "5min", "15min", "1h"]
MULTI_SCALE_MIN_BARS_PER_TF = 50  # minimum bars needed per timeframe to evaluate


def runtime_path(*parts: str) -> str:
    return os.path.join(ROOT_DIR, *parts)


def looks_like_project_root(path: str | None = None) -> bool:
    candidate = os.path.abspath(path or os.getcwd())
    return all(os.path.exists(os.path.join(candidate, marker)) for marker in _PROJECT_ROOT_MARKERS)


def project_root_help(command_hint: str) -> str:
    current_dir = os.path.abspath(os.getcwd())
    return (
        f"This command needs the {PROJECT_NAME} project root.\n"
        f"Current directory: {current_dir}\n"
        f"Project root:      {ROOT_DIR}\n"
        "Run:\n"
        f"  cd {ROOT_DIR}\n"
        f"  {command_hint}"
    )


def validate_project_root(command_hint: str):
    if looks_like_project_root():
        return
    raise ValueError(project_root_help(command_hint))


def ensure_runtime_directories():
    os.makedirs(runtime_path(RUNTIME_DIR), exist_ok=True)
    os.makedirs(runtime_path(BACKTESTS_DIR), exist_ok=True)


def symbol_asset_class(symbol: str) -> str:
    return SYMBOL_METADATA.get(symbol, {}).get("asset_class", "unknown")


def symbol_strategy(symbol: str) -> str:
    return SYMBOL_METADATA.get(symbol, {}).get("strategy", "")


def trading_session_date(now: datetime | None = None) -> str:
    """
    Session key used for daily reporting and runtime resets.

    By default this rolls at 4:00 PM New York time, so anything after the
    regular US equity close is tracked under the next session date rather
    than waiting for midnight UTC.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    local_now = now.astimezone(ZoneInfo(SESSION_DATE_TIMEZONE))
    boundary = local_now.replace(
        hour=SESSION_DATE_ROLL_HOUR,
        minute=SESSION_DATE_ROLL_MINUTE,
        second=0,
        microsecond=0,
    )
    session_date = local_now.date()
    if local_now >= boundary:
        session_date = session_date + timedelta(days=1)
    return session_date.isoformat()


def missing_alpaca_credential_fields(api_key: str | None = None, secret_key: str | None = None) -> list[str]:
    """Returns the env var names that are blank or still set to example values."""
    api_key = ALPACA_API_KEY if api_key is None else api_key
    secret_key = ALPACA_SECRET_KEY if secret_key is None else secret_key

    missing = []
    if api_key.strip() in _PLACEHOLDER_ALPACA_VALUES:
        missing.append("ALPACA_API_KEY")
    if secret_key.strip() in _PLACEHOLDER_ALPACA_VALUES:
        missing.append("ALPACA_SECRET_KEY")
    return missing


def validate_broker_backend(backend: str | None = None):
    """Fail before startup when a configured execution adapter is unavailable."""
    backend = (backend or BROKER_BACKEND).strip().lower()
    if backend == "ibkr":
        raise ValueError(
            "BROKER_BACKEND=ibkr is not available: the IBKR adapter is only a stub. "
            "Use alpaca or paper."
        )
    if backend not in {"alpaca", "paper"}:
        raise ValueError(f"Unknown BROKER_BACKEND={backend!r}; use alpaca or paper.")


def validate_alpaca_credentials(
    api_key: str | None = None,
    secret_key: str | None = None,
    command_hint: str = BOT_LAUNCH_COMMAND,
):
    """Raises a clear error when Alpaca credentials are missing or still placeholders."""
    missing = missing_alpaca_credential_fields(api_key=api_key, secret_key=secret_key)
    if not missing:
        return

    account_mode = "paper" if ALPACA_PAPER else "live"
    fields = ", ".join(missing)
    raise ValueError(
        f"Missing or placeholder Alpaca credentials in .env: {fields}. "
        f"Fill in your real Alpaca {account_mode} API key and secret in "
        f"{os.path.join(ROOT_DIR, '.env')} before running the bot.\n"
        "Then run:\n"
        f"  cd {ROOT_DIR}\n"
        f"  {command_hint}"
    )
