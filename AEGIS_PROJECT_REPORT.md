
<div align="center">

# AEGIS QUANTITATIVE TRADING SYSTEM
## Technical Project Report

**Classification:** Internal Technical Documentation  
**Version:** 3.0  
**Audit Date:** October 2026  
**Status:** Active — Paper Trading Mode  

---

*Prepared by the Aegis Engineering Team*  
*This document is for research and educational reference only.*  
*Trading involves risk. This system does not guarantee profit.*

</div>

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Overview & Objectives](#2-project-overview--objectives)
3. [Technology Stack & Programming Languages](#3-technology-stack--programming-languages)
4. [System Architecture](#4-system-architecture)
5. [Bot Inventory — All Agents & Services](#5-bot-inventory--all-agents--services)
6. [How the Bots Coordinate with Each Other](#6-how-the-bots-coordinate-with-each-other)
7. [How Each Bot Makes Trades — End-to-End Trade Lifecycle](#7-how-each-bot-makes-trades--end-to-end-trade-lifecycle)
8. [Machine Learning Models](#8-machine-learning-models)
9. [Algorithms & Mathematical Methods](#9-algorithms--mathematical-methods)
10. [Trading Strategies — Mechanics](#10-trading-strategies--mechanics)
11. [Risk Management Engine](#11-risk-management-engine)
12. [Telegram Notification System — Full Technical Breakdown](#12-telegram-notification-system--full-technical-breakdown)
13. [Data Layer & Market Data Sources](#13-data-layer--market-data-sources)
14. [Dashboard & Monitoring Infrastructure](#14-dashboard--monitoring-infrastructure)
15. [Backtesting & Research Pipeline](#15-backtesting--research-pipeline)
16. [Deployment & Operations](#16-deployment--operations)
17. [Testing & Quality Assurance](#17-testing--quality-assurance)
18. [Security & Safety Controls](#18-security--safety-controls)
19. [Known Gaps & Limitations](#19-known-gaps--limitations)
20. [Glossary](#20-glossary)

---

## 1. Executive Summary

Aegis (branded **Aegis Quantum / Aegis Trading Bot v3**) is a Python-based, multi-strategy quantitative trading system built for research, paper-trading demonstrations, and educational exploration of algorithmic trading. It is the result of iterative development across three major versions and integrates production-grade software engineering practices with institutional-style trading architecture.

The system runs in **paper-trading mode** against Alpaca's paper brokerage and IEX market data. It orchestrates multiple intelligent agents — including an ML-powered XGBoost signal filter, a Gaussian Hidden Markov Model for market regime detection, a Reinforcement Learning exit agent, a local Large Language Model investment committee, a GARCH-based volatility forecaster, and a Bayesian hyperparameter optimizer — all coordinating in parallel threads within a supervised monolithic runtime.

**Key metrics at October 2026 audit:**

| Metric | Value |
|---|---|
| Automated tests | **152 passed** in 2.94 seconds |
| Active U.S. trading symbols | **9** (SPY, QQQ, AAPL, TSLA, FISV, AMD, CB, CMCSA, DIS) |
| Strategy families | **4** (Mean Reversion, Momentum Breakout, Trend Following, Red/Green) |
| Active ML models | **3** (XGBoost classifier, Gaussian HMM, PPO RL agent) |
| Runtime mode | PAPER (no real money) |
| Doctor check result | **0 failures, 3 warnings** |
| Dashboard health | HTTP 200 |

> **Disclaimer:** This system is not a validated investment product. No profit is guaranteed. The accuracy claims from any ML model are research figures, not certified performance. All trading is conducted in paper mode with simulated capital only.

---

## 2. Project Overview & Objectives

### 2.1 Mission Statement

To build a complete, production-quality algorithmic trading platform that demonstrates end-to-end integration of modern quantitative finance techniques, machine learning, and intelligent agent systems — entirely in Python, running autonomously 24/7 in paper mode.

### 2.2 Core Objectives

| Objective | Implementation |
|---|---|
| Multi-strategy execution | 4 strategy classes across 9 symbols |
| Intelligent signal filtering | XGBoost binary classifier gates every trade |
| Market regime awareness | 3-state Gaussian HMM continuously refit every 10 minutes |
| Adaptive risk sizing | Kelly Criterion + volatility + analyst confidence scaling |
| Human-style trade review | Local LLM (qwen2.5) committee debates each trade |
| Autonomous operation | Supervised worker threads with heartbeat + restart-backoff |
| Real-time monitoring | Telegram bot + Flask dashboard |
| Research pipeline | Optuna optimizer + backtesting + global data harvester |

### 2.3 Version History

| Version | Key Additions |
|---|---|
| v1 | Single strategy, basic Alpaca execution, simple stop-loss |
| v2 | Portfolio guardrails, persistent state, restart recovery, market filters, multi-strategy |
| **v3 (current)** | Market Analyst Bot, HMM regime detection, XGBoost filter, LLM committee, Watchdog, GARCH, Optuna, PPO exit agent, Telegram commands, NSE loop, TradingView bridge |

---

## 3. Technology Stack & Programming Languages

### 3.1 Primary Language

| Language | Version | Usage |
|---|---|---|
| **Python** | **3.13** (tested) / 3.11+ (minimum) | 100% of application code — bot engine, ML models, dashboard, agents, API clients, tests |

Python 3.13 is used for its improved per-interpreter GIL, faster startup, and full compatibility with all ML and data science libraries in the stack.

### 3.2 Full Library Dependency Table

#### Core Trading & Brokerage

| Library | Version | Purpose |
|---|---|---|
| `alpaca-py` | **0.43.5** | Alpaca broker REST API + WebSocket streaming. Handles order placement, account info, historical bar fetching, options chain, and live market data stream |
| `websocket-client` | 1.9.0 | Low-level WebSocket connection for Alpaca real-time stream |
| `websockets` | 16.1.1 | Async WebSocket support |
| `sseclient-py` | 1.9.0 | Server-sent events for live data feeds |
| `upstox-python-sdk` | 2.28.0 | Upstox Indian broker SDK (listed; no active import found at audit) |
| `curl_cffi` | 0.15.0 | HTTP client with TLS fingerprint bypass (used by yfinance internally) |

#### Machine Learning & Statistics

| Library | Version | Purpose |
|---|---|---|
| `xgboost` | **3.3.0** | Gradient-boosted decision tree classifier — 7-feature binary trade filter + analyst aggregator |
| `hmmlearn` | **0.3.3** | Hidden Markov Models — 3-state Gaussian HMM for market regime detection |
| `scikit-learn` | **1.9.0** | ML utilities: preprocessing, cross-validation, statistical arbitrage z-score |
| `scipy` | **1.18.0** | GARCH-style volatility model, statistical tests |
| `optuna` | **4.9.0** | Bayesian hyperparameter optimization (TPE algorithm, 50 trials, walk-forward Sharpe) |
| `vaderSentiment` | **3.3.2** | Rule-based sentiment scoring for news headlines |
| `joblib` | 1.5.3 | Parallel execution and ML model persistence |
| `stable-baselines3` | dev env | PPO Reinforcement Learning exit agent (not pinned in requirements.txt) |
| `gymnasium` | dev env | RL training environment for the PPO exit agent |

#### Data & Analytics

| Library | Version | Purpose |
|---|---|---|
| `pandas` | **3.0.5** | Primary data structure for OHLCV bars, signals, backtests, P&L |
| `numpy` | **2.5.1** | Vectorized numerical computing: ATR, rolling stats, indicator math |
| `yfinance` | **1.5.2** | Yahoo Finance: global OHLCV, macro series (VIX/TNX/IRX), earnings, NSE fallback |
| `beautifulsoup4` | 4.15.0 | HTML/RSS news parsing for sentiment pipeline |

#### Web Framework & Dashboard

| Library | Version | Purpose |
|---|---|---|
| `Flask` | **3.1.3** | Operations dashboard web application |
| `waitress` | **3.0.2** | Production-grade pure-Python WSGI server |
| `Jinja2` | 3.1.6 | HTML templating engine for dashboard |
| `Werkzeug` | 3.1.8 | WSGI utilities |

#### Database & Persistence

| Library | Version | Purpose |
|---|---|---|
| `SQLAlchemy` | 2.0.51 | ORM for SQLite trading memory |
| `alembic` | 1.18.5 | Database schema migrations |
| `peewee` | 4.2.6 | Lightweight ORM for TradingMemory |
| `redis` | 8.1.0 | Redis client for experimental distributed message bus |
| `fakeredis` | 2.37.0 | In-memory Redis mock for tests |

#### Configuration & Utilities

| Library | Version | Purpose |
|---|---|---|
| `python-dotenv` | 1.2.2 | Loads `.env` file into environment variables |
| `pydantic` | 2.13.4 | Data validation and typed settings |
| `PyYAML` | 6.0.3 | YAML configuration parsing |
| `colorlog` | 6.12.0 | Color-coded terminal log levels |
| `rich` | 15.0.0 | Rich terminal output and progress bars |
| `tqdm` | 4.70.0 | Progress bars for training/download loops |
| `requests` | 2.34.2 | HTTP client — Telegram API, news feeds, Ollama |
| `msgpack` | 1.2.1 | Efficient binary serialization |
| `protobuf` | 7.35.1 | Protocol Buffers for Alpaca data stream |
| `pytest` | **9.1.1** | Test suite runner (152 tests) |

### 3.3 Auxiliary Technologies

| Technology | Role |
|---|---|
| **SQLite** | Embedded database for TradingMemory (trades, decisions, regime history) |
| **JSON / JSONL** | Runtime state persistence and streaming event logs |
| **CSV** | Trade logs, signal history, equity curve, guardrail records |
| **Ollama** | Local LLM inference server — serves `qwen2.5` at `http://127.0.0.1:11434` |
| **qwen2.5** | Open-source LLM for InvestmentCommittee and VirtualHumanBrain |
| **Redis** | In-memory data store for experimental distributed architecture |
| **Pine Script** | TradingView strategy language — 4 `.pine` files mirror Python strategies |
| **macOS launchd** | Service management and auto-restart via `setup_launchd.sh` |
| **Linux systemd** | Service management via `aegis.service.template` |
| **Telegram Bot API** | REST API at `api.telegram.org` — outbound alerts + inbound commands |

---

## 4. System Architecture

### 4.1 Architectural Pattern

Aegis uses a **Supervised Monolithic Compatibility Runtime** — a single Python process with multiple coordinated worker threads. The `bot/distributed/` microservice architecture exists in code but is experimental and not active.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            run_bot.py                                    │
│                                                                          │
│  ProcessLock  →  RuntimeCoordinator  →  RuntimeSupervisor               │
│                                                  │                       │
│                        ┌─────────────────────────┴──────────────────┐   │
│                        ▼                                             ▼   │
│                  US TradingBot                          NSE TradingBot   │
│                  (9 symbols)                            (optional)       │
│                        │                                                 │
│     ┌──────────────────┼────────────────────────────┐                   │
│     │                  │   Parallel Threads          │                   │
│     │  ┌───────────────▼────────────────────────┐   │                   │
│     │  │ Market Analyst Bot (every 5 min)        │   │                   │
│     │  │  • RegimeDetector (HMM)                 │   │                   │
│     │  │  • SentimentAnalyzer (VADER)            │   │                   │
│     │  │  • TechnicalScanner                     │   │                   │
│     │  │  • MacroOverlay (VIX/TNX/IRX)           │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     │  ┌────────────────────────────────────────┐   │                   │
│     │  │ StrategyOptimizer (Optuna, every 7d)   │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     │  ┌────────────────────────────────────────┐   │                   │
│     │  │ TradingView Webhook (daemon thread)     │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     │  ┌────────────────────────────────────────┐   │                   │
│     │  │ Telegram Notifier + Listener threads   │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     │  ┌────────────────────────────────────────┐   │                   │
│     │  │ NewsTerminal (RSS poller, daemon)       │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     │                                               │                   │
│     │           Per-bar Trading Loop (_tick)         │                   │
│     │  ┌────────────────────────────────────────┐   │                   │
│     │  │ Fetch OHLCV bars (Alpaca REST)         │   │                   │
│     │  │ Evaluate strategies (signal)           │   │                   │
│     │  │ Apply entry gates (ML, regime, risk)   │   │                   │
│     │  │ Submit Alpaca paper orders             │   │                   │
│     │  │ Manage stops / exits                   │   │                   │
│     │  │ Persist state, log events              │   │                   │
│     │  └────────────────────────────────────────┘   │                   │
│     └───────────────────────────────────────────────┘                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 Shared State Layer                                │   │
│  │  RuntimeStateStore (JSON) | Portfolio | TradingMemory (SQLite)  │   │
│  │  CSV logs | JSONL events | RuntimeCoordinator flags              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                │                                         │
│  ┌─────────────────────────────▼────────────────────────────────────┐   │
│  │        Flask Dashboard (Waitress, localhost:5012)                 │   │
│  │  /api/snapshot | /api/runtime | /api/bars/<symbol> | /healthz    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
        ▲                   ▲                    ▲
 Alpaca REST/Stream   Telegram Bot API   TradingView Webhook
 IEX Market Data       (bidirectional)    (supplemental alerts)
```

### 4.2 Startup Sequence

```
1.  run_bot.py → acquire ProcessLock (prevent duplicate instances)
2.  Build RuntimeCoordinator (shared pause/resume flags)
3.  Build RuntimeSupervisor (manages worker threads + heartbeats)
4.  TradingBot.__init__():
      a. Init ML Signal Filter (XGBoost) — FIRST, before threads (macOS requirement)
      b. Create AlpacaBroker
      c. Create Portfolio, RiskManager, StressTestEngine, GARCHForecaster
      d. Create Guardian, DriftMonitor, Watchdog
      e. Build all strategy objects via build_strategies()
      f. Load RuntimeStateStore → restore positions and state
      g. Create DeskManager + launch NewsTerminal thread
      h. Init InvestmentCommittee (Ollama LLM, if enabled)
      i. Init VirtualHumanBrain (Ollama LLM, if enabled)
      j. Launch Alpaca WebSocket Streamer thread
      k. Init MarketAnalystBot (HMM + VADER + scanner + macro)
      l. Init Watchdog
      m. Init StrategyOptimizer (Optuna)
      n. Launch TradingView Webhook daemon thread (if enabled)
      o. Init TelegramNotifier + launch Notifier and Listener threads
      p. Init EarningsCalendar + prefetch all 9 symbols
      q. Init USOptionsScanner
5.  _startup_sync() → connect Alpaca, reconcile portfolio (up to 5 retries with backoff)
6.  Send Telegram 🚀 startup notification
7.  Enter run_forever() main loop
```

---

## 5. Bot Inventory — All Agents & Services

Aegis contains **20 distinct agents, bots, and services**. Below is the complete inventory.

| # | Agent / Service | Class Name | File | Active? | Purpose |
|---|---|---|---|---|---|
| 1 | **US Trading Bot** | `TradingBot` | `bot/main.py` | ✅ Yes | Core U.S. execution engine — fetches bars, evaluates strategies, applies gates, submits Alpaca paper orders |
| 2 | **NSE Trading Bot** | `NSETradingBot` | `bot/nse/nse_loop.py` | ⚙️ Configurable | Indian equity paper-trading loop (disabled by default: `NSE_ENABLED=false`) |
| 3 | **Market Analyst Bot** | `MarketAnalystBot` | `bot/analyst/analyst.py` | ✅ Yes | Aggregates sentiment + regime + technical + macro into composite signal (every 5 min) |
| 4 | **Regime Detector** | `RegimeDetector` | `bot/analyst/regime.py` | ✅ Yes | 3-state Gaussian HMM — TRENDING / MEAN_REVERTING / HIGH_VOLATILITY |
| 5 | **ML Signal Filter** | `MLSignalFilter` | `bot/ml_filter.py` | ✅ Yes | XGBoost binary classifier — gates every trade on 7-feature win probability |
| 6 | **Investment Committee** | `InvestmentCommittee` | `bot/agents/committee.py` | ⚙️ Configurable | 3-role LLM debate (Bull → Bear → Fund Manager) via local Ollama |
| 7 | **Virtual Human Brain** | `VirtualHumanBrain` | `bot/agents/virtual_brain.py` | ⚙️ Configurable | Reads trading playbook and provides discretionary entry review via Ollama |
| 8 | **Watchdog** | `Watchdog` | `bot/watchdog.py` | ✅ Yes | Monitors API health score, failures, drawdown; blocks entries when health critical |
| 9 | **Guardian** | `Guardian` | `bot/guardian.py` | ✅ Yes | Risk controller: emergency halt, exposure scaling, recovery thresholds |
| 10 | **Strategy Optimizer** | `StrategyOptimizer` | `bot/optimizer.py` | ✅ Yes | Bayesian hyperparameter search via Optuna (50 trials, walk-forward Sharpe, every 7 days) |
| 11 | **RL Exit Agent** | `RLExitAgent` | `bot/rl_agent.py` | ✅ Yes | PPO RL model recommending exit timing per open position |
| 12 | **Telegram Notifier** | `TelegramNotifier` | `bot/notifications/__init__.py` | ✅ Yes | Two-way Telegram: outbound alerts + inbound slash commands |
| 13 | **News Terminal** | `NewsTerminal` | `bot/agents/news_terminal.py` | ✅ Yes | Polls Yahoo Finance / Google News RSS; writes headlines to `consensus.md` |
| 14 | **Desk Manager** | `DeskManager` | `bot/agents/desk_manager.py` | ✅ Yes | Manages news consensus shared state read by the analyst |
| 15 | **Post-Mortem Analyzer** | `PostMortemAnalyzer` | `bot/agents/post_mortem.py` | ✅ Yes | Reviews losing trades, proposes `[SHADOW]` rules to the trading playbook |
| 16 | **Shadow Trader** | `ShadowTrader` | `bot/shadow_trader.py` | ✅ Yes | Simulates candidate rule performance without submitting real orders |
| 17 | **US Options Scanner** | `USOptionsScanner` | `bot/us_options/options_scanner.py` | ✅ Yes | Reads options chain — PCR, max-pain, unusual activity — blocks conflicting entries |
| 18 | **TradingView Webhook** | `webhook_server.py` | `bot/tradingview/` | ⚙️ Configurable | Receives inbound TradingView alerts → supplemental trade signals |
| 19 | **Drift Monitor** | `DriftMonitor` | `bot/drift_monitor.py` | ✅ Yes | Detects concept drift in ML model predictions |
| 20 | **GARCH Forecaster** | `VolatilityForecaster` | `bot/garch_model.py` | ✅ Yes | GARCH-style annualized volatility forecast — used in committee and position sizing |

---

## 6. How the Bots Coordinate with Each Other

All bots operate in the same Python process and coordinate through **shared in-memory objects, thread-safe state, and flag-based signaling**. There is no network RPC between primary bots.

### 6.1 Coordination Hub — RuntimeCoordinator

```
┌─────────────────────────────────────────────────────────────┐
│                   RuntimeCoordinator                         │
│  Single source of truth for all pause/resume/health flags   │
│                                                             │
│  .entries_allowed()     → TradingBot checks before each bar │
│  .set_operator_pause()  → Telegram /pause writes here       │
│  .update_worker_health()→ Supervisor reads worker heartbeats│
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
              ▼                               ▼
      RuntimeSupervisor                US TradingBot
      (thread manager)                 (reads flags)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
US Worker Thread    NSE Worker Thread
  (heartbeat)         (heartbeat)
```

### 6.2 Agent-to-Agent Signal Flow

| Producer | Signal | Consumer | Mechanism |
|---|---|---|---|
| `RegimeDetector` | `RegimeResult` (state + confidence) | `MarketAnalystBot` | Direct method call |
| `SentimentAnalyzer` | Polarity score (−1 to +1) | `MarketAnalystBot` | Direct method call |
| `TechnicalScanner` | Technical signal score | `MarketAnalystBot` | Direct method call |
| `MacroOverlay` | VIX/TNX/IRX composite | `MarketAnalystBot` | Direct method call |
| `MarketAnalystBot` | `MarketSignal` per symbol | `TradingBot._analyst_signals` | Background thread writes dict; main thread reads with lock |
| `MLSignalFilter` | `win_probability` (0–1) | `RiskManager.size_position()` | Return value → Kelly multiplier |
| `MLSignalFilter` | Approve / block boolean | `TradingBot._process_symbol()` | Gate in trade entry logic |
| `InvestmentCommittee` | `(approved: bool, transcript)` | `TradingBot._process_symbol()` | Direct call; `False` blocks the trade |
| `VirtualHumanBrain` | Advisory verdict | `TradingBot` | Direct call; fail-open (continue policy) |
| `Watchdog` | `health_score`, `should_block_new_entries` | `TradingBot._tick()` | Shared object property read each tick |
| `Watchdog` | `watchdog_multiplier` (0.0–1.0) | `RiskManager.size_position()` | Argument to reduce sizing |
| `Guardian` | Risk state / halt flags | `TradingBot` | Direct property read |
| `StrategyOptimizer` | New hyperparameters | `TradingBot.strategies` | `apply_pending_updates()` at top of each tick |
| `RLExitAgent` | Exit recommendation | `TradingBot._check_exits()` | Direct call per open position per tick |
| `USOptionsScanner` | PCR, max-pain, direction | `TradingBot._process_symbol()` | Returns `should_block` flag + Telegram alert |
| `EarningsCalendar` | `is_blackout: bool` | `TradingBot._process_symbol()` | Return value per symbol |
| `NewsTerminal` | Headlines → `consensus.md` | `MarketAnalystBot` via `DeskManager` | Shared `DeskManager` reference + file write/read |
| `PostMortemAnalyzer` | Proposed playbook rule | `trading_playbook.md` | File append |
| `TelegramNotifier` | Slash command text | `TradingBot.handle_telegram_command()` | Callback reference injected at init |
| `TradingView webhook` | Alert signal dict | `TradingBot._process_tradingview_signals()` | Thread-safe `drain_signals()` queue |

### 6.3 Thread Safety Controls

| Resource | Protection |
|---|---|
| `_analyst_signals` | `threading.Lock` (`_analyst_lock`) |
| `_analyst_running` flag | Same lock |
| Telegram send queue | `queue.Queue` (thread-safe FIFO) |
| TradingView signal queue | Thread-safe queue in `signal_bridge.py` |
| `RuntimeStateStore` | Single-writer JSON pattern |
| SQLite TradingMemory | peewee's SQLite thread-safety |

### 6.4 US ↔ NSE Coordination

When both workers run, they share `RuntimeCoordinator` (pause/resume) and `TelegramNotifier` (both send to same chat with 🇺🇸/🇮🇳 flags), but maintain **isolated** portfolio state, positions, and risk limits.

---

## 7. How Each Bot Makes Trades — End-to-End Trade Lifecycle

### 7.1 Per-Bar Execution Cycle (`_tick()`)

Every N seconds (regime-adaptive), the main loop calls `_tick()`. Complete flow:

```
STEP 1:  Apply pending Optuna optimizer results
          └─ StrategyOptimizer.apply_pending_updates(strategies)

STEP 2:  Reconcile portfolio with broker
          └─ Check externally closed/modified positions

STEP 3:  Roll day if needed
          └─ Reset daily P&L, consecutive losses, day_start_equity

STEP 4:  Fetch account equity
          └─ broker.get_equity() → Watchdog.record_pnl_snapshot()

STEP 5:  Refresh entry gate
          └─ Check daily loss limit, consecutive losses, manual pause

STEP 6:  Fetch OHLCV bars for all 9 symbols
          └─ _collect_strategy_bars()
             → broker.get_bars(symbol, timeframe, limit=200) per group
             → Updates _market_data_snapshot

STEP 7:  Run Market Analyst Bot (throttled to every 5 min)
          └─ _maybe_run_analyst(bars) → spawns background thread
             → RegimeDetector.detect() per symbol (HMM or rule-based)
             → SentimentAnalyzer.score() (VADER on headlines)
             → TechnicalScanner.scan() per symbol
             → MacroOverlay.get_score() (VIX/TNX/IRX)
             → SignalAggregator.aggregate() (weighted 0.25/0.30/0.30/0.15)
             → Saves MarketSignal dict to _analyst_signals

STEP 8:  Bayesian optimizer (throttled to every 7 days)
          └─ StrategyOptimizer.maybe_optimize(strategies, broker)

STEP 9:  Watchdog health gate
          └─ If health score critical → new_entries_allowed = False

STEP 10: Per-symbol processing loop
          └─ For each of 9 symbols → _process_symbol(symbol, spec, bars)
             (see Section 7.2 for detail)

STEP 11: Process TradingView webhook signals
          └─ drain_signals() → evaluate each inbound alert

STEP 12: Log daily P&L running total

STEP 13: Persist RuntimeState to JSON
```

### 7.2 Per-Symbol Processing (`_process_symbol`) — 24-Step Decision Gate

```
── ENTRY GATES ─────────────────────────────────────────────────

1.  CHECK: new_entries_allowed == True
    └─ If False → SKIP ENTRY

2.  GET ANALYST SIGNAL
    └─ signal = _analyst_signals.get(symbol)
    └─ regime = signal.regime | confidence = signal.confidence_score

3.  EVALUATE STRATEGY
    └─ strategy.generate_signal(bars) → Signal(side=long/short/none)

4.  IF NO SIGNAL → check exits on existing positions → DONE

── IF ENTRY SIGNAL ─────────────────────────────────────────────

5.  REGIME GATE
    └─ Mean Reversion blocked in TRENDING regime → SKIP

6.  EARNINGS BLACKOUT
    └─ EarningsCalendar.is_blackout(symbol) (2d before / 1d after) → SKIP

7.  MARKET FILTERS
    └─ Opening/closing time filters, volume checks, price sanity

8.  US OPTIONS GATE
    └─ USOptionsScanner.get_signal(symbol)
    └─ PCR disagreement → Telegram options-gate alert → SKIP

9.  PORTFOLIO GUARDRAILS (12 checks)
    └─ Max 5 positions | max per strategy | correlation limit
    └─ Gross exposure ≤ 80% | heat ≤ 5% | strategy limit | asset limit
    └─ Daily loss ≤ 3% | consecutive losses < 3
    └─ Breach → log guardrail + Telegram ⚠️ → SKIP

10. ML SIGNAL FILTER (XGBoost)
    └─ features = [side, ATR, analyst_confidence, sentiment,
                   regime_score, technical_score, macro_score]
    └─ win_prob = MLSignalFilter.predict(features)
    └─ win_prob < 0.70 → SKIP (fail-open: 1.0 if model missing)

11. POSITION SIZING (Kelly + ATR)
    └─ base_qty = (equity × 1%) / ATR
    └─ × Kelly multiplier (from win_prob)
    └─ × regime scale (×0.5 if HIGH_VOLATILITY)
    └─ × confidence scale (× analyst_confidence/100)
    └─ × watchdog multiplier (0.5 degraded, 0.0 critical)
    └─ Capped at max_position_pct × equity
    └─ stop_price = entry ± 1×ATR

12. GARCH VOLATILITY FORECAST
    └─ VolatilityForecaster.forecast(bars) → annualized vol %

13. STRESS TEST REPORT
    └─ StressTestEngine.run(portfolio) → crash/drawdown scenario text

14. INVESTMENT COMMITTEE (Ollama LLM, if enabled)
    └─ Bull Analyst → thesis
    └─ Bear Risk Manager → rebuttal (with stress test data)
    └─ Fund Manager → "APPROVE" or "REJECT"
    └─ REJECT → SKIP trade

15. VIRTUAL HUMAN BRAIN (Ollama LLM, if enabled)
    └─ Reads trading_playbook.md → advisory verdict
    └─ Fail-open: does not block if LLM offline

── SUBMIT ORDER ────────────────────────────────────────────────

16. broker.submit_order(symbol, qty, side, stop_price)
    └─ Alpaca Paper API call

17. Confirm fill → Update Portfolio → persist state

18. TelegramNotifier.send_position_opened(...)

19. log_event() → JSONL | log_signal() → CSV

── EXIT MONITORING (every tick for open positions) ──────────────

20. RLExitAgent.should_exit([pnl_pct, atr, bars_held, 0, 0])

21. Strategy deterministic exit logic
    (e.g. Mean Reversion: price returns to SMA(20))

22. Hard stop-loss check (price ≤ stop_price for long)

23. Trailing stop ratchet
    └─ RiskManager.update_trailing_stop(extreme_price, atr, multiple)

24. ON CLOSE:
    └─ broker.close_position() → log trades.csv + session_events.jsonl
    └─ PostMortemAnalyzer: if loss → propose playbook rule
    └─ TelegramNotifier.send_position_closed(symbol, pnl, ...)
    └─ ML Trainer: retrain if ≥ 10 labeled pairs available
```

---

## 8. Machine Learning Models

### 8.1 XGBoost Trade Signal Filter (`MLSignalFilter`)

| Attribute | Detail |
|---|---|
| **Library** | `xgboost 3.3.0` |
| **Model type** | Binary classification (`binary:logistic`) |
| **Task** | Predict win probability of a proposed trade |
| **Input features (7)** | `side` (1=long, 0=short), `atr`, `analyst_confidence`, `sentiment_score`, `regime_score`, `technical_score`, `macro_score` |
| **Output** | `win_probability` ∈ [0, 1] |
| **Boosting rounds** | 50 |
| **Artifact** | `xgboost_model.json` (57,207 bytes) |
| **Threshold** | 0.70 (`MINIMUM_WIN_PROBABILITY`) |
| **Fail-open** | Returns 1.0 when model unavailable |
| **Retraining** | Automatic when ≥ 10 matched closed-trade pairs in session events |

**Kelly Criterion integration:**

```python
kelly_f = (b * p - q) / b       # p = win_prob, q = 1-p, b = avg_win/avg_loss
half_kelly = clamp(kelly_f/2 * 10, 0.25, 2.0)
final_qty = base_qty * half_kelly
```

### 8.2 Gaussian HMM — Regime Detector (`RegimeDetector`)

| Attribute | Detail |
|---|---|
| **Library** | `hmmlearn 0.3.3` |
| **Model type** | `GaussianHMM` (diagonal covariance) |
| **States** | 3: TRENDING · MEAN_REVERTING · HIGH_VOLATILITY |
| **Iterations** | `n_iter=75`, `tol=0.05`, `random_state=42` |
| **Refit interval** | Every **10 minutes** per symbol |
| **Min data** | 90 bars to fit, 50 valid rows |
| **Features (3)** | `atr_pct` (ATR/Close), `autocorr` (20-bar lag-1), `vol` (20-bar return std) |
| **Normalization** | Z-score before fit and predict |
| **Fallback** | Rule-based: ATR% > 3% → HIGH_VOL; autocorr > 0.3 → TRENDING |

**Effect on trading:**
- TRENDING → suppresses Mean Reversion entries
- HIGH_VOLATILITY → position sizes halved
- MEAN_REVERTING → trading loop runs 50% slower

### 8.3 PPO Reinforcement Learning Exit Agent (`RLExitAgent`)

| Attribute | Detail |
|---|---|
| **Library** | `stable-baselines3` |
| **Algorithm** | Proximal Policy Optimization (PPO) |
| **Observation** | `[pnl_pct, atr, bars_held, 0, 0]` (5D) |
| **Training** | Synthetic random-walk price paths |
| **Usage** | Called per open position per tick; can trigger early exit |
| **Status** | Experimental — not validated on real trade trajectories |

### 8.4 XGBoost Analyst Signal Aggregator

| Attribute | Detail |
|---|---|
| **Mode** | `auto` → XGBoost if artifact exists, else weighted average |
| **Artifact** | `runtime/xgb_aggregator_model.json` (absent at audit → weighted fallback) |
| **Fallback weights** | Sentiment 0.25 · Regime 0.30 · Technical 0.30 · Macro 0.15 |

### 8.5 GARCH-Style Volatility Forecaster (`VolatilityForecaster`)

| Attribute | Detail |
|---|---|
| **Library** | `scipy 1.18.0` |
| **Method** | GARCH(1,1)-style fit on log-return series |
| **Output** | Annualized volatility % |
| **Usage** | Committee context: "GARCH Annualized Volatility Forecast: X.XX%" |

### 8.6 Optuna Bayesian Hyperparameter Optimizer (`StrategyOptimizer`)

| Attribute | Detail |
|---|---|
| **Library** | `optuna 4.9.0` |
| **Algorithm** | Tree-structured Parzen Estimator (TPE) |
| **Trials** | 50 per optimization run |
| **Objective** | Walk-forward Sharpe ratio (70/30 in/out-of-sample split) |
| **Refit interval** | Every 7 days |
| **Application** | `apply_pending_updates()` called at top of every tick |

### 8.7 VADER Sentiment Analyzer

| Attribute | Detail |
|---|---|
| **Library** | `vaderSentiment 3.3.2` |
| **Method** | Lexicon + rule-based polarity |
| **Input** | Yahoo Finance + Google News RSS headlines |
| **Output** | Compound score ∈ [−1, +1] |
| **Weight in aggregator** | 25% |

---

## 9. Algorithms & Mathematical Methods

### 9.1 Technical Indicators

| Indicator | Formula / Method | Used In |
|---|---|---|
| SMA | Rolling mean of close | Mean reversion bands |
| EMA | `α·P + (1-α)·EMA_prev`, α=2/(n+1) | Trend following (50/200 cross) |
| ATR | Wilder smoothing of True Range | Stop-loss, position sizing, regime |
| RSI | Wilder avg gain/loss ratio → 100-100/(1+RS) | Mean reversion entry filter |
| Bollinger Bands | SMA ± k·σ(rolling) | Mean reversion entry/exit |
| MACD | EMA(12)−EMA(26); signal=EMA(9,MACD) | Trend following confirmation |
| Donchian Channels | Max(High,N) / Min(Low,N) | Momentum breakout |
| VWAP | Σ(Price·Volume)/Σ(Volume) | Intraday reference |
| Hurst Exponent (R/S) | Rescaled range on log returns | Regime tendency |
| Volume Profile | Current volume / N-bar avg volume | Breakout confirmation |
| Autocorrelation (lag-1) | Pearson ρ(returns, returns_shift1) | HMM feature + rule fallback |
| Order-Flow Imbalance | (close−open)/(high−low) | Microstructure confirmation |

### 9.2 Position Sizing — Kelly Criterion

$$\text{base\_qty} = \frac{\text{equity} \times 1\%}{\text{ATR}}$$

$$f^* = \frac{b \cdot p - q}{b} \quad \text{(Full Kelly)}$$

$$\text{half\_kelly} = \text{clamp}\!\left(\frac{f^*}{2} \times 10,\ 0.25,\ 2.0\right)$$

$$\text{final\_qty} = \text{base\_qty} \times \text{half\_kelly} \times \text{regime\_scale} \times \text{confidence\_scale} \times \text{watchdog\_mult}$$

### 9.3 Stop-Loss & Trailing Stop

```
Hard stop (long):   stop = entry_price − (1 × ATR)
Hard stop (short):  stop = entry_price + (1 × ATR)

Trailing update (long):
  candidate = extreme_high − (trail_mult × ATR)
  new_stop  = max(current_stop, candidate)   ← only ratchets up

Trailing update (short):
  candidate = extreme_low + (trail_mult × ATR)
  new_stop  = min(current_stop, candidate)   ← only ratchets down
```

The hard 1×ATR stop acts as the floor in the early phase; the trailing stop takes over once sufficient profit has been captured.

### 9.4 HMM Training — Baum-Welch (EM)

The 3-state HMM is trained using the Baum-Welch Expectation-Maximization algorithm on 3 normalized features per bar. At inference time, `predict_proba()` returns posterior state probabilities via the forward-backward algorithm.

### 9.5 Analyst Signal Aggregation

$$\text{composite} = 0.25 S_\text{sent} + 0.30 S_\text{regime} + 0.30 S_\text{tech} + 0.15 S_\text{macro}$$

$$\text{signal} = \begin{cases} \text{BULLISH} & \text{composite} > +0.2 \\ \text{BEARISH} & \text{composite} < -0.2 \\ \text{NEUTRAL} & \text{otherwise} \end{cases}$$

### 9.6 Walk-Forward Sharpe (Optuna Objective)

$$\text{Sharpe} = \frac{\bar{R} - R_f}{\sigma_R} \times \sqrt{252}$$

Evaluated on the 30% out-of-sample split after 50 trials of Bayesian search on the 70% in-sample window.

### 9.7 Statistical Arbitrage Z-Score

$$z_t = \frac{\text{spread}_t - \mu_\text{spread}}{\sigma_\text{spread}}$$

Entry: |z| > threshold; Exit: |z| < exit_threshold (research module).

---

## 10. Trading Strategies — Mechanics

### Strategy 1 — Mean Reversion

**Symbols:** SPY, QQQ | **Timeframe:** 15-minute bars

| Element | Specification |
|---|---|
| Long entry | close < SMA(20) − std_mult×σ **AND** RSI(14) < 35 |
| Short entry | close > SMA(20) + std_mult×σ **AND** RSI(14) > 65 |
| Long exit | close ≥ SMA(20) |
| Short exit | close ≤ SMA(20) |
| std_mult | 1.5 (SPY), 1.8 (QQQ) |
| Regime gate | Suppressed in TRENDING regime |
| Risk | 1×ATR hard stop · 1.5×ATR trail |

### Strategy 2 — Momentum Breakout

**Symbols:** AAPL, TSLA | **Timeframe:** 1-hour bars

| Element | Specification |
|---|---|
| Long entry | close > Donchian_Upper(20) **AND** volume ≥ 1.5× avg_vol(20) |
| Exit | close < Donchian_Lower(20) OR volume dried up |
| Risk | 1×ATR hard stop · 1.5×ATR trail |

### Strategy 3 — EMA Trend Following

**Symbols:** FISV, AMD, CB, CMCSA, DIS | **Timeframe:** 4-hour bars

| Element | Specification |
|---|---|
| Long entry | EMA(50) crosses above EMA(200) + MACD histogram > 0 + volume > 1.2× avg |
| Short entry | Death cross + MACD histogram < 0 (if allow_short=True) |
| Exit | Reverse cross |
| Risk | 1×ATR hard stop · 2×ATR trail |

### Strategy 4 — Red/Green Candle Breakout

**Symbols:** User-configured (empty by default) | **Timeframe:** 5-minute bars

| Element | Specification |
|---|---|
| Pattern | Bar N = red; Bar N+1 = green (close > Bar_N.high); Bar N+2 breaks Bar_{N+1}.high |
| Risk | 1×ATR hard stop · 2×ATR trail |
| Status | Idle — no symbols assigned |

---

## 11. Risk Management Engine

### 11.1 Portfolio-Level Gates (Pre-Sizing)

| Gate | Limit | Breach Effect |
|---|---|---|
| Max open positions | 5 | Entry blocked |
| Max positions per strategy | Configurable | Entry blocked |
| Correlation group limit | ≤ 1 same-direction per cluster | Entry blocked |
| Gross exposure cap | 80% of equity | Entry blocked |
| Portfolio heat cap | 5% of equity | Entry blocked |
| Strategy capital limit | Per-strategy % | Entry blocked |
| Asset class limit | Per class % | Entry blocked |
| Daily loss cap | 3% of day_start_equity | All entries halted until next day |
| Consecutive loss limit | 3 | Entries paused |
| Operator pause | Telegram `/pause` | All entries halted |
| Watchdog health | Score < critical | New entries blocked |

### 11.2 Per-Position Controls

| Control | Implementation |
|---|---|
| Hard stop-loss | entry ± 1×ATR — checked every tick |
| Trailing stop | Ratchets behind extreme_price × trail_multiple |
| ATR-based sizing | Dollar risk constant at 1% equity per instrument |
| Kelly scaling | Size adjusted by win probability from XGBoost |
| Volatility scaling | Size halved in HIGH_VOLATILITY regime |
| Confidence scaling | Size × (analyst_confidence/100) |
| Watchdog scaling | 1.0 (healthy) → 0.5 (degraded) → 0.0 (critical) |

### 11.3 Supervisor-Level Controls

| Control | Setting |
|---|---|
| Worker restart limit | Max 5 with 2s backoff |
| Pause-all-on-failure | Yes — critical worker failure halts all entries |
| Circuit breaker | N consecutive API failures → cooldown, trading suspended |
| Heartbeat TTL | 180 seconds — stale heartbeat triggers restart |

---

## 12. Telegram Notification System — Full Technical Breakdown

### 12.1 Architecture

```
TradingBot.__init__()
    │
    └─► TelegramNotifier(command_handler=self.handle_telegram_command)
            │
            ├─ Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from .env
            ├─ Creates thread-safe Queue() for outbound messages
            │
            ├─ Thread A: "telegram-notifier" (daemon)
            │    └─ _dispatch_loop()
            │       └─ Blocks on Queue.get(timeout=5)
            │       └─ _send_now(text) → POST https://api.telegram.org/
            │
            └─ Thread B: "telegram-listener" (daemon)
                 └─ _polling_loop()
                    └─ GET /getUpdates?timeout=10 every 2 seconds
                    └─ Filter: only process authorized chat_id
                    └─ On slash command → handle_telegram_command(text)
                    └─ Response → _send_now(reply)
```

### 12.2 Why the Trading Loop Never Blocks

Every notification is **fire-and-forget**:

```python
# In TradingBot — on trade open:
self._notifier.send_position_opened(symbol, side, qty, ...)

# Inside send_position_opened():
self._enqueue(msg)          # Queue.put(msg) — returns immediately

# Background "telegram-notifier" thread drains the queue:
requests.post(TELEGRAM_API_URL, json={...}, timeout=10)
```

Network latency to Telegram (typically 100–500ms) is completely absorbed by the background thread.

### 12.3 Inbound Command Flow

```
User types: /pause

Telegram → polling endpoint → _polling_loop() receives update
  Verify: str(msg["chat"]["id"]) == TELEGRAM_CHAT_ID
  Call: handle_telegram_command("/pause")

  In TradingBot.handle_telegram_command():
    self._manual_entry_pause = True
    self.state.new_entries_allowed = False
    self.state.entry_halt_reason = "Manually paused via Telegram."
    self.coordinator.set_operator_pause(True, reason)
    self._persist_state()  ← persisted across restarts

  Reply enqueued: "⏸️ Bot paused. No new entries will be taken."
```

### 12.4 All Outbound Notification Events

| Event | Trigger | Emoji | Message Content |
|---|---|---|---|
| Bot Started | `run_forever()` begins | 🚀 | Mode, equity, NSE status, timestamp |
| Bot Stopped | Shutdown | 🛑 | Stop reason |
| Position Opened | Order filled | 🟢/🔴 | Symbol, side, entry, qty, stop, risk, strategy, equity |
| Position Closed — Win | Exit with profit | ✅ | Symbol, exit price, +P&L, strategy, reason, equity |
| Position Closed — Loss | Exit with loss | ❌ | Symbol, exit price, −P&L, strategy, reason, equity |
| Entry Blocked | Guardrail fired | ⚠️ | Symbol, side, block reason, strategy, market |
| Options Gate Blocked | PCR/max-pain conflict | 📉 | Symbol, PCR, options direction, max pain level |
| Watchdog Alert | Health score drops | 🐕 | Alert message, severity level |
| Daily Summary | Market close / day roll | 📊 | Wins, losses, win rate %, daily P&L, equity |

### 12.5 All Inbound Slash Commands

| Command | What Happens |
|---|---|
| `/status` | Returns mode, equity, daily P&L, open position count, entries allowed |
| `/positions` | Lists all open positions with current unrealized P&L per symbol |
| `/logs` | Shows last 5 closed trades from `trades.csv` with P&L |
| `/pause` | Halts all new entries immediately; persisted to state JSON |
| `/resume` | Re-enables entries; checks coordinator flags first |
| `/close <SYMBOL>` | Immediately closes the named position at current market price |
| `/help` | Lists all available commands |

### 12.6 Security

- Only messages from the exact configured `TELEGRAM_CHAT_ID` are processed
- Bot token stored in `.env` — never committed to version control
- Listener is a background daemon — cannot stall the trading loop

---

## 13. Data Layer & Market Data Sources

### 13.1 Live Market Data (Trading)

| Source | Library | Data | Used For |
|---|---|---|---|
| Alpaca REST API | `alpaca-py` | OHLCV historical bars | Strategy evaluation (primary) |
| Alpaca WebSocket | `alpaca-py` + `MarketStreamer` | Real-time trade/quote stream | Live price feed for 9 US symbols |
| Alpaca Options API | `alpaca-py` | Options chain (PCR, strikes, OI) | USOptionsScanner gate |
| Alpaca Paper Account | `alpaca-py` | Account equity, positions, orders | Portfolio reconciliation |
| IEX Feed | Via Alpaca | Free equity data feed | Default: `ALPACA_DATA_FEED=iex` |

### 13.2 Research & Macro Data

| Source | Library | Data | Used For |
|---|---|---|---|
| Yahoo Finance | `yfinance` | Global OHLCV, macro series | Harvester, NSE fallback |
| VIX (`^VIX`) | `yfinance` | Volatility index | Macro overlay (weight: 0.15) |
| TNX (`^TNX`) | `yfinance` | 10Y Treasury yield | Macro overlay |
| IRX (`^IRX`) | `yfinance` | 3M T-bill rate | Macro overlay |
| Yahoo Finance RSS | `beautifulsoup4` | News headlines | Sentiment → analyst |
| Google News RSS | `beautifulsoup4` | News headlines | Sentiment fallback |

### 13.3 Persistence Layer

| Store | Format | Contents |
|---|---|---|
| `runtime/state.json` | JSON | RuntimeState: mode, equity, positions, flags |
| `runtime/session_events*.jsonl` | JSONL | All bot events (open, close, signal, error) |
| `runtime/signals.csv` | CSV | Strategy signals with scores |
| `runtime/trades.csv` | CSV | Closed trades with P&L |
| `runtime/equity_curve.csv` | CSV | Equity snapshots over time |
| `runtime/guardrails.csv` | CSV | All blocked entry reasons |
| `runtime/analyst_signals.csv` | CSV | Historical analyst scores (35,550 rows) |
| `runtime/optimizer_results.csv` | CSV | Optuna trial results (962 rows) |
| `runtime/aegis_memory.db` | SQLite | Trades, decisions, regime history |
| `runtime/global_data/daily_full.csv` | CSV | 6,905 rows × 16 tickers daily OHLCV (2005–2026) |
| `runtime/global_data/intraday_1h.csv` | CSV | 27,527 rows × 16 tickers hourly (2023–2026) |
| `xgboost_model.json` | XGBoost JSON | Trained ML signal filter artifact |
| `runtime/backtests/` | Directories | 1,100 backtest run results |

---

## 14. Dashboard & Monitoring Infrastructure

### 14.1 Flask Dashboard API

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | Full HTML operations dashboard (Jinja2) |
| `/healthz` | GET | `{"status": "ok"}` — HTTP 200 when healthy |
| `/api/snapshot` | GET | Market data snapshot per symbol (prices, status, bar count) |
| `/api/runtime` | GET | Worker status, heartbeats, mode, equity, positions |
| `/api/bars/<symbol>` | GET | Recent OHLCV bars for charting |

### 14.2 Watchdog Health Score

```
Health score (0–100) updated on every API call:
  Components: API success rate | drawdown from peak | order fill rate | data freshness

Status levels:
  HEALTHY  (≥ 80): normal operation
  DEGRADED (50–79): Watchdog Telegram alert, position sizing reduced
  CRITICAL (< 50): all new entries blocked, Telegram 🚨 critical alert
```

---

## 15. Backtesting & Research Pipeline

### 15.1 Engine Parameters

```
Input:          OHLCV CSV + strategy configuration
Starting equity: $100,000
Slippage:        2 basis points per trade
Output:          runtime/backtests/run_{timestamp}/
                 ├── summary.json  (trades, P&L, Sharpe, drawdown)
                 ├── trades.csv
                 └── equity_curve.csv
```

### 15.2 Available Backtest Scripts

| Script | Market | Strategy Focus |
|---|---|---|
| `backtest_5min.py` | NSE (BankNifty) | 5-minute price action |
| `backtest_banknifty.py` | NSE | BankNifty-specific rules |
| `backtest_indian.py` | NSE | Indian equity basket |
| `backtest_price_action.py` | Any | Candle pattern strategies |
| `backtest_refined.py` | Multi-market | Multi-strategy refined |
| `final_optimized_backtest.py` | Any | Optuna-tuned parameters |
| `exact_pattern_backtest.py` | Any | Exact pattern matching |

### 15.3 Global Research Data Pipeline

```bash
python run_global_training.py
```

Fetches via `yfinance`:
- 16 global tickers × daily OHLCV → 2005–2026 (6,905 rows)
- 16 tickers × hourly OHLCV → 2023–2026 (27,527 rows)

---

## 16. Deployment & Operations

### 16.1 macOS (launchd — Production Config)

```bash
./setup_launchd.sh
# Installs: ~/Library/LaunchAgents/com.aegis.tradingbot.plist
# KeepAlive: true — auto-restarts if killed
# Override: NSE_ENABLED=false set in plist environment

launchctl list | grep aegis   # check status
```

### 16.2 Linux (systemd)

```bash
sudo cp aegis.service.template /etc/systemd/system/aegis.service
sudo systemctl daemon-reload && sudo systemctl enable aegis
sudo systemctl start aegis
journalctl -fu aegis          # live logs
```

### 16.3 Process Lock

`bot/process_lock.py` creates a PID lock file at startup. A second instance detects the existing lock and exits — preventing double-trading.

### 16.4 Runtime Supervisor Restart Logic

```
Worker heartbeat stale (> 180s) → restart with 2s backoff
Max 5 restarts exceeded → permanently stop worker
Any worker critical failure → PAUSE_ALL_ON_WORKER_FAILURE=true halts all entries
Circuit breaker open → trading suspended until cooldown expires
```

---

## 17. Testing & Quality Assurance

### 17.1 Test Results

```
October 2026 audit: 152 passed, 1 deprecation warning, 3 subtests passed
Runtime: 2.94 seconds
```

### 17.2 Coverage Areas

| Module | What's Tested |
|---|---|
| `bot/strategies/` | Signal generation, edge cases, regime gating |
| `bot/risk_manager.py` | Sizing, Kelly, stops, trailing stops, all guardrails |
| `bot/portfolio.py` | Position add/remove, exposure calculations |
| `bot/indicators.py` | ATR, RSI, EMA, SMA, MACD mathematical correctness |
| `bot/ml_filter.py` | Model load, predict, fail-open behavior |
| `bot/analyst/` | Regime detection, sentiment, signal aggregation |
| `bot/circuit_breaker.py` | Failure counting, cooldown, recovery |
| `bot/watchdog.py` | Health score computation, alert thresholds |
| `bot/memory.py` | SQLite CRUD operations |
| `dashboard.py` | All API endpoint responses |
| `bot/brokers/` | Broker abstraction interface |

### 17.3 Doctor Check

```bash
python run_bot.py --doctor
```

Verifies (read-only, no orders):
- Alpaca API credentials and account mode
- XGBoost model artifact presence and loadability
- Local Ollama service availability
- Waitress installation
- Worker heartbeat freshness
- Paper mode confirmation

---

## 18. Security & Safety Controls

### 18.1 Credential Protection

| Measure | Implementation |
|---|---|
| No hardcoded secrets | All credentials in `.env` via `python-dotenv` |
| `.gitignore` | `.env`, `*.db`, `*.csv`, `*.jsonl`, `runtime/`, model artifacts excluded |
| Telegram auth | Commands only accepted from authorized `TELEGRAM_CHAT_ID` |
| TradingView secret | HMAC passphrase (min 32 chars) required on every webhook alert |
| Paper mode flag | `ALPACA_PAPER=true` — uses `paper-api.alpaca.markets` URL |

### 18.2 Circuit Breaker

After N consecutive Alpaca API failures → circuit OPEN → all trading suspended for cooldown period → next success → circuit CLOSES → trading resumes.

### 18.3 PDT Safeguard

On startup, checks if account is flagged as Pattern Day Trader with equity < $25,000 and warns — Alpaca will reject day trades beyond the PDT limit.

---

## 19. Known Gaps & Limitations

| # | Gap | Impact |
|---|---|---|
| 1 | Backtest coverage mismatch | Default runner tests Red/Green BankNifty, not live US strategies |
| 2 | ML retraining labels | Needs matched open/close pairs; current session events have 15 opens, 0 closed matches |
| 3 | Regime history in DB | Zero rows — dashboard shows "UNKNOWN confidence" |
| 4 | Aggregator model absent | `xgb_aggregator_model.json` missing → falls back to weighted average |
| 5 | Multi-scale gate inactive | Config enables 3/4 timeframe confirmation but `bot/main.py` does not call the engine |
| 6 | RL training quality | PPO trained on synthetic random walks, not real trade trajectories |
| 7 | TradingView public URL | Cloud delivery requires public HTTPS endpoint — not verified in paper demo |
| 8 | NSE disabled | LaunchAgent overrides `NSE_ENABLED=false` |
| 9 | No validated win-rate report | No independent walk-forward accuracy percentage established |
| 10 | Distributed architecture | `bot/distributed/` is experimental and not launched |

---

## 20. Glossary

| Term | Definition |
|---|---|
| ATR | Average True Range — N-bar volatility measure |
| Alpaca | Commission-free US brokerage with paper and live API |
| Baum-Welch | EM algorithm for fitting Hidden Markov Models |
| Circuit Breaker | Automatic trading halt after consecutive API failures |
| Donchian Channel | N-bar price channel for breakout strategies |
| EMA | Exponential Moving Average — weighted towards recent prices |
| GARCH | Generalized AutoRegressive Conditional Heteroskedasticity — volatility model |
| HMM | Hidden Markov Model — probabilistic latent state sequence model |
| IEX | Investors Exchange — free Alpaca equity data feed |
| Kelly Criterion | Formula for optimal bet sizing: f* = (bp − q) / b |
| Launchd | macOS service management daemon |
| NSE | National Stock Exchange of India |
| Ollama | Local LLM inference server |
| Optuna | Bayesian hyperparameter optimization (TPE algorithm) |
| PCR | Put/Call Ratio — options market sentiment indicator |
| PDT | Pattern Day Trader — FINRA rule requiring $25,000 minimum equity |
| PPO | Proximal Policy Optimization — RL algorithm |
| qwen2.5 | Open-source LLM used by Investment Committee |
| RSI | Relative Strength Index — momentum oscillator (0–100) |
| Sharpe Ratio | (mean return − risk-free rate) / std dev × √252 |
| SMA | Simple Moving Average |
| SQLite | Embedded serverless relational database |
| TPE | Tree-structured Parzen Estimator — Optuna's Bayesian optimizer |
| TWAP | Time-Weighted Average Price execution algorithm |
| VADER | Valence Aware Dictionary and sEntiment Reasoner |
| VWAP | Volume-Weighted Average Price |
| Waitress | Pure-Python production WSGI server |
| Walk-Forward | Backtest methodology using rolling in/out-of-sample windows |
| XGBoost | Extreme Gradient Boosting ML library |

---

<div align="center">

---

**AEGIS QUANTITATIVE TRADING SYSTEM — TECHNICAL PROJECT REPORT v3.0**

*Describes the system as built and audited in October 2026.*  
*Not a prospectus, investment offer, or performance guarantee.*  
*All trading described herein uses simulated paper capital only.*

---

</div>
