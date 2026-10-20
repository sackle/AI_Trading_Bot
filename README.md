<div align="center">

<h1>⚡ Aegis — Quantitative AI Trading System</h1>

<p>
  <strong>Multi-strategy paper-trading bot with ML/LLM advisors, regime detection, risk engine, real-time dashboard, and TradingView signal bridge</strong>
</p>

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/sackle/AI_Trading_Bot/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://alpaca.markets/"><img src="https://img.shields.io/badge/Broker-Alpaca%20Paper-yellow?logo=alpaca&logoColor=white" alt="Alpaca"></a>
  <img src="https://img.shields.io/badge/Mode-Paper%20Trading%20Only-orange" alt="Paper Trading">
  <img src="https://img.shields.io/badge/Tests-152%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/XGBoost-3.3.0-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Flask-3.1.3-black?logo=flask" alt="Flask">
</p>

<p>
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-trading-strategies">Strategies</a> •
  <a href="#-ai--ml-stack">AI / ML Stack</a> •
  <a href="#-dashboard">Dashboard</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-backtesting">Backtesting</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="AEGIS_PROJECT_REPORT.md">📋 Full Technical Report</a> •
  <a href="#-disclaimer">Disclaimer</a>
</p>

</div>

---

> **⚠️ IMPORTANT — Read before running**
> Aegis is a software research and paper-trading demonstration project. It is **not financial advice**, does not guarantee profit, and **must never be used with real money** without independent legal, compliance, and risk review. Paper results do not predict live performance. See the [full disclaimer](#-disclaimer).

### 📄 Documentation

| Document | Description |
|---|---|
| **[README.md](README.md)** *(this file)* | Setup, configuration, quick-start for all platforms |
| **[AEGIS_PROJECT_REPORT.md](AEGIS_PROJECT_REPORT.md)** | Full technical report — architecture, all ML models, algorithms, bot coordination, Telegram internals, trade lifecycle, risk engine, deployment, and glossary |
| **[PROJECT_REPORT.md](PROJECT_REPORT.md)** | Audited system report — runtime status, strategy validation, data inventory, known gaps |
| **[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)** | Paper-trading demo checklist and integration limits |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Trading Strategies](#-trading-strategies)
- [AI / ML Stack](#-ai--ml-stack)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [macOS Setup](#macos-setup)
  - [Linux Setup](#linux-setup)
  - [Windows Setup](#windows-setup)
- [Configuration](#-configuration)
- [Running the Bot](#-running-the-bot)
- [Dashboard](#-dashboard)
- [Backtesting](#-backtesting)
- [TradingView Integration](#-tradingview-integration)
- [NSE / Indian Markets](#-nse--indian-markets)
- [Risk Management](#-risk-management)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Tests](#-tests)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## 🔭 Overview

**Aegis** (internally branded **Aegis Quantum / Aegis Trading Bot v3**) is a Python-based, multi-strategy quantitative trading system built for research, paper-trading, and educational purposes. It integrates:

- **Alpaca Paper Trading** for U.S. equities execution with IEX market data
- **4 strategy families** across 9 configured U.S. symbols
- **ML/LLM advisory layer** — XGBoost signal filter, HMM regime detector, PPO exit agent, and optional local Ollama LLM committee
- **Bayesian hyperparameter tuning** via Optuna running in a background thread
- **Real-time Flask operations dashboard** (Aegis Quantum UI)
- **TradingView webhook bridge** for supplemental alert-driven entries
- **NSE paper-trading loop** for Indian equity and F&O markets (optional)
- **152 automated tests** with pytest

---

## ✨ Features

| Category | Features |
|---|---|
| **Execution** | Alpaca Paper & IEX feed · process lock · supervised worker threads · heartbeat restart/backoff · TWAP execution |
| **Strategies** | Mean reversion · Momentum breakout (Donchian) · EMA trend following · Red/Green candle breakout |
| **Risk** | 1% equity per trade · 1×ATR hard stop · trailing stops · 3% daily drawdown limit · 5-position cap · 80% gross-exposure cap · earnings blackout · circuit breaker |
| **ML / AI** | XGBoost trade filter · HMM regime detection · PPO RL exit agent · GARCH volatility forecast · Optuna Bayesian tuning · VADER/FinBERT sentiment · local Ollama LLM committee |
| **Analytics** | Analyst signal aggregator · macro overlay (VIX, TNX, IRX) · options PCR/max-pain gate · microstructure order-flow proxy · Hurst exponent · multi-timeframe scanner |
| **Dashboard** | Live equity curve · open positions · trade log · signal timeline · health endpoint |
| **Research** | Backtesting engine · global data harvester · historical simulator · memory seeder · weekend batch analyzer |
| **Notifications** | Telegram bot · watchdog webhook alert |
| **DevOps** | macOS LaunchAgent · systemd service template · process lock · structured JSON logging |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        run_bot.py                                │
│  Process Lock → RuntimeCoordinator → RuntimeSupervisor           │
│                        │                                         │
│          ┌─────────────┴──────────────┐                          │
│          ▼                            ▼                          │
│    US TradingBot                NSE TradingBot (opt.)            │
│          │                                                       │
│  ┌───────┼────────────────────────────────┐                      │
│  │       │  Analyst / Regime / ML layer   │                      │
│  │  ┌────▼──────────────────────────┐     │                      │
│  │  │  Strategy Evaluation           │     │                      │
│  │  │  (per-bar signal → gates)      │     │                      │
│  │  └────┬──────────────────────────┘     │                      │
│  │       │ risk sizing / stop placement   │                      │
│  │       ▼                               │                      │
│  │  Alpaca Paper Orders ◄────────────────┘                      │
│  │       │                                                       │
│  │  ┌────▼──────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │  │ Runtime State  │  │ SQLite Memory │  │  CSV / JSONL   │     │
│  │  │ Store (JSON)  │  │ (trades/regime)│  │  event logs    │     │
│  │  └───────────────┘  └──────────────┘  └────────┬───────┘     │
│  └──────────────────────────────────────────────── │ ────────────┘
│                                                     │
│                 ┌───────────────────────────────────▼──────────┐ │
│                 │           Flask Dashboard (Waitress)          │ │
│                 │   /api/snapshot  /api/runtime  /healthz       │ │
│                 └───────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
         ▲                          ▲
  Alpaca REST + Stream       TradingView Webhook (optional)
  IEX Market Data            Telegram Notifications (optional)
```

**Execution model:** Supervised monolithic runtime (`run_bot.py`) — a `RuntimeSupervisor` manages worker threads for U.S. and optional NSE loops, with per-worker heartbeats, restart-with-backoff, and pause-on-failure. The `bot/distributed/` microservice architecture is present in code but **experimental and not launched** by default.

---

## 📈 Trading Strategies

### U.S. Strategies (Live — Alpaca Paper)

| Strategy | Symbols | Timeframe | Entry Logic | Exit Logic | Risk |
|---|---|---|---|---|---|
| **Mean Reversion** | SPY, QQQ | 15 min | Price below lower Bollinger (20-SMA ± σ·mult) + RSI(14) < 35 for long; reverse for short | Return to 20-bar mean | 1×ATR hard stop · 1.5×ATR trail |
| **Momentum Breakout** | AAPL, TSLA | 1 hour | 20-bar Donchian channel break + volume ≥ 1.5× 20-bar avg | Downside Donchian/volume break exits long | 1×ATR hard stop · 1.5×ATR trail |
| **Trend Following** | FISV, AMD, CB, CMCSA, DIS | 4 hour | EMA(50)/EMA(200) golden cross + MACD histogram direction + volume > 1.2× avg | Death cross | 1×ATR hard stop · 2×ATR trail |
| **Red/Green Breakout** | User-configured | 5 min | Red candle → green candle closing above red-high → current-bar high breaks green-high | Configurable | 2×ATR trail |

### NSE Strategies (Optional — Paper Only)

| Type | Universe | Logic |
|---|---|---|
| Intraday cash equity | 20 symbols | Mean reversion with NSE tick data |
| Daily swing | 10 symbols | EMA trend following |
| F&O confirmation | 8 underlyings | FnO scanner PCR / OI shift gate |

---

## 🤖 AI / ML Stack

### Signal Intelligence Pipeline

```
Raw bar data
     │
     ├──► Technical Indicators (SMA, EMA, ATR, RSI, MACD, Donchian,
     │                          VWAP, Hurst, order-flow imbalance)
     │
     ├──► MarketAnalystBot
     │       ├─ Sentiment (VADER / Google News RSS)        weight: 0.25
     │       ├─ Regime (3-state Gaussian HMM, n_iter=75)  weight: 0.30
     │       ├─ Technical scan                             weight: 0.30
     │       └─ Macro overlay (VIX, TNX, IRX via yfinance) weight: 0.15
     │
     ├──► MLSignalFilter (XGBoost binary classifier)
     │       └─ 7 features: side, ATR, analyst confidence,
     │          sentiment, regime, technical, macro scores
     │          Trained artifact: xgboost_model.json
     │
     ├──► InvestmentCommittee (optional local Ollama LLM debate)
     │       └─ Model: qwen2.5 — advisory only, fail-open
     │
     ├──► VirtualHumanBrain (optional Ollama playbook review)
     │
     ├──► USOptionsScanner (PCR / max-pain / unusual activity)
     │
     └──► Entry gates → Alpaca Paper order
                  │
                  └──► RLExitAgent (PPO) — exit recommendation
                           Observation: P&L%, ATR, bars-held, context
```

### Models & Algorithms

| Model | Library | Purpose |
|---|---|---|
| XGBoost classifier | `xgboost==3.3.0` | Trade win-probability filter (7 features, 50 rounds) |
| Gaussian HMM | `hmmlearn==0.3.3` | 3-state market regime detection (trending / mean-reverting / high-vol) |
| PPO RL agent | `stable-baselines3` | Exit timing recommendation |
| GARCH volatility | `scipy==1.18.0` | Volatility forecast for position sizing |
| Optuna Bayesian tuner | `optuna==4.9.0` | Strategy hyperparameter search (50 trials, 7-day refit) |
| VADER sentiment | `vaderSentiment==3.3.2` | News headline polarity scoring |
| Statistical arbitrage | `scikit-learn==1.9.0` | Z-score spread mean reversion (research module) |
| XGBoost aggregator | `xgboost==3.3.0` | Analyst signal meta-learner (falls back to weighted avg) |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | **3.13** |
| pip | 23+ | latest |
| Alpaca account | Paper account | Paper account |
| RAM | 4 GB | 8 GB |
| Disk | 1 GB | 2 GB |
| Network | Internet | Stable broadband |
| Ollama (optional) | — | Latest + `qwen2.5` pulled |

Get a free **Alpaca Paper** account at [alpaca.markets](https://alpaca.markets) — no real money needed.

---

### macOS Setup

```bash
# 1. Clone the repository
git clone https://github.com/sackle/AI_Trading_Bot.git
cd AI_Trading_Bot

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env in your editor and fill in your Alpaca Paper API credentials
nano .env          # or: code .env / open -e .env

# 5. Run readiness checks (read-only, no orders submitted)
python run_bot.py --check
python run_bot.py --doctor

# 6. Run the test suite
python -m pytest -q

# 7. Start the bot and dashboard (two separate terminals)
python run_bot.py          # Terminal 1 — trading engine
python run_dashboard.py    # Terminal 2 — operations dashboard
```

**macOS LaunchAgent (auto-start on login):**
```bash
chmod +x setup_launchd.sh
./setup_launchd.sh
# Bot will auto-start at login and be kept alive by launchd
```

---

### Linux Setup

```bash
# 1. Install Python 3.11+ if needed (Ubuntu/Debian example)
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 2. Clone the repository
git clone https://github.com/sackle/AI_Trading_Bot.git
cd AI_Trading_Bot

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure environment variables
cp .env.example .env
nano .env    # fill in your Alpaca Paper API credentials

# 6. Run readiness checks
python run_bot.py --check
python run_bot.py --doctor

# 7. Run the test suite
python -m pytest -q

# 8. Start the bot and dashboard (two separate terminals / tmux panes)
python run_bot.py          # Terminal 1
python run_dashboard.py    # Terminal 2
```

**Linux systemd service (run as a service):**
```bash
# Edit the template to set your username and project path
nano aegis.service.template

# Install and enable
sudo cp aegis.service.template /etc/systemd/system/aegis.service
sudo systemctl daemon-reload
sudo systemctl enable aegis
sudo systemctl start aegis
sudo systemctl status aegis

# View live logs
journalctl -fu aegis
```

**Optional: Redis for distributed state** (if enabling experimental distributed mode)
```bash
sudo apt install -y redis-server
sudo systemctl start redis
```

---

### Windows Setup

> **Recommended:** Use WSL2 (Ubuntu) and follow the Linux instructions above for the best compatibility. Native Windows is supported but less tested.

```powershell
# 1. Open PowerShell as Administrator and install Python 3.11+ from:
#    https://www.python.org/downloads/windows/
#    Ensure "Add Python to PATH" is checked during installation

# 2. Clone the repository
git clone https://github.com/sackle/AI_Trading_Bot.git
cd AI_Trading_Bot

# 3. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# If execution policy blocks the script:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure environment variables
copy .env.example .env
notepad .env    # fill in your Alpaca Paper API credentials

# 6. Run readiness checks
python run_bot.py --check
python run_bot.py --doctor

# 7. Run the test suite
python -m pytest -q

# 8. Start the bot and dashboard (two separate terminals)
# Terminal 1:
python run_bot.py

# Terminal 2:
python run_dashboard.py
```

**Windows Task Scheduler (auto-start):**
- Create a Basic Task → trigger: At log on → action: `Start a program`
- Program: `C:\path\to\AI_Trading_Bot\.venv\Scripts\python.exe`
- Arguments: `run_bot.py`
- Start in: `C:\path\to\AI_Trading_Bot`

---

## ⚙️ Configuration

All configuration is done through environment variables in your `.env` file. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

### Core Settings

```ini
# ── Alpaca Credentials ──────────────────────────────────────────────
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_PAPER=true                    # ALWAYS true for paper trading
ALPACA_DATA_FEED=iex                 # iex (free) or sip (subscription)

# ── Broker Backend ───────────────────────────────────────────────────
BROKER_BACKEND=alpaca                # alpaca | paper (virtual, no broker)

# ── Analyst & Watchdog ───────────────────────────────────────────────
ANALYST_ENABLED=true
ANALYST_INTERVAL_SECONDS=300
ANALYST_AGGREGATOR_MODE=auto         # auto | xgboost | weighted
WATCHDOG_ENABLED=true
WATCHDOG_MEMORY_LIMIT_MB=2048

# ── Strategy Optimization ────────────────────────────────────────────
OPTIMIZER_ENABLED=true
OPTIMIZER_REFIT_INTERVAL_DAYS=7
OPTIMIZER_MAX_TRIALS=50

# ── Worker Supervision ───────────────────────────────────────────────
WORKER_MAX_RESTARTS=5
WORKER_RESTART_BACKOFF_SECONDS=2
PAUSE_ALL_ON_WORKER_FAILURE=true

# ── NSE (Indian Markets — optional) ─────────────────────────────────
NSE_ENABLED=false                    # Enable only if needed
NSE_PAPER_EQUITY=500000

# ── Local LLM Committee (requires Ollama) ───────────────────────────
AI_COMMITTEE_ENABLED=false           # Set true after: ollama pull qwen2.5
AI_VIRTUAL_BRAIN_ENABLED=false

# ── TradingView Webhook (optional) ───────────────────────────────────
TRADINGVIEW_ENABLED=false
TRADINGVIEW_WEBHOOK_SECRET=          # openssl rand -hex 32

# ── Red/Green Breakout (optional, no default symbols) ───────────────
RED_GREEN_BREAKOUT_ENABLED=false
RED_GREEN_BREAKOUT_SYMBOLS=          # comma-separated, e.g. NVDA,META
```

### Obtaining Alpaca API Keys

1. Go to [alpaca.markets](https://alpaca.markets) and create a free account
2. Navigate to **Paper Trading** → **Overview**
3. Click **Generate New Key** under API Keys
4. Copy the **API Key ID** and **Secret Key** into your `.env`

> **Never use Live Trading keys during development.** The `ALPACA_PAPER=true` flag is your safety net — confirm it is set before starting the bot.

---

## ▶️ Running the Bot

### Available Commands

```bash
# Check configuration validity (read-only, no orders)
python run_bot.py --check

# Full health diagnostics (read-only, no orders)
python run_bot.py --doctor

# Start the trading engine
python run_bot.py

# Start the operations dashboard
python run_dashboard.py

# Run backtests
python run_backtest.py

# Batch weekend analysis and reporting
python run_weekend_jobs.py

# Download global market data for research
python run_global_training.py

# Run all tests
python -m pytest -q

# Run tests with verbose output
python -m pytest -v

# Watchdog script (keeps bot alive, for use outside launchd/systemd)
./bot_watchdog.sh
```

---

## 📊 Dashboard

The **Aegis Quantum** operations dashboard runs locally at `http://127.0.0.1:5012/` by default.

```bash
python run_dashboard.py
```

### Dashboard API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard UI |
| `/healthz` | GET | Health check — returns HTTP 200 when healthy |
| `/api/snapshot` | GET | Current bot state snapshot |
| `/api/runtime` | GET | Worker / supervisor runtime status |
| `/api/bars/<symbol>` | GET | Recent OHLCV bars for a symbol |

Dashboard panels include: account equity curve, open positions, recent trades, analyst signal timeline, regime state, ML filter output, watchdog health score, and worker heartbeats.

> Do not expose the dashboard publicly without adding authentication and TLS. It is designed for local use only.

---

## 🔬 Backtesting

```bash
# Run the built-in backtester (defaults to Red/Green on BankNifty 5m data)
python run_backtest.py

# Run specific backtest scripts
python backtest_5min.py
python backtest_banknifty.py
python backtest_indian.py
python backtest_price_action.py
python backtest_refined.py
python final_optimized_backtest.py
python exact_pattern_backtest.py
```

Backtest defaults:
- Starting equity: **\$100,000**
- Slippage: **2 basis points**
- Results saved to: `runtime/backtests/`

> **Note:** The default `run_backtest.py` runner currently evaluates the Red/Green BankNifty strategy, not the live U.S. strategies. Results are research artifacts, not validated out-of-sample performance.

---

## 📡 TradingView Integration

Aegis includes a webhook server that can receive TradingView alerts and use them as supplemental trade signals.

```ini
# In .env:
TRADINGVIEW_ENABLED=true
TRADINGVIEW_WEBHOOK_SECRET=<output of: openssl rand -hex 32>
TRADINGVIEW_WEBHOOK_HOST=127.0.0.1
TRADINGVIEW_WEBHOOK_PORT=8080
TRADINGVIEW_SIGNAL_MODE=supplement   # supplement | override
TRADINGVIEW_PUBLIC_URL=https://your-tunnel-url.example.com/webhook
```

TradingView alert body template:
```json
{
  "passphrase": "{{your_TRADINGVIEW_WEBHOOK_SECRET}}",
  "symbol": "{{ticker}}",
  "action": "buy",
  "price": {{close}}
}
```

> TradingView cloud requires a public HTTPS URL on port 443. Use a TLS-terminating reverse proxy or a temporary tunnel (e.g. `cloudflared tunnel`) for local development. Tunnel URLs change on restart — update the TradingView alert when they do.

PineScript strategies for TradingView are in `pinescripts/`:
- `mean_reversion.pine`
- `momentum_breakout.pine`
- `trend_following.pine`
- `red_green_breakout.pine`

---

## 🇮🇳 NSE / Indian Markets

The NSE module is implemented and configurable but **disabled by default** (`NSE_ENABLED=false`).

```ini
NSE_ENABLED=true
NSE_PAPER_EQUITY=500000      # Paper capital in INR
NSE_LOOP_INTERVAL_SECONDS=60
NSE_DATA_WORKERS=6
```

When enabled, it runs a separate supervised paper-trading loop covering:
- **20 intraday cash equity** symbols (mean reversion, 5-min)
- **10 daily swing** symbols (EMA trend following)
- **8 F&O underlyings** (FnO scanner with PCR / OI confirmation)

Data sources: `nsepy`, `nsepython`, and `yfinance` as fallback. All NSE trading is simulated locally in `runtime/nse_paper_state.json`.

---

## 🛡 Risk Management

Aegis implements a layered risk framework:

| Control | Setting |
|---|---|
| **Position sizing** | 1% of equity per trade, ATR-distance based |
| **Hard stop-loss** | 1× ATR(14) from entry |
| **Trailing stop** | 1.5–2× ATR depending on strategy |
| **Daily drawdown limit** | Stop new entries after 3% equity loss in a day |
| **Consecutive loss limit** | Pause after 3 back-to-back losses |
| **Max open positions** | 5 |
| **Max notional per position** | 35% of equity |
| **Gross exposure cap** | 80% of equity |
| **Portfolio heat limit** | 5% of equity |
| **Earnings blackout** | No entries 2 days before / 1 day after earnings |
| **Regime gate** | Mean reversion suppressed in trending regime |
| **Options disagreement gate** | Block entries when options PCR conflicts |
| **Volatility scaling** | Position size reduced in HIGH_VOLATILITY regime |
| **Correlation limit** | Prevents over-concentration in correlated assets |
| **Circuit breaker** | Emergency halt on extreme conditions |
| **Supervisor restart** | Max 5 restarts with 2s backoff; pause-all on failure |

---

## 📁 Project Structure

```
AI_Trading_Bot/
├── bot/                          # Core bot package
│   ├── main.py                   # TradingBot — main per-bar loop
│   ├── supervisor.py             # RuntimeSupervisor — worker management
│   ├── runtime_coordinator.py    # Shared state coordination
│   ├── runtime_state.py          # RuntimeStateStore
│   ├── config.py                 # (root) All configuration / strategy map
│   ├── agents/                   # AI advisor agents
│   │   ├── committee.py          # InvestmentCommittee (local LLM debate)
│   │   ├── virtual_brain.py      # VirtualHumanBrain (playbook review)
│   │   ├── news_terminal.py      # NewsTerminal (RSS aggregator)
│   │   ├── desk_manager.py       # DeskManager
│   │   ├── post_mortem.py        # PostMortemAnalyzer (loss review)
│   │   └── trading_playbook.md   # LLM advisory playbook
│   ├── analyst/                  # Market analysis modules
│   │   ├── analyst.py            # MarketAnalystBot
│   │   ├── regime.py             # RegimeDetector (HMM)
│   │   ├── aggregator.py         # Signal aggregator (XGBoost / weighted)
│   │   ├── sentiment.py          # VADER sentiment engine
│   │   ├── scanner.py            # Technical scanner
│   │   └── macro.py              # Macro overlay (VIX/TNX/IRX)
│   ├── strategies/               # Pluggable strategy implementations
│   │   ├── mean_reversion.py
│   │   ├── momentum_breakout.py
│   │   ├── trend_following.py
│   │   ├── red_green_breakout.py
│   │   └── statistical_arbitrage.py
│   ├── brokers/                  # Broker abstraction layer
│   │   ├── base.py               # BaseBroker interface
│   │   ├── alpaca_broker.py      # Alpaca Paper/Live broker
│   │   ├── paper_broker.py       # Local virtual paper broker
│   │   └── ibkr_broker.py        # IBKR stub (not implemented)
│   ├── distributed/              # Experimental microservice workers
│   │   ├── orchestrator.py
│   │   ├── strategy_worker.py
│   │   ├── risk_engine.py
│   │   ├── analyst_worker.py
│   │   ├── market_data_feeder.py
│   │   ├── broker_service.py
│   │   ├── position_manager.py
│   │   └── webhook_service.py
│   ├── nse/                      # NSE / Indian market modules
│   │   ├── nse_loop.py
│   │   ├── broker_nse.py
│   │   ├── data_client.py
│   │   └── fno_scanner.py
│   ├── us_options/               # US options analytics
│   │   ├── options_scanner.py    # PCR / max-pain / unusual activity
│   │   └── pricing_engine.py
│   ├── notifications/
│   │   └── telegram_bot.py       # Telegram trade notifications
│   ├── tradingview/
│   │   ├── webhook_server.py     # Inbound alert receiver
│   │   └── signal_bridge.py      # Alert → strategy signal bridge
│   ├── training/
│   │   ├── historical_simulator.py
│   │   └── memory_seeder.py
│   ├── execution/
│   │   └── twap.py               # TWAP order execution
│   ├── data/
│   │   └── global_harvester.py   # Multi-asset OHLCV harvester
│   ├── earnings/                 # Earnings calendar blackout
│   ├── indicators.py             # Technical indicator library
│   ├── microstructure.py         # Order-flow / market-microstructure
│   ├── ml_filter.py              # MLSignalFilter (XGBoost)
│   ├── ml_pretrain.py            # Offline model pre-trainer
│   ├── rl_agent.py               # RLExitAgent (PPO)
│   ├── garch_model.py            # GARCH volatility forecasting
│   ├── regime_hmm.py             # Standalone HMM implementation
│   ├── optimizer.py              # StrategyOptimizer (Optuna)
│   ├── risk_manager.py           # RiskManager
│   ├── portfolio.py              # Portfolio tracker
│   ├── market_filters.py         # Entry gate filters
│   ├── circuit_breaker.py        # Emergency halt logic
│   ├── guardian.py               # Guardian risk controller
│   ├── watchdog.py               # WatchdogMonitor
│   ├── screener.py               # Symbol screener
│   ├── shadow_trader.py          # ShadowTrader (paper evaluation)
│   ├── strategy_evolver.py       # Genetic strategy evolver (research)
│   ├── drift_monitor.py          # Concept-drift detection
│   ├── multiscale.py             # Multi-timeframe confirmation
│   ├── network.py                # Network connectivity checks
│   ├── memory.py                 # TradingMemory (SQLite)
│   ├── state_store.py            # Persistent JSON state
│   ├── metrics.py                # Performance metrics
│   ├── artifacts.py              # Artifact save/load helpers
│   ├── resilience.py             # Retry / resilience utilities
│   ├── process_lock.py           # Single-instance lock
│   ├── health_check.py           # Health check helpers
│   ├── logger.py                 # Structured logging setup
│   ├── pipeline.py               # Research pipeline
│   ├── backtest.py               # Backtesting engine
│   ├── stress_test.py            # Portfolio stress tests
│   ├── market_streamer.py        # Alpaca WebSocket stream client
│   ├── api_command_center.py     # REST command API
│   ├── runtime_coordinator.py
│   ├── weekend_analyzer.py       # Weekend batch analysis
│   └── optimizer.py
├── pinescripts/                  # TradingView Pine Script strategies
│   ├── mean_reversion.pine
│   ├── momentum_breakout.pine
│   ├── trend_following.pine
│   └── red_green_breakout.pine
├── static/
│   └── dashboard.css             # Dashboard stylesheet
├── templates/
│   └── dashboard.html            # Dashboard Jinja2 template
├── tests/                        # Automated test suite (152 tests)
├── config.py                     # Master config + strategy symbol map
├── dashboard.py                  # Flask dashboard app
├── run_bot.py                    # Main entry point
├── run_dashboard.py              # Dashboard launcher
├── run_backtest.py               # Backtest launcher
├── run_global_training.py        # Global data + training runner
├── run_weekend_jobs.py           # Weekend batch jobs
├── backtest_5min.py              # 5-minute backtest script
├── backtest_banknifty.py         # BankNifty backtest
├── backtest_indian.py            # Indian market backtest
├── backtest_price_action.py      # Price action backtest
├── backtest_refined.py           # Refined multi-strategy backtest
├── final_optimized_backtest.py   # Optimized parameter backtest
├── exact_pattern_backtest.py     # Exact pattern matching backtest
├── check_alpaca.py               # Alpaca connectivity check
├── deploy_production.py          # Production deployment helper
├── download_banknifty.py         # BankNifty data downloader
├── setup_launchd.sh              # macOS LaunchAgent installer
├── start_bot.sh                  # Simple start script
├── start_services.sh             # Start all services
├── stop_services.sh              # Stop all services
├── bot_watchdog.sh               # External watchdog shell script
├── aegis.service.template        # systemd service template (Linux)
├── requirements.txt              # Pinned Python dependencies
├── pytest.ini                    # pytest configuration
├── .env.example                  # Environment variable template
├── .gitignore                    # Excludes credentials, data, artifacts
├── README.md                     # This file
├── PROJECT_REPORT.md             # Full system audit report
└── DEMO_RUNBOOK.md               # Paper-trading demo checklist
```

---

## 📦 Dependencies

All dependencies are pinned in [`requirements.txt`](requirements.txt).

### Core Libraries

| Library | Version | Purpose |
|---|---|---|
| `alpaca-py` | 0.43.5 | Alpaca broker REST + WebSocket client |
| `pandas` | 3.0.5 | Data manipulation and OHLCV processing |
| `numpy` | 2.5.1 | Numerical computing |
| `scipy` | 1.18.0 | GARCH, statistical routines |
| `scikit-learn` | 1.9.0 | ML utilities, statistical arbitrage |

### Machine Learning

| Library | Version | Purpose |
|---|---|---|
| `xgboost` | 3.3.0 | Trade signal filter + analyst aggregator |
| `hmmlearn` | 0.3.3 | Market regime detection (Gaussian HMM) |
| `optuna` | 4.9.0 | Bayesian hyperparameter optimization |
| `vaderSentiment` | 3.3.2 | News sentiment analysis |
| `joblib` | 1.5.3 | Model persistence and parallel jobs |

### Web Framework & Dashboard

| Library | Version | Purpose |
|---|---|---|
| `Flask` | 3.1.3 | Operations dashboard web server |
| `waitress` | 3.0.2 | Production WSGI server for dashboard |
| `Jinja2` | 3.1.6 | HTML templating |
| `Werkzeug` | 3.1.8 | WSGI utilities |

### Data & Market Data

| Library | Version | Purpose |
|---|---|---|
| `yfinance` | 1.5.2 | Yahoo Finance — global OHLCV, macro data |
| `beautifulsoup4` | 4.15.0 | HTML/RSS news parsing |
| `requests` | 2.34.2 | HTTP client |
| `websocket-client` | 1.9.0 | WebSocket market stream |
| `websockets` | 16.1.1 | Async WebSocket |
| `sseclient-py` | 1.9.0 | Server-sent events for live feeds |
| `curl_cffi` | 0.15.0 | HTTP client with TLS fingerprint bypass |

### Database & State

| Library | Version | Purpose |
|---|---|---|
| `SQLAlchemy` | 2.0.51 | ORM / SQLite interface |
| `alembic` | 1.18.5 | Database migrations |
| `peewee` | 4.2.6 | Lightweight ORM for memory DB |
| `redis` | 8.1.0 | Redis client (experimental distributed mode) |
| `fakeredis` | 2.37.0 | In-memory Redis mock for testing |

### Configuration & Utilities

| Library | Version | Purpose |
|---|---|---|
| `python-dotenv` | 1.2.2 | `.env` file loader |
| `pydantic` | 2.13.4 | Config validation and data models |
| `PyYAML` | 6.0.3 | YAML config parsing |
| `colorlog` | 6.12.0 | Color-coded terminal logging |
| `rich` | 15.0.0 | Rich terminal output and progress bars |
| `tqdm` | 4.70.0 | Progress bars |
| `msgpack` | 1.2.1 | Efficient binary serialization |
| `protobuf` | 7.35.1 | Protocol Buffers for Alpaca data |

### Testing

| Library | Version | Purpose |
|---|---|---|
| `pytest` | 9.1.1 | Test framework (152 tests pass) |
| `fakeredis` | 2.37.0 | Redis mock in tests |

### Optional (Not in requirements.txt — install separately if needed)

| Library | Purpose |
|---|---|
| `stable-baselines3` | PPO RL exit agent (installed in dev environment) |
| `gymnasium` | RL environment for exit agent training |
| `upstox-python-sdk` | Upstox broker integration (listed as dep, no active import found) |
| `ollama` (server) | Local LLM — required for `AI_COMMITTEE_ENABLED=true` |
| `qwen2.5` (model) | LLM model for committee: `ollama pull qwen2.5` |

---

## 🧪 Tests

```bash
# Run all tests (quiet)
python -m pytest -q

# Run all tests (verbose)
python -m pytest -v

# Run a specific test file
python -m pytest tests/test_risk_manager.py -v

# Run with coverage report
pip install pytest-cov
python -m pytest --cov=bot --cov-report=term-missing -q
```

**Test results (October 2026 audit):** `152 passed, 1 deprecation warning, 3 subtests passed` in 2.94 seconds.

Tests cover: strategies, risk manager, portfolio, indicators, ML filter, regime detector, analyst, dashboard API, broker abstraction, circuit breaker, watchdog, memory, and more.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for your changes and ensure all 152 existing tests still pass:
   ```bash
   python -m pytest -q
   ```

3. **Follow the existing code style** — structured logging via `bot/logger.py`, pydantic config validation, fail-open/safe defaults for optional components.

4. **Document your changes** — update `PROJECT_REPORT.md` if you change architecture or strategies.

5. **Never commit:**
   - Real API keys or credentials (`.env`)
   - Runtime data files (`*.db`, `*.csv`, `*.jsonl`, `runtime/`)
   - Trained model artifacts (`*.pkl`, `*.pt`, `xgboost_model.json`)
   - One-off debug/patch scripts (`fix_*.py`, `patch_*.py`)

6. Submit a **Pull Request** with a clear description of your changes.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Alpaca Markets](https://alpaca.markets) — Commission-free paper trading API
- [XGBoost](https://xgboost.readthedocs.io) — Gradient boosting ML framework
- [Optuna](https://optuna.org) — Bayesian hyperparameter optimization
- [hmmlearn](https://hmmlearn.readthedocs.io) — Hidden Markov Models
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — News sentiment analysis
- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance market data
- [Flask](https://flask.palletsprojects.com) — Operations dashboard web framework
- [Ollama](https://ollama.ai) — Local LLM inference server

---

## ⚠️ Disclaimer

**This project is for educational and research purposes only.**

- Aegis is a **paper-trading demonstration system**. It does not guarantee profits or predict future market performance.
- No part of this repository constitutes financial, investment, or trading advice.
- **Never connect real money** to this system without independent legal, compliance, risk-management, and technical review by qualified professionals.
- Past paper-trading results do not predict live trading performance. Live markets involve slippage, liquidity constraints, API failures, and regulatory requirements not fully modeled here.
- The ML models (XGBoost, HMM, PPO) have not been independently validated for out-of-sample trading accuracy. A model threshold does not guarantee a win rate.
- All use is at your own risk. The authors and contributors accept no liability for financial losses.

**By using this software, you agree to use it solely for paper trading and research purposes.**

---

<div align="center">
  <p>Built with ❤️ for quantitative trading research</p>
  <p>
    <a href="https://github.com/sackle/AI_Trading_Bot/issues">Report a Bug</a> •
    <a href="https://github.com/sackle/AI_Trading_Bot/issues">Request a Feature</a> •
    <a href="https://github.com/sackle/AI_Trading_Bot/discussions">Discussions</a>
  </p>
</div>
