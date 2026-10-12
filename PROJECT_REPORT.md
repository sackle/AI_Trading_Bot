# Aegis Quantitative Trading System — Project Report

**Audit date:** 12 October 2026  
**Scope:** Source code, runtime configuration, saved data/artifacts, readiness check, running services, and automated tests in this workspace. This describes the project as implemented; it is not a claim of trading profitability or investment advice.

## 1. Executive summary

Aegis is a Python-based, multi-strategy **paper-trading** system with a local operations dashboard. The live U.S. path uses Alpaca Paper trading and Alpaca IEX market data. It has three configured U.S. strategy families across nine symbols, a shared portfolio/risk layer, optional analyst/ML/LLM/options/earnings features, persistent runtime state, and a Flask dashboard.

The dashboard branding is **Aegis Quantum**; startup/runtime messages call the engine **Aegis Trading Bot v3**. The main code is under `bot/`, with separate launchers for bot, dashboard, backtests, and global training; `runtime/` holds generated state and artifacts. A concise demo guide exists at `DEMO_RUNBOOK.md`; no maintained root `README.md` was found during this audit.

The running service is a supervised **monolithic compatibility runtime**, not the distributed microservice architecture suggested by some module names. At the audit, the bot LaunchAgent and dashboard were running, the runtime reported PAPER mode and a healthy U.S. worker, the market-data snapshot was market-closed for nine symbols, the dashboard health endpoint returned HTTP 200, and the read-only readiness command reported **0 failures and 3 warnings**. The NSE worker was disabled in the actual LaunchAgent configuration. No orders were submitted by this audit.

The system is a functional paper-demo baseline, **not a validated production investment firm**. In particular, the repository does not establish a trustworthy 75% win accuracy or any guaranteed profit. There is no clean, sufficiently large, independently held-out result for the deployed strategies. The newest saved backtest summary contains zero trades, and the generic backtest runner currently evaluates a different strategy (Red/Green Bank Nifty) from the three configured U.S. live strategies.

## 2. Runtime architecture

```text
Alpaca Paper / IEX bars ──> US TradingBot ──> strategy signals
        │                       │                    │
        └─ news/options         ├─ analyst/regime    ├─ market & risk gates
                                ├─ ML/LLM advisors   └─ Alpaca orders
                                │                           │
                                └── runtime state, events, trade/equity logs
                                             │
                                      Flask dashboard

run_bot.py -> process lock -> RuntimeCoordinator -> RuntimeSupervisor -> US worker
                                                               └─────> NSE worker (disabled here)
```

### Technology stack

Python 3.13 in the tested environment; `alpaca-py` 0.43.5; pandas 3.0.5 and NumPy 2.5.1; XGBoost 3.3.0; hmmlearn 0.3.3; Optuna 4.9.0; Flask 3.1.3 with Waitress 3.0.2; yfinance 1.5.2; scikit-learn 1.9.0; SQLite; pytest 9.1.1; plus requests/websocket clients and VADER sentiment. The currently installed environment also has `stable-baselines3` and `gymnasium`, but they are not pinned in `requirements.txt`; Transformers is not installed, so the sentiment engine uses VADER rather than FinBERT. `upstox-python-sdk` is listed as a dependency but no active source import was found in this audit.

- **Launcher/supervision:** [`run_bot.py`](run_bot.py) obtains a process lock, shares a `RuntimeCoordinator`, and runs worker threads under `RuntimeSupervisor`. Worker heartbeats, restart/backoff, and pause-on-worker-failure are supported. The macOS LaunchAgent uses `KeepAlive`; the current launch configuration overrides `NSE_ENABLED=false`.
- **Trading engine:** [`bot/main.py`](bot/main.py) contains `TradingBot`: obtains bars through the broker/data clients, evaluates the configured strategy for a new bar, applies entry gates and risk sizing, submits orders, confirms fills, manages stops/exits, and persists state/events.
- **Broker/data:** Alpaca Paper is selected (`BROKER_BACKEND=alpaca`, `ALPACA_PAPER=true`, IEX feed). Historical bars are read through Alpaca REST; an Alpaca stream client is also created for configured U.S. symbols. The current saved snapshot says `alpaca_bars_rest`, `IEX`, and `market_closed`.
- **Shared state:** `RuntimeStateStore`, `RuntimeCoordinator`, `Portfolio`, SQLite memory, JSON runtime state, CSV logs, and JSONL session events support restart recovery, reconciliation, dashboard reporting, and analysis.
- **Dashboard:** [`dashboard.py`](dashboard.py) serves the local operations UI and read APIs including `/api/snapshot`, `/api/runtime`, `/api/bars/<symbol>`, and `/healthz`. At the audit, `http://127.0.0.1:5012/healthz` returned HTTP 200. Dashboard health confirms the UI process responds; it does not by itself prove that a market feed is currently streaming.
- **Distributed code:** `bot/distributed/` contains worker/service/message-bus components, but the readiness check explicitly reports that these are experimental and not launched. They should not be described as the active production architecture.

## 3. Trading strategies

The live U.S. strategy map in [`config.py`](config.py) contains nine unique symbols:

| Strategy | Symbols and bar interval | Entry/exit logic | Risk overlay |
|---|---|---|---|
| Mean reversion | SPY and QQQ, 15-minute | 20-bar SMA and standard-deviation bands; configured band multipliers 1.5 (SPY) and 1.8 (QQQ). Long below the lower band with RSI(14) < 35; short above the upper band with RSI > 65. Exit when price returns to the 20-bar mean. | Universal 1×ATR(14) hard stop; configured 1.5×ATR trailing behavior. Regime gate is intended to suppress this strategy in a trending regime. |
| Momentum breakout | AAPL and TSLA, 1-hour | 20-bar Donchian breakout with current volume at least 1.5× the previous 20-bar average. The code opens long on an upside breakout; downside Donchian/volume break exits the long rather than opening a short. | 1×ATR hard stop and 1.5×ATR trail. |
| Trend following | FISV, AMD, CB, CMCSA, DIS, 4-hour | EMA(50)/EMA(200) golden/death cross, confirmed by MACD histogram direction and volume > 1.2× its 20-bar average. Shorts are enabled in config, subject to account/broker shortability. Exit on reverse cross. | 1×ATR hard stop and 2×ATR trail. |
| Red/Green breakout | No assigned symbols; optional 5-minute strategy | Red candle followed by a green candle closing above the red candle high, then current-bar high breaks the green candle high. | Universal risk layer; strategy-specific 2×ATR trail. Enabled in `.env` but deliberately idle because the symbol list is empty. |

There is documentation drift: strategy docstrings still describe the momentum strategy as BTC/USD and the trend strategy as GLD/USO, while the active U.S. config uses AAPL/TSLA and FISV/AMD/CB/CMCSA/DIS. The configuration and `build_strategies()` code, not those old docstrings, define the current symbols.

### Strategy comparison (mechanics, not measured performance)

| Family | Designed for | Relative advantage | Main failure mode |
|---|---|---|---|
| Mean reversion | Range-bound markets | Buys/sells unusually stretched prices and exits toward the mean | Can keep fading a strong trend; regime filter is important |
| Donchian breakout | Persistent directional moves | Captures range breaks with volume confirmation | Whipsaws and false breakouts in choppy markets |
| EMA trend following | Longer trends | Simple, interpretable trend/crossover rules | Lag and repeated losses around sideways crossovers |
| Red/Green breakout | Short-term candle-pattern break | Small, explicit pattern rule | Very limited evidence; not running without an assigned symbol |

No apples-to-apples performance comparison is available in the project artifacts. The table compares rule mechanics only.

## 4. Algorithms and analytical methods

1. **Technical indicators:** SMA, rolling standard deviation, EMA, ATR with Wilder smoothing, RSI with Wilder smoothing, MACD (12/26/9), Donchian channels, volume profile, Hurst/R-S estimate, VWAP/rolling VWAP, momentum confidence, and an order-flow imbalance proxy. See [`bot/indicators.py`](bot/indicators.py) and [`bot/microstructure.py`](bot/microstructure.py). Several are shared utilities; not every indicator is used by every live strategy.
2. **Market regime:** `MarketAnalystBot` uses `RegimeDetector`: a three-state diagonal-covariance Gaussian HMM, `n_iter=75`, fit after at least 90 bars/50 valid feature rows and refit about every 10 minutes. Its features are ATR percentage, rolling return autocorrelation, and rolling volatility. If HMM fitting/prediction is unavailable, it uses ATR% and lag-1 autocorrelation rules. The rule fallback emits a fixed 0.5 confidence with roughly equal state probabilities; neither value is a statistically calibrated probability.
3. **Analyst signal aggregation:** weighted components are sentiment 0.25, regime 0.30, technical 0.30, and macro 0.15. The weighted aggregate calls a signal bullish/bearish above/below ±0.2 and maps absolute score to a 0–100 confidence. This score is a model score, not historical win probability. `ANALYST_AGGREGATOR_MODE=auto` can use XGBoost, but the separate aggregator artifact `runtime/xgb_aggregator_model.json` is absent; its implementation therefore falls back to the weighted signal until that model exists and has sufficient outcomes.
4. **Trade ML filter:** `MLSignalFilter` uses an XGBoost binary classifier with seven inputs: long/short, ATR, analyst confidence, sentiment, regime, technical, and macro scores. The loaded root artifact `xgboost_model.json` is 57,207 bytes, has 7 features and 50 boosting rounds. `.env` sets the nominal minimum win probability to 0.70. The filter's fail-open behavior returns 1.0 when no model/prediction is available, so a missing or failed model does not block entries.
5. **Parameter tuning:** Optuna searches strategy parameters (50 trials, 7-day refit interval in the current `.env`) in a background thread. The objective is a simplified 70/30 walk-forward Sharpe calculation over recent bars; it is not a full broker/fees/risk-engine replay.
6. **Exit policy:** a PPO `RLExitAgent` model artifact exists and loaded in the current Python environment. The exit path consults it before the strategy's deterministic exit. Its observation is P&L %, ATR, bars-held, and two zero-valued context fields; the current call passes a fixed bars-held value of 10. The training environment includes synthetic random-walk paths, while real trades are approximated as linearly interpolated trajectories. Treat this as experimental, not validated reinforcement-learning performance.
7. **Other methods:** GARCH-style volatility forecasting, stress-test calculations, correlation/portfolio limits, a separate offline HMM implementation, a z-score statistical-arbitrage strategy, and genetic strategy evolution exist in modules. The separate HMM, genetic evolver, and statistical-arbitrage strategy are not registered in the live strategy map. The multi-scale confirmation module/config requests 3-of-4 timeframe confluence, but `bot/main.py` does not call that engine; it is **not an active live gate**.

## 5. Bots, agents, and services

The repository mostly uses role/class names rather than unique branded bot names. A search found no standalone class/service named “Fedar”; the analyst implementation is `MarketAnalystBot`.

| Bot/agent/module | Purpose | Audit status / caveat |
|---|---|---|
| `TradingBot` (U.S.) | Core U.S. strategy loop, Alpaca orders, portfolio, fills, stops, and exits | Running in PAPER mode; one healthy supervised U.S. worker at the snapshot |
| `MarketAnalystBot` | Sentiment + regime + technical scan + macro analysis and signal record | Enabled. Saves analyst signals; the regime-history table currently has 0 rows, so confidence persistence is not evidenced in the current DB snapshot |
| `MLSignalFilter` | XGBoost trade-win filter | Enabled; `xgboost_model.json` loaded. Current session-event logs have 15 opens but no matched close pairs by the trainer's required order-id schema, so the background trainer cannot build fresh labeled examples from those events at present |
| `StrategyOptimizer` | Background Optuna parameter search | Enabled; code writes optimizer results and applies pending parameter updates between ticks. Optimization output is not itself proof of out-of-sample edge |
| `Watchdog` and `Guardian` | Health score, API/order failures, drawdown, emergency/entry restriction, exposure scaling | Enabled in current config and instantiated by the U.S. bot |
| `InvestmentCommittee` | Optional local-LLM trade debate/approval | Enabled; local Ollama `qwen2.5` is available. LLM output is advisory logic, not a calibrated signal |
| `VirtualHumanBrain` | Optional playbook-based discretionary entry review | Enabled; uses local Ollama. Failure policy is `continue`, so advisor unavailability is not necessarily fail-closed |
| `NewsTerminal` / `DeskManager` | Yahoo Finance RSS headlines written into shared `consensus.md` | Started by the core bot; feed availability is external and can degrade |
| `PostMortemAnalyzer` | Reviews losses and proposes `[SHADOW]` playbook rules | Module is invoked by the bot; generated rules require review and do not constitute validated learning |
| `USOptionsScanner` | Alpaca options-chain PCR/max-pain/unusual-activity confirmation | Enabled; cached for 300 seconds; configured to gate conflicting equity entries |
| Earnings calendar | Uses earnings dates to block entries around reports | Enabled; configured 2 days before and 1 day after |
| `RLExitAgent` | PPO exit recommendation | Model loads, but training data/validation limitations above apply |
| `NSETradingBot` and `FnOScanner` | Indian cash-equity mean reversion, daily trend-following, and F&O confirmation | Implemented, but **disabled in the actual LaunchAgent** (`NSE_ENABLED=false`); its log files do not mean this worker is currently running |
| TradingView webhook | Optional supplemental alert receiver | Enabled locally; public HTTPS route was unreachable at the readiness check, so external TradingView cloud delivery is not verified |
| `ShadowTrader` | Paper/simulated evaluation of candidate rules | Present in core bot; dashboard currently has no active shadow-strategy evidence |
| Distributed workers/services | Separate strategy, risk, market-data, broker, analyst, and position components | Experimental/unlaunched; active runtime remains centralized |
| `ResearchPipeline`, `StrategyEvolver`, `WeekendAnalyzer`, `UniversalSimulator` | Offline research, data harvest, simulated training, reporting | Research/experimental tools, not all connected to live execution |

NSE code configures 20 cash intraday symbols and 10 daily swing assignments (5 overlap, giving 25 unique cash names), plus eight unique F&O underlyings. Those are code/config universes, not active U.S. Alpaca symbols in the current LaunchAgent.

## 6. Data sources and datasets

### External source families

- **Alpaca:** U.S. bars/trading (Paper account), IEX feed, optional market stream, news client, and options chain.
- **Yahoo Finance (`yfinance`):** historical global basket, macro series (`^VIX`, `^TNX`, `^IRX`), earnings calendar, and fallback NSE quotes/bars.
- **NSE endpoints/libraries:** `nsepy` and `nsepython` where installed; `yfinance` fallback. NSE execution is disabled in the current service.
- **News RSS:** Google News RSS fallback for analyst sentiment and Yahoo Finance RSS for the shared news terminal.
- **TradingView:** webhook alert input; local receiver is present, public reachability failed at audit.

At this inventory level, there are **5 configured external source families** and **3 persisted local price-history files** listed below. If each endpoint/feed is counted separately, the number changes; the project does not have one canonical, versioned ML dataset.

### Local price history and training files observed

| File | Contents at audit | Notes |
|---|---|---|
| `runtime/global_data/daily_full.csv` | 6,905 date rows, 16 tickers × OHLCV (80 data columns); 2005-01-03 through 2026-09-06 | Fetched by `bot/data/global_harvester.py` from yfinance; multi-asset research data, not the live order feed |
| `runtime/global_data/intraday_1h.csv` | 27,527 timestamps, same 16 tickers × OHLCV; 2023-10-09 through 2026-09-06 | yfinance hourly research history; actual timestamps/data quality should be checked before training claims |
| `banknifty_5m.csv` | 3,225 Bank Nifty 5-minute bars; 2026-07-06 through 2026-09-02 | Local historical CSV; provenance/adjustment/independent validation metadata is not established in the file |
| `runtime/global_training_events.jsonl` | 5 event lines | Too small to be a meaningful training corpus; generated by a simulator and not a verified live-trade label set |
| `runtime/session_events*.jsonl` | 2,934 event lines across current/rotated session logs | Current ML trainer found 15 opens and 0 matched closes using its required `entry_order_id` pairing, so it cannot retrain from these events now |
| `session_events_pretrain.jsonl` | Historical pretraining event file exists at project root | The ML filter scans `runtime/session_events*.jsonl`; this root-level file is not in that scan path |

The live U.S. bot's current configured universe is nine symbols: SPY, QQQ, AAPL, TSLA, FISV, AMD, CB, CMCSA, and DIS. On the audit snapshot, the market was closed and runtime data status was `market_closed`; that is not evidence of fresh real-time streaming at that moment.

### Runtime records (rows/files currently on disk)

These are operational logs, not separate independent training datasets. Counts are raw rows and may include repeated sessions or legacy records.

| Artifact | Count |
|---|---:|
| `runtime/analyst_signals.csv` | 35,550 rows |
| `runtime/signals.csv` | 241 rows |
| `runtime/guardrails.csv` | 84 rows |
| `runtime/equity_curve.csv` | 16,167 rows |
| `runtime/nse_signals.csv` / `runtime/nse_trades.csv` | 286 / 81 rows |
| `runtime/optimizer_results.csv` | 962 rows |
| Persisted `runtime/backtests/run_*` folders | 1,100 runs |
| `runtime/reports/performance_*.json` | 26 reports |
| `runtime/aegis_memory.db` | 5 trade rows; 0 decisions, regime-history rows, playbook changes, or model-version rows |
| Root-level `trades.csv` / `daily_pnl.csv` | 8 / 17 rows; legacy summaries, not reconciled here to the broker ledger |

The downloaded/curated OHLCV files, live provider feeds, trade-event labels, and runtime logs are different data categories. Their row totals must not be added together and called one training-set size.

## 7. Risk management and execution controls

Current configured paper risk profile:

- Risk sizing target: **1% of equity per trade**, based on ATR distance; universal hard stop at **1× ATR(14)**.
- Daily new-entry stop: **3% drawdown**; stop after **3 consecutive losses**.
- Maximum **5 open positions**, **35% equity** notional per position, **80% gross exposure**, and **5% portfolio heat**.
- Additional sizing can be reduced for high-volatility regime, low analyst confidence, ML probability, Watchdog health, and Guardian risk state.
- ATR-volatility bands, 200-EMA alignment for momentum/trend, opening/closing time filters, correlation limits, earnings blackout, options disagreement, analyst/regime gates, and broker fill confirmation are implemented.
- Supervisor restart defaults include a 2-second backoff, at most 5 restarts, and pause-all-on-worker-failure.

These are intended controls, not a guarantee of maximum realized loss. Gaps, slippage, API/broker behavior, rejected/unfilled stops, and system/network failures can make actual outcomes worse than theoretical risk sizing.

## 8. Backtesting, model validation, and accuracy

- [`run_backtest.py`](run_backtest.py) accepts an OHLCV CSV and runs [`bot/backtest.py`](bot/backtest.py). However, that runner's `build_strategies()` currently registers only `RedGreenBreakoutStrategy` on `^NSEBANK` 5-minute bars. It does **not** replay the configured U.S. mean-reversion, momentum, and trend strategies.
- Backtest defaults include $100,000 starting equity and 2 basis points slippage. The latest `runtime/backtests/latest_summary.json` says **0 trades**, $0 realized P&L, and $100,000 ending equity. Therefore it is not evidence of live-strategy performance.
- `runtime/backtests/` has 1,100 saved run directories, but a high run count does not demonstrate independent data, correct strategy coverage, or out-of-sample validation.
- `bot/pipeline.py` labels itself a research/optimization pipeline, but its current scoring is explicitly mock/simulated and it writes a JSON result rather than executing the backtester or changing production config. Do not treat its displayed Sharpe as measured strategy performance.
- XGBoost artifacts are present, but no verified accuracy/calibration report, class balance report, train/test split, or sufficiently matched fresh outcome set was found. The persisted memory DB currently has zero regime-history and decision rows. The analyst's regime confidence and aggregate confidence are not calibrated win probabilities.
- The project therefore has **no defensible validated trade-accuracy percentage at this audit**. The 75% target cannot be guaranteed by setting a threshold; a high entry threshold may simply reduce trade count, and a strategy can still lose.

## 9. Dashboard, persistence, and tests

- Dashboard routes include `/api/snapshot`, `/api/runtime`, `/api/bars/<symbol>`, `/healthz`, and a logic-check endpoint. Runtime data is assembled from persistent bot state, event logs, signal/guardrail/equity histories, memory, and broker-facing state.
- `TradingMemory` stores trades, decisions, regime history, playbook changes, and model versions in SQLite. At this audit, the DB had five trade rows but no regime-history rows. That explains why a dashboard can display **UNKNOWN / “No confidence score recorded”** even though code to persist valid regime output exists: no persisted rows are currently available in that table.
- Latest runtime snapshot: mode PAPER; `runtime_status=running`; one U.S. worker heartbeat was current at `2026-10-12 05:41 UTC`, status `running`, zero restarts; nine symbols; market closed; zero locally persisted open positions; dashboard health HTTP 200.
- Test suite run for this report: **152 passed, 1 deprecation warning, 3 subtests passed** in 2.94 seconds.
- Read-only `run_bot.py --doctor` with the LaunchAgent's `NSE_ENABLED=false` reported **0 failures / 3 warnings**: experimental distributed modules not launched, Red/Green idle because it has no symbols, and TradingView public cloud route unreachable. It confirmed PAPER mode, loaded XGBoost artifact, local `qwen2.5`, Waitress installed, and healthy U.S. worker. No orders were submitted.

## 10. Important implementation and reproducibility gaps

1. **Backtest coverage mismatch:** research runner tests Red/Green Bank Nifty, not the live U.S. strategies.
2. **Regime-history visibility:** the DB has zero regime records despite the persistence hook; confidence should be treated as unavailable until valid rows appear.
3. **ML retraining labels:** current runtime event files have no matched closed-trade examples under the model trainer's pairing schema; the `run_global_training.py` custom-path call is also incompatible with `MLSignalFilter.train()` and falls back to the normal runtime-event scan.
4. **Model claims:** XGBoost, HMM, and PPO are present, but no independent walk-forward accuracy/calibration report substantiates a 75% win rate.
5. **Feature flags vs integration:** the multi-scale 3/4 confirmation flag is true, but the live loop does not call the confirmation engine. The distributed architecture is also not launched.
6. **External hooks:** TradingView callback is not reachable from the public cloud at audit; NSE is disabled by the service override; Red/Green has no configured symbol.
7. **Config/documentation drift:** momentum/trend docstrings list outdated instruments; `.env.example` defaults differ from the local `.env`; optimizer comments/defaults differ from the current local values; optional RL dependencies are installed in this environment but are not pinned in `requirements.txt`.
8. **Ledger consistency:** runtime memory, root-level CSVs, NSE logs, broker state, and session JSONL have different counts and are not one reconciled canonical historical dataset.

## 11. Key source files

- Runtime/config: [`config.py`](config.py), [`run_bot.py`](run_bot.py), [`bot/main.py`](bot/main.py), [`bot/supervisor.py`](bot/supervisor.py), [`bot/runtime_coordinator.py`](bot/runtime_coordinator.py)
- Strategies/risk: [`bot/strategies/mean_reversion.py`](bot/strategies/mean_reversion.py), [`bot/strategies/momentum_breakout.py`](bot/strategies/momentum_breakout.py), [`bot/strategies/trend_following.py`](bot/strategies/trend_following.py), [`bot/risk_manager.py`](bot/risk_manager.py), [`bot/market_filters.py`](bot/market_filters.py)
- Analyst/models: [`bot/analyst/analyst.py`](bot/analyst/analyst.py), [`bot/analyst/regime.py`](bot/analyst/regime.py), [`bot/analyst/aggregator.py`](bot/analyst/aggregator.py), [`bot/ml_filter.py`](bot/ml_filter.py), [`bot/optimizer.py`](bot/optimizer.py), [`bot/rl_agent.py`](bot/rl_agent.py)
- Data/research/UI: [`bot/data/global_harvester.py`](bot/data/global_harvester.py), [`bot/backtest.py`](bot/backtest.py), [`run_backtest.py`](run_backtest.py), [`dashboard.py`](dashboard.py), [`requirements.txt`](requirements.txt)

