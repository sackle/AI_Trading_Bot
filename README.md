# Aegis Quantitative Trading System

Aegis is a Python paper-trading project with an Alpaca-backed U.S. trading loop, strategy and risk controls, a local operations dashboard, and optional research/analysis components. It is an educational/software project, not financial advice. Trading can lose money; paper results do not guarantee live performance.

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the audited architecture, strategy rules, bot inventory, data sources, validation status, and known limitations. See [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for the demonstration checklist.

## Safe local setup

Tested with Python 3.13. Use an **Alpaca Paper** account for development and demonstration.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally and enter your Alpaca **paper** API credentials. Never commit `.env`, paste credentials into source files, or switch to live mode for a demo. The repository ignores `.env`, local runtime state, generated logs/data, and trained model artifacts.

Run the read-only readiness check before starting the bot:

```bash
python run_bot.py --check
python run_bot.py --doctor
```

Start the paper bot and dashboard in separate terminals:

```bash
python run_bot.py
python run_dashboard.py
```

The dashboard is served locally (the configured default is `http://127.0.0.1:5012/`). The U.S. worker uses Alpaca's market clock and remains idle when the market is closed. Do not expose the dashboard or webhook publicly without reviewing authentication, network, and webhook-secret settings.

## Tests

```bash
python -m pytest -q
```

The audited workspace passed 152 tests on 12 October 2026. Runtime checks, account connectivity, optional integrations, and tests do not establish trading profitability or a validated win rate.

## Data and generated files

Market history, account/session logs, runtime databases, CSV exports, backtest outputs, and downloaded research datasets are intentionally local and excluded from Git. Supply authorized data and credentials in your own environment. The model artifacts are generated separately; without them, optional model behavior may fall back or be unavailable.

