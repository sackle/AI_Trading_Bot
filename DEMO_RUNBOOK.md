# Paper-trading demo runbook

This project is configured for a paper-mode demonstration. It does not prove
profitability, and live trading must remain disabled unless credentials,
controls, and deployment are reviewed separately.

## Before the demo

From the project directory, activate the virtual environment and run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_bot.py --check
python run_bot.py --doctor
python -m pytest -q
```

`--doctor` is read-only: it checks optional services, the ML artifact, local
Ollama, and the last supervised-worker health without submitting orders. Fix
any `FAIL` before presenting. A warning means an explicitly optional feature
or external integration is not fully available.

Use `python run_bot.py` as the supported runtime. The older independent
`bot.distributed` service modules remain experimental and are not launched.
For compatibility, `python -m bot.distributed.orchestrator` now delegates to
the same supervised runtime; its old `--services` fan-out mode is rejected.
The readiness check reports that the separate service modules are not part of
the supported execution path.

## Start the demo

Use two terminals:

```bash
python run_bot.py
```

```bash
python run_dashboard.py
```

The bot should report `PAPER` mode. Open the local dashboard URL printed by the
dashboard launcher and check `/healthz`; it should return HTTP 200 while the
required workers are healthy. Do not present a stale health snapshot as a live
bot.

## Integration limits to disclose

- TradingView's cloud service needs a public HTTPS URL on port 443. Configure
  `TRADINGVIEW_PUBLIC_URL` with the full callback URL, and keep the receiver
  bound to loopback via `TRADINGVIEW_WEBHOOK_HOST=127.0.0.1` behind an approved
  TLS reverse proxy or tunnel. Do not expose the local dev machine or put
  Alpaca/broker credentials in alert text; the webhook passphrase is the only
  credential the alert needs. A temporary tunnel URL changes when its process
  stops and must be updated in the TradingView alert. The active demo uses a
  temporary Cloudflare Quick Tunnel; it is for development/demo use only and
  has no uptime guarantee. In the TradingView alert body, set `passphrase` to
  the local `.env` value of `TRADINGVIEW_WEBHOOK_SECRET`. Generate that value
  with `openssl rand -hex 32`; the receiver rejects secrets shorter than 32
  characters.
- Red/green breakout remains idle until `RED_GREEN_BREAKOUT_SYMBOLS` contains
  an operator-selected symbol that is not already owned by another strategy.
- The ML retrainer needs at least 10 paired, closed paper trades with both win
  and loss outcomes. The checked-in model can provide inference before then;
  the retrainer must not invent labels or training rows.
- No entry is guaranteed. Market-hours, broker availability, risk gates, model
  decisions, and valid data can all correctly result in no trades.
