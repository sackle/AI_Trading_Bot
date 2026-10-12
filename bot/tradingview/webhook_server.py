import logging
import hmac
import math
import re
from queue import Full
from flask import Flask, request, jsonify
import config
from bot.tradingview import tv_signal_queue

logger = logging.getLogger("tradingview.webhook")

app = Flask(__name__)

# Valid TradingView Egress IP Addresses (as of 2024-2025)
TV_IPS = {
    "52.89.214.238",
    "34.212.75.30",
    "54.218.53.128",
    "52.32.178.7"
}

@app.route("/api/v1/webhook/tradingview", methods=["POST"])
def tradingview_webhook():
    if not config.TRADINGVIEW_ENABLED:
        return jsonify({"error": "TradingView integration is disabled"}), 403

    # (Optional) Verify IP. Commented out for local testing via ngrok.
    # client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    # if client_ip and client_ip.split(",")[0].strip() not in TV_IPS:
    #     logger.warning(f"Unauthorized IP attempted webhook: {client_ip}")
    #     return jsonify({"error": "Unauthorized IP"}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Authenticate via shared secret passphrase
    expected_secret = str(config.TRADINGVIEW_WEBHOOK_SECRET or "")
    passphrase = payload.get("passphrase")
    if len(expected_secret) < 32:
        logger.error("TradingView webhook requires a secret of at least 32 characters.")
        return jsonify({"error": "Webhook authentication is not securely configured"}), 503
    if not isinstance(passphrase, str) or not hmac.compare_digest(passphrase, expected_secret):
        logger.warning("Webhook authentication failed: invalid passphrase.")
        return jsonify({"error": "Unauthorized"}), 401

    # Extract critical fields
    raw_symbol = payload.get("symbol")
    action = payload.get("action", "")
    
    if not isinstance(raw_symbol, str) or not isinstance(action, str) or not raw_symbol.strip() or not action.strip():
        return jsonify({"error": "Missing symbol or action"}), 400

    raw_symbol = raw_symbol.strip()
    action = action.lower().strip()
    if action not in {"buy", "long", "sell", "short", "cover", "close"}:
        return jsonify({"error": "Unsupported action"}), 400

    # Map TradingView ticker (e.g. "NASDAQ:AAPL") to internal symbol ("AAPL")
    symbol = config.TRADINGVIEW_SYMBOL_MAP.get(raw_symbol, raw_symbol)
    # If the symbol has a colon but wasn't mapped, strip the exchange prefix
    if ":" in symbol:
        symbol = symbol.split(":")[1]
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9.^/_-]{1,32}", symbol):
        return jsonify({"error": "Invalid symbol"}), 400

    raw_price = payload.get("price", 0.0)
    try:
        price = float(raw_price or 0.0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid price"}), 400
    if not math.isfinite(price) or price < 0:
        return jsonify({"error": "Invalid price"}), 400

    raw_qty = payload.get("qty")
    if raw_qty is not None:
        try:
            qty = float(raw_qty)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid quantity"}), 400
        if not math.isfinite(qty) or qty <= 0:
            return jsonify({"error": "Invalid quantity"}), 400

    strategy = payload.get("strategy", "tradingview_alert")
    if not isinstance(strategy, str) or not strategy.strip() or len(strategy) > 80:
        return jsonify({"error": "Invalid strategy"}), 400

    signal_data = {
        "symbol": symbol,
        "action": action,
        "strategy": strategy.strip(),
        "price": price,
        "stop_loss": payload.get("stop_loss"),
        "take_profit": payload.get("take_profit"),
        "qty": raw_qty
    }

    # Publish to Redis bus if available (Distributed mode)
    global _bus
    if _bus:
        signal_event = {
            "symbol": symbol,
            "strategy": strategy.strip(),
            "signal": action,
            "confidence": 1.0,  # Trust TV signals by default
            "price": price,
            "timestamp": payload.get("timestamp", __import__("pandas").Timestamp.now().isoformat()),
            "atr": 1.0,
            "is_webhook": True,
        }
        try:
            _bus.publish(f"signal.{symbol}", signal_event)
        except Exception as exc:
            logger.error("Failed to publish TradingView signal: %s", exc)
            return jsonify({"error": "Signal bus unavailable"}), 503
        logger.info(f"Published TradingView signal to bus: {action.upper()} {symbol}")
    else:
        # Queue is bounded so a slow trading worker cannot exhaust memory.
        try:
            tv_signal_queue.put_nowait(signal_data)
        except Full:
            return jsonify({"error": "Trading worker is busy; retry the alert"}), 503
        logger.info(f"Enqueued TradingView signal locally: {action.upper()} {symbol}")

    return jsonify({"status": "received", "symbol": symbol}), 200

_bus = None

def start_webhook_server(bus=None):
    """Run the webhook WSGI server; caller may place this in a daemon thread."""
    global _bus
    _bus = bus
    if config.TRADINGVIEW_ENABLED:
        logger.info(f"Starting TradingView Webhook server on port {config.TRADINGVIEW_WEBHOOK_PORT}...")
        from waitress import serve

        serve(
            app,
            host=config.TRADINGVIEW_WEBHOOK_HOST,
            port=config.TRADINGVIEW_WEBHOOK_PORT,
            threads=8,
            max_request_body_size=64 * 1024,
        )
