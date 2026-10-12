#!/bin/bash
# start_services.sh - Start all trading bot services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

echo "Starting Aegis Trading Bot Services..."
echo "Working directory: $SCRIPT_DIR"

# Start health check service
echo "Starting health check server..."
$VENV_PYTHON -c "from bot.health_check import get_health_checker; h=get_health_checker(port=5001); h.start()" &
echo "Health check started on http://localhost:5001"

# Start Redis if not running
echo "Checking Redis..."
if ! redis-cli ping 2>/dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 1
fi
echo "Redis is running"

# Start individual services
echo "Starting services..."

# Position Manager
$VENV_PYTHON -m bot.distributed.position_manager &
echo "Position Manager started"

# Risk Engine
$VENV_PYTHON -m bot.distributed.risk_engine &
echo "Risk Engine started"

# Strategy Worker
$VENV_PYTHON -m bot.distributed.strategy_worker &
echo "Strategy Worker started"

# Market Data Feeder
$VENV_PYTHON -m bot.distributed.market_data_feeder &
echo "Market Data Feeder started"

# Broker Service
$VENV_PYTHON -m bot.distributed.broker_service &
echo "Broker Service started"

# Wait a moment for services to initialize
sleep 2

# Start Orchestrator
$VENV_PYTHON -m bot.distributed.orchestrator &
echo "Orchestrator started"

echo "All services started. Check logs for details."
echo "Health endpoint: http://localhost:5001/health"
echo "Metrics endpoint: http://localhost:5001/metrics"