#!/bin/bash
# stop_services.sh - Stop all trading bot services

echo "Stopping Aegis Trading Bot Services..."

# Kill all bot processes
pkill -f "bot.distributed" 2>/dev/null
pkill -f "health_check" 2>/dev/null

# Kill health check server (port 8080)
lsof -ti:8080 | xargs kill -9 2>/dev/null

echo "All services stopped"