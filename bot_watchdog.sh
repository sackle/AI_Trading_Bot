#!/bin/bash
cd /Users/sarthak/Downloads/alpaca_trading_bot
if ! pgrep -f "run_bot.py" > /dev/null
then
    source .venv/bin/activate
    nohup python run_bot.py >> runtime/bot.log 2>&1 &
fi
