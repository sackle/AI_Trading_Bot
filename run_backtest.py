#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bot.backtest import BacktestRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Alpaca bot strategies in historical backtest/replay mode.")
    parser.add_argument("--data", required=True, help="Path to CSV data with OHLCV bars.")
    parser.add_argument("--symbol", default=None, help="Default symbol when the CSV has no symbol column.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--equity", type=float, default=100000.0, help="Starting equity.")
    parser.add_argument("--replay", action="store_true", help="Print the event-by-event replay log.")
    args = parser.parse_args()

    runner = BacktestRunner(
        data_path=args.data,
        default_symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.equity,
        replay=args.replay,
    )
    result = runner.run()

    print("Backtest complete.")
    print(f"Output directory: {result.output_dir}")
    print(f"Ending equity: ${result.summary['ending_equity']:.2f}")
    print(f"Realized P&L: ${result.summary['realized_pnl']:.2f}")
    print(f"Trades: {result.summary['total_trades']} | Win rate: {result.summary['win_rate']:.1f}%")
    print(f"Max drawdown: ${result.summary['max_drawdown']:.2f}")

    if args.replay and result.replay_lines:
        print()
        print("Replay")
        for line in result.replay_lines:
            print(line)


if __name__ == "__main__":
    main()
