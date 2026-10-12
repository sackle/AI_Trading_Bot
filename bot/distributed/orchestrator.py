"""Compatibility entrypoint for the supervised trading runtime.

The former implementation launched independent Redis services whose signal,
risk, position, and broker paths were not integrated end-to-end. Starting that
partial fan-out could create duplicate workers and crash loops. This module now
delegates to the supported supervisor in :mod:`run_bot`; the experimental
distributed service modules remain available for development, but are not
launched as a trading system.

Usage::

    python -m bot.distributed.orchestrator
    python -m bot.distributed.orchestrator --check
    python -m bot.distributed.orchestrator --dry-run
"""
from __future__ import annotations

import argparse
import sys

from run_bot import _run_readiness_check, main as supervised_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aegis supervised trading runtime (compatibility entrypoint)")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="Validate configuration without starting workers")
    checks.add_argument("--dry-run", action="store_true", help="Run the read-only service readiness check")
    parser.add_argument("--screener", action="store_true", help="Run the market screener before launching workers")
    parser.add_argument(
        "--services",
        nargs="+",
        help="Deprecated: independent distributed services are not a supported execution mode",
    )
    args = parser.parse_args(argv)

    if args.services:
        parser.error(
            "--services is no longer supported: use the coordinated supervised runtime instead"
        )
    if args.dry_run:
        return _run_readiness_check()

    # run_bot.main parses process argv. Forward only the supported options and
    # restore argv afterward for callers that invoke this entrypoint in-process.
    forwarded = []
    if args.check:
        forwarded.append("--check")
    if args.screener:
        forwarded.append("--screener")
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *forwarded]
    try:
        return supervised_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
