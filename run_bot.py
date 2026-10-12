"""Supervised launcher for the coordinated US and NSE paper-trading runtime."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import signal
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import config
from bot.process_lock import RuntimeProcessLock
from bot.runtime_coordinator import RuntimeCoordinator
from bot.supervisor import RuntimeSupervisor, WorkerSpec


def _print_missing_dependency(exc: ModuleNotFoundError) -> None:
    missing_name = exc.name or "a required package"
    print(
        f"\nMissing Python dependency: {missing_name}.\n"
        "Install the project dependencies first:\n"
        f"  cd {config.ROOT_DIR}\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt"
    )


def _make_us_worker(coordinator: RuntimeCoordinator):
    def run(stop_event: threading.Event) -> None:
        from bot.main import TradingBot

        TradingBot(coordinator=coordinator).run_forever(stop_event=stop_event)

    return run


def _make_nse_worker(coordinator: RuntimeCoordinator):
    def run(stop_event: threading.Event) -> None:
        from bot.nse.nse_loop import NSETradingBot

        NSETradingBot(coordinator=coordinator).run_forever(stop_event=stop_event)

    return run


def _preload_ml_backend() -> None:
    """Load XGBoost before worker threads to avoid native-runtime crashes on macOS."""
    if not config.ML_FILTER_ENABLED:
        return

    try:
        # MLSignalFilter imports XGBoost. Import it on the main thread before
        # RuntimeSupervisor starts the US/NSE worker threads; loading its native
        # OpenMP runtime from a worker thread can terminate the process on macOS.
        from bot import ml_filter  # noqa: F401
    except Exception as exc:
        config.ML_FILTER_ENABLED = False
        print(f"[ML] Backend preload failed; continuing without ML filter: {exc}")


def _public_webhook_url_status(value: str) -> tuple[bool, str]:
    """Validate that a TradingView callback is a plausible public HTTPS URL."""
    if not value:
        return False, "not configured; cloud alerts cannot reach the local receiver"
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not host or parsed.username or parsed.password:
            return False, "must be an HTTPS URL without embedded credentials"
        if parsed.query or parsed.fragment:
            return False, "query strings and fragments are not allowed in the callback URL"
        if not parsed.path.rstrip("/").endswith("/api/v1/webhook/tradingview"):
            return False, "URL must end with /api/v1/webhook/tradingview"
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return False, "host resolves to a non-public IP address"
        if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(".local"):
            return False, "a local hostname is not publicly reachable"
    except ValueError:
        return False, "invalid URL"
    return True, "HTTPS callback URL configured (reachability is not verified by this check)"


def _webhook_secret_status(value: str) -> tuple[bool, str]:
    """Require a nontrivial passphrase before enabling an internet-facing hook."""
    if len(value) < 32:
        return False, "must be at least 32 characters before public webhook use (value redacted)"
    return True, f"configured ({len(value)} characters; value redacted)"


def _webhook_endpoint_status(value: str) -> tuple[bool, str]:
    """Make a safe GET probe; the POST-only webhook must answer with 405."""
    valid_url, detail = _public_webhook_url_status(value)
    if not valid_url:
        return False, detail
    request = urllib.request.Request(value, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:
        return False, f"public route unreachable ({type(exc).__name__})"
    if status == 405:
        return True, "public HTTPS route reachable; GET is rejected as expected"
    return False, f"unexpected HTTP {status} from the callback route"


def _run_readiness_check() -> int:
    """Read-only demo preflight; never starts a worker or submits an order."""
    from bot.runtime_coordinator import RuntimeCoordinator

    failures = 0
    warnings = 0

    def report(label: str, ok: bool, detail: str, *, warning: bool = False) -> None:
        nonlocal failures, warnings
        state = "OK" if ok else ("WARN" if warning else "FAIL")
        print(f"[{state}] {label}: {detail}")
        if not ok and warning:
            warnings += 1
        elif not ok:
            failures += 1

    mode = "PAPER" if config.ALPACA_PAPER else "LIVE"
    print(f"Read-only readiness check — mode={mode}, backend={config.BROKER_BACKEND}.")
    report(
        "paper demo mode",
        bool(config.ALPACA_PAPER),
        "paper trading is active" if config.ALPACA_PAPER else "LIVE mode is configured; switch to paper before a demo",
    )
    report(
        "distributed service modules",
        False,
        "experimental components are not launched; the compatibility orchestrator delegates to the supervised runtime",
        warning=True,
    )
    print("Optional feature configuration:")
    feature_flags = (
        ("analyst", getattr(config, "ANALYST_ENABLED", False)),
        ("ML filter", getattr(config, "ML_FILTER_ENABLED", False)),
        ("optimizer", getattr(config, "OPTIMIZER_ENABLED", False)),
        ("AI committee", getattr(config, "AI_COMMITTEE_ENABLED", False)),
        ("virtual brain", getattr(config, "AI_VIRTUAL_BRAIN_ENABLED", False)),
        ("watchdog", getattr(config, "WATCHDOG_ENABLED", False)),
        ("US options", getattr(config, "US_OPTIONS_ENABLED", False)),
        ("earnings filter", getattr(config, "EARNINGS_ENABLED", False)),
        ("red/green strategy", getattr(config, "RED_GREEN_BREAKOUT_ENABLED", False)),
        ("TradingView webhook", getattr(config, "TRADINGVIEW_ENABLED", False)),
        ("NSE worker", getattr(config, "NSE_ENABLED", False)),
    )
    for name, enabled in feature_flags:
        print(f"  {name}: {'enabled' if enabled else 'disabled'}")

    if config.RED_GREEN_BREAKOUT_ENABLED and not config.RED_GREEN_BREAKOUT_SYMBOLS:
        report(
            "red/green strategy",
            False,
            "enabled but idle until non-overlapping symbols are assigned",
            warning=True,
        )
    if config.TRADINGVIEW_ENABLED:
        report(
            "TradingView secret",
            *_webhook_secret_status(config.TRADINGVIEW_WEBHOOK_SECRET),
        )
        report(
            "TradingView public callback",
            *_public_webhook_url_status(config.TRADINGVIEW_PUBLIC_URL),
            warning=True,
        )
        report(
            "webhook bind address",
            config.TRADINGVIEW_WEBHOOK_HOST in {"127.0.0.1", "localhost", "::1"},
            f"{config.TRADINGVIEW_WEBHOOK_HOST} (loopback only is recommended behind a tunnel/proxy)",
            warning=True,
        )
        report(
            "TradingView cloud delivery",
            *_webhook_endpoint_status(config.TRADINGVIEW_PUBLIC_URL),
            warning=True,
        )

    report(
        "Waitress WSGI server",
        importlib.util.find_spec("waitress") is not None,
        "installed" if importlib.util.find_spec("waitress") else "missing; install requirements.txt",
    )

    if config.ML_FILTER_ENABLED:
        model_candidates = (
            Path(config.runtime_path("xgboost_model.json")),
            Path(config.ROOT_DIR) / "xgboost_model.json",
        )
        model_path = next((path for path in model_candidates if path.is_file()), None)
        try:
            import xgboost as xgb

            if model_path:
                model = xgb.Booster()
                model.load_model(str(model_path))
                report("ML model", True, f"loaded artifact {model_path.name}")
            else:
                report(
                    "ML model",
                    False,
                    "no artifact found; runtime training needs at least 10 paired closed trades",
                    warning=True,
                )
        except Exception as exc:
            report("ML model", False, f"could not load: {type(exc).__name__}: {exc}")

    if config.AI_COMMITTEE_ENABLED or config.AI_VIRTUAL_BRAIN_ENABLED:
        try:
            request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                models = json.loads(response.read().decode("utf-8")).get("models", [])
            names = [str(item.get("name", "")) for item in models]
            available = any(name.split(":", 1)[0] == "qwen2.5" for name in names)
            report(
                "local Ollama model",
                available,
                "qwen2.5 available" if available else "service responds, but qwen2.5 is not listed",
                warning=not available,
            )
        except Exception as exc:
            report("local Ollama model", False, f"unreachable: {type(exc).__name__}", warning=True)

    try:
        runtime = RuntimeCoordinator().snapshot()
        workers = runtime.get("workers", {})
        if workers:
            detail = ", ".join(
                f"{name}={'healthy' if worker.get('healthy') else worker.get('status', 'unknown')}"
                for name, worker in workers.items()
            )
            active_runtime = runtime.get("runtime_status") == "running"
            report(
                "running workers",
                bool(runtime.get("healthy")),
                detail,
                warning=not active_runtime,
            )
            if not runtime.get("entries_allowed"):
                reason = runtime.get("system_pause_reason") or runtime.get("operator_pause_reason") or "entry gate paused"
                intentional_operator_pause = bool(runtime.get("operator_entry_pause")) and not bool(
                    runtime.get("system_entry_pause")
                )
                report("entry gate", False, str(reason), warning=intentional_operator_pause)
        else:
            report("running workers", False, "runtime is not started", warning=True)
    except Exception as exc:
        report("running workers", False, f"status unavailable: {type(exc).__name__}", warning=True)

    print(f"Readiness summary: {failures} failure(s), {warnings} warning(s). No orders were submitted.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised Alpaca/NSE paper trading runtime")
    parser.add_argument(
        "--screener",
        action="store_true",
        help="Run the automated market screener before launching workers.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration without starting workers or touching brokers.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run a read-only readiness check for optional services and worker health.",
    )
    args = parser.parse_args()

    try:
        config.validate_project_root(config.BOT_LAUNCH_COMMAND)
        config.validate_broker_backend()
        config.validate_alpaca_credentials(command_hint=config.BOT_LAUNCH_COMMAND)

        if args.doctor:
            return _run_readiness_check()

        if args.check:
            enabled = ["us"] + (["nse"] if config.NSE_ENABLED else [])
            print(
                "Configuration valid. "
                f"mode={'PAPER' if config.ALPACA_PAPER else 'LIVE'}, "
                f"backend={config.BROKER_BACKEND}, workers={', '.join(enabled)}."
            )
            return 0

        _preload_ml_backend()

        if args.screener:
            from bot.screener import run_screener

            run_screener()

        worker_names = ["us"] + (["nse"] if config.NSE_ENABLED else [])
        coordinator = RuntimeCoordinator()
        stop_event = threading.Event()

        def request_shutdown(_signum, _frame):
            stop_event.set()

        previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
        previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
        specs = [WorkerSpec("us", _make_us_worker(coordinator))]
        if config.NSE_ENABLED:
            specs.append(WorkerSpec("nse", _make_nse_worker(coordinator)))

        with RuntimeProcessLock():
            coordinator.begin_session(
                mode="PAPER" if config.ALPACA_PAPER else "LIVE",
                workers=worker_names,
            )
            print(
                "Starting supervised trading runtime "
                f"({'paper' if config.ALPACA_PAPER else 'live'} mode): "
                f"{', '.join(worker_names)}."
            )
            try:
                RuntimeSupervisor(coordinator, specs, stop_event=stop_event).run()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
                signal.signal(signal.SIGINT, previous_sigint)
        return 0
    except ModuleNotFoundError as exc:
        _print_missing_dependency(exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        # The supervisor's finally block persists stopped worker state and
        # joins non-daemon workers before the process exits.
        print("\nShutting down supervised runtime.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
