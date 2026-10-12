"""Friendly launcher for the local Flask dashboard."""

from __future__ import annotations

import sys

import config


def _print_missing_dependency(exc: ModuleNotFoundError):
    missing_name = exc.name or "a required package"
    print(
        f"\nMissing Python dependency: {missing_name}.\n"
        "Install the project dependencies first:\n"
        f"  cd {config.ROOT_DIR}\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt"
    )


def main() -> int:
    try:
        config.validate_project_root(config.DASHBOARD_LAUNCH_COMMAND)

        import dashboard

        host = dashboard.DEFAULT_DASHBOARD_HOST
        requested_port = dashboard.parse_dashboard_port(dashboard.os.getenv("DASHBOARD_PORT"))
        selected_port = dashboard.choose_dashboard_port(host=host, preferred_port=requested_port)
        print(dashboard.format_dashboard_start_message(host, requested_port, selected_port))
        from waitress import serve

        serve(dashboard.app, host=host, port=selected_port, threads=8)
        return 0
    except ModuleNotFoundError as exc:
        _print_missing_dependency(exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
