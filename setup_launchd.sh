#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLIST_PATH="${HOME}/Library/LaunchAgents/com.sarthak.alpaca_bot.plist"
LABEL="com.sarthak.alpaca_bot"
LEGACY_LABEL="com.aegis.tradingbot"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
RUNTIME_DIR="${PROJECT_DIR}/runtime"
DOMAIN="gui/$(id -u)"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Project virtualenv Python is missing: ${PYTHON}" >&2
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents" "${RUNTIME_DIR}"
cd "${PROJECT_DIR}"

# This launch agent is intentionally PAPER-only and US/Alpaca-only. Use
# Alpaca's own market clock in the trading worker so holidays, early closes,
# and daylight-saving changes do not depend on a hand-maintained schedule.
if ! ALPACA_PAPER=true BROKER_BACKEND=alpaca NSE_ENABLED=false \
    "${PYTHON}" run_bot.py --check; then
    echo "Paper-mode readiness check failed; launch agent was not changed." >&2
    exit 1
fi

cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${PROJECT_DIR}/run_bot.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>${RUNTIME_DIR}/launchd.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${RUNTIME_DIR}/launchd.stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${PROJECT_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>ALPACA_PAPER</key>
        <string>true</string>
        <key>BROKER_BACKEND</key>
        <string>alpaca</string>
        <key>NSE_ENABLED</key>
        <string>false</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>PYTHONFAULTHANDLER</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLIST

plutil -lint "${PLIST_PATH}"

# Unload and persistently disable the old experimental orchestrator so it
# cannot race the supervised Alpaca runner or repeatedly relaunch itself.
launchctl bootout "${DOMAIN}/${LEGACY_LABEL}" 2>/dev/null || true
launchctl disable "${DOMAIN}/${LEGACY_LABEL}" 2>/dev/null || true

# Replace the old shell-script-based agent. The direct venv invocation avoids
# launchd's previous EPERM failure while preserving restart supervision.
launchctl bootout "${DOMAIN}" "${PLIST_PATH}" 2>/dev/null || true
launchctl enable "${DOMAIN}/${LABEL}"
launchctl bootstrap "${DOMAIN}" "${PLIST_PATH}"

echo "Installed ${LABEL} in ${DOMAIN}."
echo "The supervised US Alpaca worker is PAPER-only; it stays idle while Alpaca reports the market closed."
echo "NSE is disabled for this launch agent. Logs: ${RUNTIME_DIR}/launchd.{stdout,stderr}.log"
