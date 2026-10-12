"""Shared network safeguards for synchronous third-party API clients."""

from __future__ import annotations

from requests.adapters import HTTPAdapter


class DefaultTimeoutAdapter(HTTPAdapter):
    """Apply finite connect/read timeouts when an SDK omits them."""

    def __init__(self, timeout: float, **kwargs):
        self.timeout = (float(timeout), float(timeout))
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def configure_client_timeout(client, timeout: float) -> bool:
    """Install a bounded timeout on an alpaca-py requests-backed client.

    alpaca-py currently creates a private ``requests.Session`` without
    exposing a timeout setting. Mounting an adapter preserves its normal
    retry/auth behavior while ensuring a network outage cannot pin a worker
    forever. Return False when a future SDK implementation is not compatible.
    """
    session = getattr(client, "_session", None)
    if session is None or not hasattr(session, "mount"):
        return False
    session.mount("http://", DefaultTimeoutAdapter(timeout))
    session.mount("https://", DefaultTimeoutAdapter(timeout))
    return True
