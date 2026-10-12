#!/usr/bin/env python3
"""Health check endpoint for Phase 3 deployments"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import redis


class HealthChecker:
    """Health check endpoint for each microservice."""
    
    def __init__(self, port: int = 5001, redis_host: str = "localhost", redis_port: int = 6379):
        self.port = port
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started_at = time.time()
        self._checks: dict[str, bool] = {}
        self._lock = threading.Lock()
    
    def set_check(self, name: str, status: bool) -> None:
        """Register a health check result."""
        with self._lock:
            self._checks[name] = status
    
    def start(self) -> None:
        """Start the health check HTTP server in background."""
        class HealthHandler(BaseHTTPRequestHandler):
            checker = self
            
            def log_message(self, format, *args):
                pass  # Suppress logging
            
            def do_GET(self):
                if self.path == "/health":
                    self._handle_health()
                elif self.path == "/healthz":
                    self._handle_health()
                else:
                    self.send_error(404)
            
            def _handle_health(self):
                checks = self.checker._checks.copy()
                checks["uptime"] = time.time() - self.checker._started_at
                
                # Test Redis connectivity
                try:
                    self.checker.redis.ping()
                    checks["redis"] = "ok"
                except Exception:
                    checks["redis"] = "error"
                
                all_healthy = all(
                    v in ("ok", True) or (isinstance(v, str) and v.startswith("ok"))
                    for k, v in checks.items() if k not in ("uptime", "redis")
                )
                
                status_code = 200 if all_healthy else 503
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "healthy" if all_healthy else "unhealthy",
                    "checks": checks
                }).encode())
        
        self._server = HTTPServer(("0.0.0.0", self.port), HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()


# Global health checker instance
health_checker: HealthChecker | None = None


def get_health_checker(port: int = 5001) -> HealthChecker:
    """Get or create the global health checker."""
    global health_checker
    if health_checker is None:
        health_checker = HealthChecker(port=port)
    return health_checker