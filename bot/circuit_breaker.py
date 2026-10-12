#!/usr/bin/env python3
"""Circuit breaker pattern for Phase 3 reliability"""

import time
import threading
from enum import Enum
from typing import Callable, TypeVar, Any

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting external dependencies.
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30)
        
        @breaker.decorate
        def call_external_api():
            ...
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 30.0,
        expected_exception: tuple[type[Exception], ...] = (Exception,),
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_time: float | None = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._failure_time is None or time.time() - self._failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                self._state = CircuitState.HALF_OPEN
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def decorate(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for function-level circuit breaker."""
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self._failure_count = 0
        self._failure_time = None
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        if self._failure_time is None:
            self._failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._failure_time = None