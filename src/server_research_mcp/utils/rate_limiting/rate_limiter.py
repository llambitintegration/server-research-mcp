"""Core rate limiting functionality."""

import time
import logging
import threading
from typing import Optional, Dict, Any, Callable, TypeVar, cast
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Request limits
    max_requests_per_minute: int = 10
    max_requests_per_hour: int = 100
    
    # Timing controls
    min_request_interval: float = 1.0  # Minimum seconds between requests
    
    # Retry configuration
    max_retries: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter_factor: float = 0.1
    
    # Error detection
    rate_limit_error_keywords: list[str] = field(default_factory=lambda: [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "quota exceeded",
        "throttled",
        "try again later"
    ])
    
    # Logging
    log_rate_limits: bool = True
    

class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Thread-safe rate limiter with multiple time windows."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.Lock()
        
        # Request tracking
        self._minute_requests: deque = deque()
        self._hour_requests: deque = deque()
        self._last_request_time: Optional[float] = None
        
        # Retry tracking
        self._retry_counts: Dict[str, int] = defaultdict(int)
        self._last_retry_times: Dict[str, float] = defaultdict(float)
    
    def check_rate_limit(self, identifier: str = "default") -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed under rate limits.
        
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        with self._lock:
            now = time.time()
            
            # Clean old requests
            self._clean_old_requests(now)
            
            # Check minimum interval
            if self._last_request_time:
                time_since_last = now - self._last_request_time
                if time_since_last < self.config.min_request_interval:
                    wait_time = self.config.min_request_interval - time_since_last
                    return False, wait_time
            
            # Check minute limit
            if len(self._minute_requests) >= self.config.max_requests_per_minute:
                oldest_minute = self._minute_requests[0]
                wait_time = 60 - (now - oldest_minute) + 0.1
                return False, wait_time
            
            # Check hour limit
            if len(self._hour_requests) >= self.config.max_requests_per_hour:
                oldest_hour = self._hour_requests[0]
                wait_time = 3600 - (now - oldest_hour) + 0.1
                return False, wait_time
            
            return True, None
    
    def record_request(self, identifier: str = "default"):
        """Record a successful request."""
        with self._lock:
            now = time.time()
            self._minute_requests.append(now)
            self._hour_requests.append(now)
            self._last_request_time = now
            self._retry_counts[identifier] = 0
            
            if self.config.log_rate_limits:
                logger.debug(f"Request recorded for {identifier}. "
                           f"Minute: {len(self._minute_requests)}/{self.config.max_requests_per_minute}, "
                           f"Hour: {len(self._hour_requests)}/{self.config.max_requests_per_hour}")
    
    def record_error(self, identifier: str = "default", error: Exception = None) -> float:
        """
        Record an error and calculate retry delay.
        
        Returns:
            Retry delay in seconds
        """
        with self._lock:
            self._retry_counts[identifier] += 1
            retry_count = self._retry_counts[identifier]
            
            # Check if it's a rate limit error
            is_rate_limit = self._is_rate_limit_error(error)
            
            # Calculate base delay
            if is_rate_limit and hasattr(error, 'retry_after') and error.retry_after:
                base_delay = error.retry_after
            else:
                base_delay = min(
                    self.config.initial_retry_delay * (self.config.backoff_factor ** (retry_count - 1)),
                    self.config.max_retry_delay
                )
            
            # Add jitter
            import random
            jitter = base_delay * self.config.jitter_factor * (2 * random.random() - 1)
            delay = max(0.1, base_delay + jitter)
            
            self._last_retry_times[identifier] = time.time()
            
            if self.config.log_rate_limits:
                logger.warning(f"Error recorded for {identifier}. "
                             f"Retry #{retry_count}, delay: {delay:.2f}s. "
                             f"Error: {str(error)}")
            
            return delay
    
    def should_retry(self, identifier: str = "default") -> bool:
        """Check if retry should be attempted."""
        return self._retry_counts[identifier] < self.config.max_retries
    
    def reset_retries(self, identifier: str = "default"):
        """Reset retry counter for identifier."""
        with self._lock:
            self._retry_counts[identifier] = 0
    
    def _clean_old_requests(self, now: float):
        """Remove requests outside time windows."""
        # Clean minute window
        minute_cutoff = now - 60
        while self._minute_requests and self._minute_requests[0] < minute_cutoff:
            self._minute_requests.popleft()
        
        # Clean hour window
        hour_cutoff = now - 3600
        while self._hour_requests and self._hour_requests[0] < hour_cutoff:
            self._hour_requests.popleft()
    
    def _is_rate_limit_error(self, error: Optional[Exception]) -> bool:
        """Check if error is a rate limit error."""
        if not error:
            return False
        
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in self.config.rate_limit_error_keywords)
    
    def wait_if_needed(self, identifier: str = "default"):
        """Wait if rate limit requires it."""
        allowed, wait_time = self.check_rate_limit(identifier)
        if not allowed and wait_time:
            if self.config.log_rate_limits:
                logger.info(f"Rate limit reached for {identifier}. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)


def rate_limited(rate_limiter: RateLimiter, identifier: Optional[str] = None):
    """Decorator to add rate limiting to functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Use function name as identifier if not provided
            func_id = identifier or func.__name__
            
            # Wait if rate limited
            rate_limiter.wait_if_needed(func_id)
            
            # Try to execute with retries
            last_error = None
            while True:
                try:
                    # Check rate limit
                    allowed, wait_time = rate_limiter.check_rate_limit(func_id)
                    if not allowed:
                        if wait_time:
                            time.sleep(wait_time)
                        continue
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record success
                    rate_limiter.record_request(func_id)
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    # Check if we should retry
                    if not rate_limiter.should_retry(func_id):
                        logger.error(f"Max retries exceeded for {func_id}")
                        raise
                    
                    # Calculate retry delay
                    delay = rate_limiter.record_error(func_id, e)
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # Should never reach here, but for type safety
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected rate limiting state")
        
        return cast(Callable[..., T], wrapper)
    
    return decorator


# Global rate limiters for different services
_rate_limiters: Dict[str, RateLimiter] = {}


def create_rate_limiter(
    name: str,
    config: Optional[RateLimitConfig] = None,
    **kwargs
) -> RateLimiter:
    """Create or get a named rate limiter."""
    if name not in _rate_limiters:
        if config is None:
            config = RateLimitConfig(**kwargs)
        _rate_limiters[name] = RateLimiter(config)
    
    return _rate_limiters[name]


def get_rate_limiter(name: str) -> Optional[RateLimiter]:
    """Get a named rate limiter if it exists."""
    return _rate_limiters.get(name)
