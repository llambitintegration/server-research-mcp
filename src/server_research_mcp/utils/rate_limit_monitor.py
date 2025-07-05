"""
Rate Limit Monitoring Module

Provides comprehensive statistics collection and monitoring for rate limiting events.
Monkey-patches RateLimiter methods to collect statistics and provides summary reporting.
"""

import atexit
import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps
import threading

from .rate_limiting import RateLimiter, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitStats:
    """Statistics for a specific rate limiter identifier."""
    identifier: str
    total_requests: int = 0
    rate_limited_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    min_wait_time: float = float('inf')
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_request(self, wait_time: float = 0.0, success: bool = True, error_type: Optional[str] = None):
        """Record a request event."""
        self.total_requests += 1
        self.request_times.append(time.time())
        
        if wait_time > 0:
            self.rate_limited_requests += 1
            self.total_wait_time += wait_time
            self.max_wait_time = max(self.max_wait_time, wait_time)
            self.min_wait_time = min(self.min_wait_time, wait_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] += 1
    
    def add_retry(self):
        """Record a retry event."""
        self.total_retries += 1
    
    @property
    def rate_limited_percentage(self) -> float:
        """Calculate percentage of requests that were rate limited."""
        if self.total_requests == 0:
            return 0.0
        return (self.rate_limited_requests / self.total_requests) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time for rate limited requests."""
        if self.rate_limited_requests == 0:
            return 0.0
        return self.total_wait_time / self.rate_limited_requests
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate current requests per minute based on recent activity."""
        if not self.request_times:
            return 0.0
        
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t <= 60]
        return len(recent_requests)


class RateLimitMonitor:
    """Global rate limit monitoring system."""
    
    def __init__(self):
        self.stats: Dict[str, RateLimitStats] = {}
        self.enabled = True
        self._lock = threading.Lock()
        self._original_methods = {}
        self._patched = False
        
        # Register cleanup on exit
        atexit.register(self.print_summary)
    
    def enable(self):
        """Enable monitoring and apply monkey patches."""
        if not self._patched:
            self._apply_monkey_patches()
            self._patched = True
        self.enabled = True
        logger.info("Rate limit monitoring enabled")
    
    def disable(self):
        """Disable monitoring."""
        self.enabled = False
        logger.info("Rate limit monitoring disabled")
    
    def get_stats(self, identifier: str) -> RateLimitStats:
        """Get or create stats for an identifier."""
        with self._lock:
            if identifier not in self.stats:
                self.stats[identifier] = RateLimitStats(identifier)
            return self.stats[identifier]
    
    def record_request(self, identifier: str, wait_time: float = 0.0, success: bool = True, error_type: Optional[str] = None):
        """Record a request event."""
        if not self.enabled:
            return
        
        stats = self.get_stats(identifier)
        stats.add_request(wait_time, success, error_type)
        
        if wait_time > 0:
            logger.debug(f"Rate limit wait recorded: {identifier} waited {wait_time:.2f}s")
    
    def record_retry(self, identifier: str):
        """Record a retry event."""
        if not self.enabled:
            return
        
        stats = self.get_stats(identifier)
        stats.add_retry()
        logger.debug(f"Retry recorded for {identifier}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        with self._lock:
            summary = {
                "total_identifiers": len(self.stats),
                "overall_stats": {
                    "total_requests": sum(s.total_requests for s in self.stats.values()),
                    "total_rate_limited": sum(s.rate_limited_requests for s in self.stats.values()),
                    "total_retries": sum(s.total_retries for s in self.stats.values()),
                    "total_wait_time": sum(s.total_wait_time for s in self.stats.values()),
                },
                "identifier_stats": {}
            }
            
            # Calculate overall percentages
            total_requests = summary["overall_stats"]["total_requests"]
            if total_requests > 0:
                summary["overall_stats"]["rate_limited_percentage"] = (
                    summary["overall_stats"]["total_rate_limited"] / total_requests
                ) * 100
            else:
                summary["overall_stats"]["rate_limited_percentage"] = 0.0
            
            # Individual identifier stats
            for identifier, stats in self.stats.items():
                summary["identifier_stats"][identifier] = {
                    "total_requests": stats.total_requests,
                    "rate_limited_requests": stats.rate_limited_requests,
                    "rate_limited_percentage": stats.rate_limited_percentage,
                    "success_rate": stats.success_rate,
                    "total_retries": stats.total_retries,
                    "average_wait_time": stats.average_wait_time,
                    "max_wait_time": stats.max_wait_time,
                    "requests_per_minute": stats.requests_per_minute,
                    "error_types": dict(stats.error_types)
                }
            
            return summary
    
    def print_summary(self):
        """Print a formatted summary of rate limiting statistics."""
        if not self.stats:
            return
        
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("RATE LIMITING SUMMARY")
        print("="*80)
        
        # Overall stats
        overall = summary["overall_stats"]
        print(f"Total Requests: {overall['total_requests']}")
        print(f"Rate Limited: {overall['total_rate_limited']} ({overall['rate_limited_percentage']:.1f}%)")
        print(f"Total Retries: {overall['total_retries']}")
        print(f"Total Wait Time: {overall['total_wait_time']:.1f}s")
        
        if overall['total_rate_limited'] > 0:
            avg_wait = overall['total_wait_time'] / overall['total_rate_limited']
            print(f"Average Wait Time: {avg_wait:.2f}s")
        
        print("\nPER-IDENTIFIER BREAKDOWN:")
        print("-" * 80)
        
        # Sort identifiers by total requests
        sorted_identifiers = sorted(
            summary["identifier_stats"].items(),
            key=lambda x: x[1]["total_requests"],
            reverse=True
        )
        
        for identifier, stats in sorted_identifiers:
            print(f"\n{identifier}:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Rate Limited: {stats['rate_limited_requests']} ({stats['rate_limited_percentage']:.1f}%)")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Retries: {stats['total_retries']}")
            print(f"  Current Rate: {stats['requests_per_minute']:.1f} req/min")
            
            if stats['rate_limited_requests'] > 0:
                print(f"  Avg Wait: {stats['average_wait_time']:.2f}s")
                print(f"  Max Wait: {stats['max_wait_time']:.2f}s")
            
            if stats['error_types']:
                print(f"  Errors: {dict(stats['error_types'])}")
        
        print("="*80)
    
    def _apply_monkey_patches(self):
        """Apply monkey patches to RateLimiter methods to collect statistics."""
        # Store original methods
        self._original_methods['check_rate_limit'] = RateLimiter.check_rate_limit
        self._original_methods['record_request'] = RateLimiter.record_request
        self._original_methods['record_error'] = RateLimiter.record_error
        
        # Create wrapped methods
        def wrapped_check_rate_limit(self, identifier: str = "default"):
            allowed, wait_time = monitor._original_methods['check_rate_limit'](self, identifier)
            if not allowed and wait_time:
                monitor.record_request(identifier, wait_time, success=False)
            return allowed, wait_time
        
        def wrapped_record_request(self, identifier: str = "default"):
            result = monitor._original_methods['record_request'](self, identifier)
            monitor.record_request(identifier, 0.0, success=True)
            return result
        
        def wrapped_record_error(self, identifier: str = "default", error: Exception = None):
            monitor.record_retry(identifier)
            error_type = type(error).__name__ if error else "Unknown"
            monitor.record_request(identifier, 0.0, success=False, error_type=error_type)
            return monitor._original_methods['record_error'](self, identifier, error)
        
        # Apply patches
        RateLimiter.check_rate_limit = wrapped_check_rate_limit
        RateLimiter.record_request = wrapped_record_request
        RateLimiter.record_error = wrapped_record_error
        
        logger.info("Rate limiter monkey patches applied")
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self.stats.clear()
        logger.info("Rate limit statistics reset")


# Global monitor instance
monitor = RateLimitMonitor()


def enable_monitoring():
    """Enable global rate limit monitoring."""
    monitor.enable()


def disable_monitoring():
    """Disable global rate limit monitoring."""
    monitor.disable()


def get_monitoring_stats() -> Dict[str, Any]:
    """Get current monitoring statistics."""
    return monitor.get_summary()


def print_monitoring_summary():
    """Print current monitoring summary."""
    monitor.print_summary()


def reset_monitoring_stats():
    """Reset monitoring statistics."""
    monitor.reset_stats()


# Auto-enable monitoring on import
enable_monitoring() 