"""Backoff strategies for rate limiting."""

import time
import random
from abc import ABC, abstractmethod
from typing import Optional


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""
    
    @abstractmethod
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate delay for given attempt number."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the backoff strategy."""
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff with optional jitter."""
    
    def __init__(
        self,
        base: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        jitter_range: float = 0.1
    ):
        self.base = base
        self.max_delay = max_delay
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay."""
        if attempt <= 0:
            return 0
        
        # Calculate exponential delay
        delay = min(base_delay * (self.base ** (attempt - 1)), self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += (2 * random.random() - 1) * jitter_amount
        
        return max(0.1, delay)  # Minimum 100ms
    
    def reset(self):
        """No state to reset for exponential backoff."""
        pass


class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy."""
    
    def __init__(
        self,
        increment: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        jitter_range: float = 0.1
    ):
        self.increment = increment
        self.max_delay = max_delay
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate linear backoff delay."""
        if attempt <= 0:
            return 0
        
        # Calculate linear delay
        delay = min(base_delay + (self.increment * (attempt - 1)), self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += (2 * random.random() - 1) * jitter_amount
        
        return max(0.1, delay)
    
    def reset(self):
        """No state to reset for linear backoff."""
        pass


class ConstantBackoff(BackoffStrategy):
    """Constant delay backoff strategy."""
    
    def __init__(self, delay: float = 1.0, jitter: bool = False, jitter_range: float = 0.1):
        self.delay = delay
        self.jitter = jitter
        self.jitter_range = jitter_range
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Return constant delay."""
        if attempt <= 0:
            return 0
        
        delay = self.delay
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += (2 * random.random() - 1) * jitter_amount
        
        return max(0.1, delay)
    
    def reset(self):
        """No state to reset for constant backoff."""
        pass


class AdaptiveBackoff(BackoffStrategy):
    """Adaptive backoff that learns from success/failure patterns."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        success_factor: float = 0.9,
        failure_factor: float = 1.5,
        min_delay: float = 0.5
    ):
        self.initial_delay = initial_delay
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.success_factor = success_factor
        self.failure_factor = failure_factor
        self.consecutive_failures = 0
    
    def get_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Get adaptive delay based on history."""
        if attempt <= 0:
            return 0
        
        return self.current_delay
    
    def record_success(self):
        """Record a successful request."""
        self.consecutive_failures = 0
        self.current_delay = max(
            self.min_delay,
            self.current_delay * self.success_factor
        )
    
    def record_failure(self):
        """Record a failed request."""
        self.consecutive_failures += 1
        self.current_delay = min(
            self.max_delay,
            self.current_delay * self.failure_factor
        )
    
    def reset(self):
        """Reset to initial state."""
        self.current_delay = self.initial_delay
        self.consecutive_failures = 0
