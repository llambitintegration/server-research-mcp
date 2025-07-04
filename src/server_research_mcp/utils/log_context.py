"""Simple logging utilities for compatibility."""

import logging
from contextlib import contextmanager
from functools import wraps


@contextmanager
def log_context(**kwargs):
    """Simple context manager that does nothing but maintains compatibility."""
    yield


def log_execution(func=None, *, measure_time=True, **kwargs):
    """Simple execution decorator for compatibility."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(f.__module__)
            logger.info(f"Starting {f.__name__}")
            try:
                result = f(*args, **kwargs)
                logger.info(f"Completed {f.__name__}")
                return result
            except Exception as e:
                logger.error(f"Failed {f.__name__}: {e}")
                raise
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


@contextmanager
def log_timer(logger, operation, level=logging.INFO):
    """Simple timer context for compatibility."""
    logger.log(level, f"Starting {operation}")
    try:
        yield
    finally:
        logger.log(level, f"Completed {operation}")


# Aliases for compatibility
log_execution_time = log_execution 