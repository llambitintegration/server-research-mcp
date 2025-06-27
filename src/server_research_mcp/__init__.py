"""Server Research MCP - A CrewAI-based research system with MCP integration."""

import sys, io
# Ensure stdout/stderr can handle UTF-8 characters regardless of system locale
try:
    # Python 3.7+ provides reconfigure
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    # Fallback for older Python versions
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .crew import ServerResearchMcp
from .main import run, train, replay, test

__version__ = "0.1.0"

__all__ = [
    'ServerResearchMcp',
    'run',
    'train',
    'replay',
    'test'
]
