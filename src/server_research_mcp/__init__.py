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

from .crew import ServerResearchMcpCrew

__version__ = "0.1.0"

# Simple compatibility functions for run.py and tests
def run():
    """Run the research crew with user input."""
    import asyncio
    from .main import main
    asyncio.run(main())

def train():
    """Training function for compatibility."""
    print("Training functionality would be implemented here")
    
def replay():
    """Replay function for compatibility.""" 
    print("Replay functionality would be implemented here")
    
def test():
    """Test function for compatibility."""
    print("Test functionality would be implemented here")

__all__ = [
    'ServerResearchMcpCrew',
    'run',
    'train', 
    'replay',
    'test'
]
