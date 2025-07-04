"""Simple logging configuration for MCP Server Research."""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from contextlib import contextmanager

# Cross-platform emoji/symbol mapping
# On Windows, use ASCII alternatives to avoid encoding issues
SYMBOLS = {
    'success': '✅' if os.name != 'nt' else '[OK]',
    'error': '❌' if os.name != 'nt' else '[ERR]',
    'warning': '⚠️' if os.name != 'nt' else '[WARN]',
    'rocket': '🚀' if os.name != 'nt' else '[START]',
    'boom': '💥' if os.name != 'nt' else '[FAIL]',
    'party': '🎉' if os.name != 'nt' else '[DONE]',
    'wave': '👋' if os.name != 'nt' else '[HI]',
    'info': 'ℹ️' if os.name != 'nt' else '[INFO]',
    'debug': '🔧' if os.name != 'nt' else '[DBG]',
    'test': '🧪' if os.name != 'nt' else '[TEST]',
    'docs': '📝' if os.name != 'nt' else '[DOC]',
    'search': '🔍' if os.name != 'nt' else '[FIND]',
    'stats': '📊' if os.name != 'nt' else '[STAT]',
    'task': '📋' if os.name != 'nt' else '[TASK]',
    'target': '🎯' if os.name != 'nt' else '[AIM]',
    'cycle': '🔄' if os.name != 'nt' else '[LOOP]',
    'star': '⭐' if os.name != 'nt' else '[STAR]',
    'mobile': '📱' if os.name != 'nt' else '[MOB]',
    'computer': '💻' if os.name != 'nt' else '[PC]',
    'star2': '🌟' if os.name != 'nt' else '[SHINE]',
    'clipboard': '📋' if os.name != 'nt' else '[CLIP]',
    'construction': '🚧' if os.name != 'nt' else '[WIP]',
    'art': '🎨' if os.name != 'nt' else '[ART]',
    'balance': '⚖️' if os.name != 'nt' else '[BAL]',
    'shield': '🛡️' if os.name != 'nt' else '[GUARD]',
    'tag': '🏷️' if os.name != 'nt' else '[TAG]',
    'bulb': '💡' if os.name != 'nt' else '[IDEA]'
}


def get_symbol(name: str) -> str:
    """Get cross-platform symbol by name."""
    return SYMBOLS.get(name, f'[{name.upper()}]')


class CrewAIHandler(logging.StreamHandler):
    """Console handler that stays quiet during crew execution."""
    
    def __init__(self):
        super().__init__(sys.stderr)
        self._crew_active = False
        
        # Set encoding explicitly for Windows compatibility
        if hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, OSError):
                pass  # Fallback handled by SYMBOLS mapping
    
    def emit(self, record):
        # Only show warnings/errors when crew is running
        if self._crew_active and record.levelno < logging.WARNING:
            return
        super().emit(record)
    
    def set_crew_active(self, active: bool):
        self._crew_active = active


class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode safely across platforms."""
    
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # Fallback: remove non-ASCII characters
            msg = record.getMessage()
            clean_msg = msg.encode('ascii', 'replace').decode('ascii')
            record.msg = clean_msg
            return super().format(record)


def setup_logging():
    """Configure logging with sensible defaults and cross-platform Unicode support."""
    # Suppress LiteLLM verbose output at the environment level
    os.environ['LITELLM_LOG'] = 'ERROR'
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root.handlers.clear()
    
    # File handler - captures everything with UTF-8 encoding
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/app.log", maxBytes=10_000_000, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root.addHandler(file_handler)
    
    # Console handler - respects crew execution
    console_handler = CrewAIHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(SafeFormatter('%(levelname)s: %(message)s'))
    root.addHandler(console_handler)
    
    # Quiet noisy libraries
    for lib in ['urllib3', 'httpx', 'chromadb', 'openai', 'anthropic', 'litellm']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Specifically suppress LiteLLM verbose output
    logging.getLogger('LiteLLM').setLevel(logging.ERROR)
    logging.getLogger('litellm').setLevel(logging.ERROR)
    
    # Store console handler globally for crew context
    global _console_handler
    _console_handler = console_handler


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


@contextmanager
def crew_execution():
    """Suppress console noise during crew execution."""
    global _console_handler
    if '_console_handler' in globals() and _console_handler:
        _console_handler.set_crew_active(True)
    try:
        yield
    finally:
        if '_console_handler' in globals() and _console_handler:
            _console_handler.set_crew_active(False)


def log_execution(func):
    """Simple execution logging decorator."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(f"{get_symbol('rocket')} Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{get_symbol('success')} Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"{get_symbol('error')} Failed {func.__name__}: {e}")
            raise
    return wrapper


# Aliases for compatibility
log_execution_time = log_execution 