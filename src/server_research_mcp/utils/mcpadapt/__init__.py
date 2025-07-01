__version__ = "0.1.10"

# Core MCPAdapt functionality
from .core import MCPAdapt, ToolAdapter

# CrewAI adapter (our focus)
from .crewai_adapter import CrewAIAdapter

# Utility functions
from .utils.modeling import resolve_refs_and_remove_defs

__all__ = [
    "MCPAdapt",
    "ToolAdapter", 
    "CrewAIAdapter",
    "resolve_refs_and_remove_defs",
]
