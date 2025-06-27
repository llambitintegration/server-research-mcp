"""
MCP Tools package for server-research-mcp.

This package provides a unified, extensible tool system with plug-and-play
functionality for different agents and crews. All tools are consolidated
in mcp_tools.py for better maintainability.
"""

# Import all tools from the unified mcp_tools module
from .mcp_tools import (
    # Memory Tools (Historian Agent)
    MemorySearchTool,
    MemoryCreateEntityTool,
    MemoryAddObservationTool,
    
    # Context7 Tools (Future agents)
    Context7ResolveTool,
    Context7DocsTool,
    
    # Research Tools (Future Researcher agent)
    ZoteroSearchTool,
    ZoteroExtractTool,
    SequentialThinkingTool,
    
    # Legacy Tools (Preserved for compatibility)
    SchemaValidationTool,
    
    # Tool collection helpers
    get_historian_tools,
    get_context7_tools,
    get_researcher_tools,
    get_archivist_tools,
    get_all_mcp_tools,
    historian_mcp_tools,  # Backward compatibility alias
)

# Infrastructure exports
from .mcp_base_tool import MCPBaseTool
from .tool_factory import (
    mcp_tool,
    json_schema_to_pydantic,
    create_tools_from_mcp_schema,
)
from .mcp_manager import get_mcp_manager, MCPManager

# Backward compatibility - deprecated but maintained
def get_publisher_tools():
    """Deprecated: Use get_context7_tools() or other specific tool collections."""
    return []

__all__ = [
    # Memory Tools
    "MemorySearchTool",
    "MemoryCreateEntityTool", 
    "MemoryAddObservationTool",
    
    # Context7 Tools
    "Context7ResolveTool",
    "Context7DocsTool",
    
    # Research Tools
    "ZoteroSearchTool",
    "ZoteroExtractTool",
    "SequentialThinkingTool",
    
    # Legacy Tools
    "SchemaValidationTool",
    
    # Tool Collection Functions
    "get_historian_tools",
    "get_context7_tools", 
    "get_researcher_tools",
    "get_archivist_tools",
    "get_all_mcp_tools",
    "historian_mcp_tools",
    "get_publisher_tools",  # Deprecated
    
    # Infrastructure
    "MCPBaseTool",
    "mcp_tool",
    "json_schema_to_pydantic",
    "create_tools_from_mcp_schema",
    "get_mcp_manager",
    "MCPManager",
]
