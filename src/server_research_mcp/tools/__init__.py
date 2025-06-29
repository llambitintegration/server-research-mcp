"""
MCP Tools package for server-research-mcp.

This package provides a unified, extensible tool system using mcpadapt
for clean integration with MCP servers.
"""

# Import the mcpadapt-based tool functions
from .mcp_tools import (
    # Tool collection functions
    get_historian_tools,
    get_researcher_tools,
    get_archivist_tools,
    get_publisher_tools,
    get_context7_tools,
    get_all_mcp_tools,
    
    # Basic tools for compatibility
    SchemaValidationTool,
    IntelligentSummaryTool,
    add_basic_tools,
)

# MCPAdapt adapter
# Legacy custom adapter removed - using official mcpadapt.crewai_adapter.CrewAIAdapter

__all__ = [
    # Tool Collection Functions
    "get_historian_tools",
    "get_researcher_tools",
    "get_archivist_tools",
    "get_publisher_tools",
    "get_context7_tools",
    "get_all_mcp_tools",
    
    # Basic Tools
    "SchemaValidationTool",
    "IntelligentSummaryTool",
    "add_basic_tools",
    
    # MCPAdapt Integration
    # Custom adapter removed - use mcpadapt.crewai_adapter.CrewAIAdapter directly
]
