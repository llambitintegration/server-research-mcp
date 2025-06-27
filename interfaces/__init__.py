"""
Interface definitions for server-research-mcp.

This package provides Protocol interfaces that define contracts for MCP server
interactions, enabling dependency injection and better testing while maintaining
backward compatibility with existing implementations.
"""

from ._types import (
    ToolResponse,
    ToolParameters,
    EntityData,
)

from .protocols import (
    MemoryServerProtocol,
    FilesystemServerProtocol,
    ResearchServerProtocol,
    MCPManagerProtocol,
)

__all__ = [
    # Type definitions
    "ToolResponse",
    "ToolParameters", 
    "EntityData",
    
    # Protocol interfaces
    "MemoryServerProtocol",
    "FilesystemServerProtocol",
    "ResearchServerProtocol",
    "MCPManagerProtocol",
] 