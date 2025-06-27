"""
Protocol interfaces for MCP server interactions.

This module defines Protocol interfaces that establish contracts for different
types of MCP servers and managers. These protocols enable dependency injection,
better testing, and architectural flexibility while maintaining backward
compatibility with existing implementations.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List
from ._types import ToolResponse, ToolParameters, EntityData


@runtime_checkable
class MemoryServerProtocol(Protocol):
    """Protocol defining memory server interface for knowledge graph operations."""
    
    async def search_nodes(self, query: str) -> ToolResponse:
        """Search memory graph nodes by query string."""
        ...
    
    async def create_entities(self, entities: List[EntityData]) -> ToolResponse:
        """Create new entities in the memory graph."""
        ...
    
    async def add_observations(self, observations: List[Dict[str, Any]]) -> ToolResponse:
        """Add observations to existing entities in the memory graph."""
        ...
    
    async def read_graph(self) -> ToolResponse:
        """Read the entire memory graph structure."""
        ...


@runtime_checkable
class FilesystemServerProtocol(Protocol):
    """Protocol defining filesystem server interface for file operations."""
    
    async def read_file(self, path: str) -> ToolResponse:
        """Read contents of a file at the specified path."""
        ...
    
    async def write_file(self, path: str, content: str) -> ToolResponse:
        """Write content to a file at the specified path."""
        ...
    
    async def list_directory(self, path: str) -> ToolResponse:
        """List contents of a directory at the specified path."""
        ...


@runtime_checkable
class ResearchServerProtocol(Protocol):
    """Protocol defining research/browser server interface for web operations."""
    
    async def web_search(self, query: str) -> ToolResponse:
        """Perform a web search with the given query."""
        ...
    
    async def read_url(self, url: str) -> ToolResponse:
        """Read content from the specified URL."""
        ...
    
    async def take_screenshot(self, url: str) -> ToolResponse:
        """Take a screenshot of the webpage at the specified URL."""
        ...


@runtime_checkable
class MCPManagerProtocol(Protocol):
    """
    Protocol defining the MCP Manager interface.
    
    This protocol matches the existing MCPManager implementation exactly,
    ensuring backward compatibility while enabling dependency injection.
    """
    
    async def call_tool(self, server: str, tool: str, 
                       arguments: ToolParameters) -> ToolResponse:
        """
        Call a tool on the specified MCP server.
        
        Args:
            server: Name of the MCP server (e.g., 'memory', 'context7')
            tool: Name of the tool/method to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Response from the MCP server
        """
        ... 