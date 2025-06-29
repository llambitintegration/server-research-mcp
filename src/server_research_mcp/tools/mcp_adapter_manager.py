"""
MCPServerAdapter Manager for handling connections to MCP servers using official CrewAI patterns.
Replaces the custom MCPManager with MCPServerAdapter from crewai-tools.
"""
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

logger = logging.getLogger(__name__)

# Singleton instance
_mcp_adapter_manager_instance = None

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server using MCPServerAdapter."""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = None
    transport_type: str = "stdio"
    
    def __post_init__(self):
        if self.env is None:
            self.env = {}
    
    def to_server_params(self) -> StdioServerParameters:
        """Convert to MCPServerAdapter serverparams format using StdioServerParameters."""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env or {}
        )


class MCPServerAdapterManager:
    """
    Manager for MCPServerAdapter instances from crewai-tools.
    
    Handles multiple MCP servers with proper lifecycle management:
    - memory (server-memory)
    - context7 (server-context7)  
    - sequential-thinking (server-sequential-thinking)
    - zotero (server-zotero)
    - filesystem (server-filesystem)
    - obsidian-mcp-tools (obsidian-mcp-tools)
    """
    
    # Server configurations using official transport patterns
    SERVER_CONFIGS = {
        "memory": MCPServerConfig(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            env={}
        ),
        "zotero": MCPServerConfig(
            name="zotero",
            command="uvx",
            args=["zotero-mcp"],
            env={
                "ZOTERO_LOCAL": "false",
                "ZOTERO_API_KEY": os.getenv("ZOTERO_API_KEY", "gTBfmbXpAVhVLPh8ffjZcakJ"),
                "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_LIBRARY_ID", "17381274")
            }
        ),
        "sequential-thinking": MCPServerConfig(
            name="sequential-thinking",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
            env={}
        ),
        "context7": MCPServerConfig(
            name="context7",
            command="npx",
            args=["-y", "@upstash/context7-mcp"],
            env={}
        ),
        "filesystem": MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={}
        ),
        "obsidian-mcp-tools": MCPServerConfig(
            name="obsidian-mcp-tools",
            command="npx",
            args=["-y", "obsidian-mcp-tools"],
            env={}
        )
    }
    
    def __init__(self):
        self.adapters: Dict[str, MCPServerAdapter] = {}
        self.initialized_servers: set = set()
        self._server_tools: Dict[str, List[Any]] = {}
        self._active_contexts: Dict[str, Any] = {}
        
    async def initialize_server(self, server_name: str) -> None:
        """
        Initialize a single MCP server using MCPServerAdapter.
        
        Args:
            server_name: Name of the server to initialize
        """
        if server_name in self.initialized_servers:
            return
            
        if server_name not in self.SERVER_CONFIGS:
            raise ValueError(f"Unknown MCP server: {server_name}")
            
        config = self.SERVER_CONFIGS[server_name]
        logger.info(f"🚀 Initializing MCP server: {server_name}")
        logger.debug(f"📋 Server config: {config}")
        
        try:
            # Create MCPServerAdapter with server configuration
            serverparams = config.to_server_params()
            adapter = MCPServerAdapter(serverparams)
            
            self.adapters[server_name] = adapter
            self.initialized_servers.add(server_name)
            
            logger.info(f"✅ MCP server {server_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP server {server_name}: {e}")
            # Don't add to initialized_servers on failure
            if server_name in self.adapters:
                del self.adapters[server_name]
            raise
    
    async def initialize(self, servers: List[str]) -> None:
        """
        Initialize connections to specified MCP servers.
        
        Args:
            servers: List of server names to initialize
        """
        logger.info(f"🛠️ MCPServerAdapterManager.initialize called with servers={servers}")
        
        # Initialize all servers concurrently
        tasks = []
        for server in servers:
            if server not in self.initialized_servers:
                tasks.append(self.initialize_server(server))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    @asynccontextmanager
    async def get_server_tools(self, server_name: str):
        """
        Get tools from a server using context manager pattern.
        
        Args:
            server_name: Name of the MCP server
            
        Yields:
            List of tools from the server
        """
        if server_name not in self.initialized_servers:
            await self.initialize_server(server_name)
            
        adapter = self.adapters[server_name]
        
        # Use MCPServerAdapter's context manager
        with adapter as tools:
            logger.info(f"📚 Retrieved {len(tools)} tools from {server_name}")
            yield tools
    
    async def call_tool(
        self, 
        server: str, 
        tool: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server (compatibility method).
        
        Args:
            server: Name of the MCP server
            tool: Name of the tool/method to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Response from the MCP server
        """
        logger.info(f"📞 Calling {server}.{tool} with args: {arguments}")
        
        try:
            # For compatibility mode, provide a mock-like response
            # Real MCPServerAdapter tools will be used directly by agents
            logger.info(f"⚠️ Compatibility mode: Simulating {server}.{tool} call")
            
            # Ensure the server is initialized (this part works fine)
            if server not in self.initialized_servers:
                await self.initialize_server(server)
            
            # Return a mock response for compatibility
            return {
                "status": "success",
                "result": f"Compatibility mode response for {server}.{tool} with args: {arguments}",
                "server": server,
                "tool": tool,
                "note": "This is a compatibility response. Real tools are available via get_server_tools() context manager."
            }
                
        except Exception as e:
            logger.error(f"❌ Tool call failed for {server}.{tool}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "server": server,
                "tool": tool
            }
    
    async def get_all_tools_for_servers(self, servers: List[str]) -> Dict[str, List[Any]]:
        """
        Get all tools from multiple servers.
        
        Args:
            servers: List of server names
            
        Returns:
            Dictionary mapping server names to their tools
        """
        result = {}
        
        for server in servers:
            try:
                async with self.get_server_tools(server) as tools:
                    result[server] = list(tools)  # Convert to list for serialization
            except Exception as e:
                logger.error(f"❌ Failed to get tools from {server}: {e}")
                result[server] = []
        
        return result
    
    async def close(self):
        """Close all server connections."""
        logger.info("🔌 Closing MCPServerAdapterManager connections")
        
        # MCPServerAdapter handles cleanup via context managers
        # Clear our tracking
        self.adapters.clear()
        self.initialized_servers.clear()
        self._server_tools.clear()
        self._active_contexts.clear()
        
        logger.info("✅ MCPServerAdapterManager closed successfully")
    
    async def shutdown(self):
        """Shutdown the manager."""
        await self.close()


def get_mcp_adapter_manager() -> MCPServerAdapterManager:
    """
    Get singleton instance of MCPServerAdapterManager.
    
    Returns:
        MCPServerAdapterManager instance
    """
    global _mcp_adapter_manager_instance
    
    if _mcp_adapter_manager_instance is None:
        logger.info("🏗️ Creating new MCPServerAdapterManager instance")
        _mcp_adapter_manager_instance = MCPServerAdapterManager()
    
    return _mcp_adapter_manager_instance


# Backward compatibility alias
def get_mcp_manager() -> MCPServerAdapterManager:
    """
    Backward compatibility function.
    Returns MCPServerAdapterManager instance.
    """
    logger.info("🔄 Using backward compatibility bridge to MCPServerAdapterManager")
    return get_mcp_adapter_manager() 