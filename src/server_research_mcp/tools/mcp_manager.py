"""
MCP Manager for handling connections to MCP servers.
This is a placeholder implementation that would be replaced with actual MCP client code.
"""
from typing import List, Dict, Any, Optional
import asyncio
import logging
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton instance
_mcp_manager_instance = None

# Global in-memory store for mock memory server to allow cross-session persistence during tests
_mock_memory_entities: list = []


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env is None:
            self.env = {}


class MockMCPClient:
    """Mock MCP client for testing purposes."""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.connected = False
    
    async def connect(self):
        """Mock connection."""
        self.connected = True
        return True
    
    async def disconnect(self):
        """Mock disconnection."""
        self.connected = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call."""
        return {
            "status": "success",
            "server": self.server_name,
            "tool": tool_name,
            "arguments": arguments,
            "result": f"Mock result from {self.server_name}.{tool_name}"
        }


class MCPManager:
    """
    Manager for MCP (Model Context Protocol) server connections.
    
    This would integrate with actual MCP servers like:
    - memory (server-memory)
    - context7 (server-context7)  
    - sequential-thinking (server-sequential-thinking)
    - zotero (server-zotero)
    """
    
    # Server configurations for testing
    SERVER_CONFIGS = {
        "memory": MCPServerConfig(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            env={}
        ),
        "zotero": MCPServerConfig(
            name="zotero",
            command="npx",
            args=["-y", "zotero-mcp-server"],
            env={
                "ZOTERO_API_KEY": "ZepzC2gMqwCquCc1rCxbPH96",
                "ZOTERO_LIBRARY_ID": "17381274",
                "ZOTERO_LOCAL": "false"
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
        self.initialized_servers: set = set()
        self._mock_mode = True  # Remove this when implementing real MCP
        self._initialized = False
        self.clients: Dict[str, MockMCPClient] = {}
        
    async def initialize(self, servers: List[str]) -> None:
        """
        Initialize connections to specified MCP servers.
        
        Args:
            servers: List of server names to initialize
        """
        logger.info("🛠️ MCPManager.initialize called with servers=%s (mock_mode=%s)", servers, self._mock_mode)
        for server in servers:
            if server not in self.initialized_servers:
                logger.info(f"Initializing MCP server: {server}")
                # In real implementation, this would:
                # 1. Start the MCP server process if needed
                # 2. Establish connection
                # 3. Load server capabilities
                
                # For testing, create mock clients for known servers
                if server in self.SERVER_CONFIGS:
                    self.clients[server] = MockMCPClient(server)
                    await self.clients[server].connect()
                    
                self.initialized_servers.add(server)
        
        self._initialized = True
        
    async def call_tool(
        self, 
        server: str, 
        tool: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Args:
            server: Name of the MCP server
            tool: Name of the tool/method to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Response from the MCP server
        """
        # Ensure server is initialized
        if server not in self.initialized_servers:
            await self.initialize([server])
            
        logger.info(f"Calling {server}.{tool} with args: {arguments}")
        
        # Mock responses for testing
        if self._mock_mode:
            logger.debug("🔄 Returning mock response for %s.%s (mock mode active)", server, tool)
            return self._get_mock_response(server, tool, arguments)
            
        # Real implementation would:
        # 1. Send request to MCP server
        # 2. Wait for response
        # 3. Parse and return result
        
    def _get_mock_response(
        self, 
        server: str, 
        tool: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide mock responses for testing.
        """
        if server == "memory":
            if tool == "search_nodes":
                query = (arguments.get("query") or "").lower()
                matching = [e for e in _mock_memory_entities if query in e.get("name", "").lower()]
                return {
                    "status": "success",
                    "nodes": matching,
                    "message": f"Found {len(matching)} nodes for query: {arguments.get('query')}"
                }
            elif tool == "create_entities":
                first_entity = {}
                if isinstance(arguments.get("entities"), list) and arguments["entities"]:
                    first_entity = arguments["entities"][0]
                else:
                    first_entity = {
                        "name": arguments.get("name"),
                        "entity_type": arguments.get("entity_type"),
                        "observations": arguments.get("observations", [])
                    }

                # Persist entity in global mock store
                _mock_memory_entities.append(first_entity)
                return {
                    "status": "success",
                    "entity_id": "mock-entity-123",
                    "message": f"Created entity: {first_entity.get('name')}"
                }
            elif tool == "add_observations":
                entity_name = arguments.get("entityName") or arguments.get("entity_name")
                observations = arguments.get("observations", [])

                # Add observations to matching entities in mock store
                updated = 0
                for ent in _mock_memory_entities:
                    if ent.get("name") == entity_name:
                        ent_observations = ent.setdefault("observations", [])
                        ent_observations.extend(observations)
                        updated += 1
                return {
                    "status": "success",
                    "updated_entities": updated,
                    "message": f"Added {len(observations)} observations to {updated} entities"
                }
                
        elif server == "context7":
            if tool == "resolve-library-id":
                return {
                    "libraryId": f"/mock/{arguments.get('libraryName', 'unknown')}/docs",
                    "found": True,
                    "name": arguments.get('libraryName', 'unknown')
                }
            elif tool == "get-library-docs":
                return {
                    "content": f"Mock documentation for {arguments.get('context7CompatibleLibraryID', 'unknown')}",
                    "tokens": max(arguments.get('tokens', 10000), 10000),
                    "topic": arguments.get('topic', ''),
                    "libraryId": arguments.get('context7CompatibleLibraryID', 'unknown')
                }
            # Legacy support for old tool names
            elif tool == "context7_resolve":
                return {
                    "library_id": f"/mock/{arguments.get('library_name')}/docs",
                    "found": True
                }
            elif tool == "context7_get_docs":
                return {
                    "documentation": f"Mock documentation for {arguments.get('context7_library_id')}",
                    "tokens": 1000
                }
                
        elif server == "sequential-thinking":
            return {
                "thought_recorded": True,
                "thought_number": arguments.get("thought_number"),
                "continue": arguments.get("next_thought_needed", False)
            }
            
        elif server == "zotero":
            if tool == "search" or tool == "search_items":
                return {
                    "status": "success",
                    "items": [
                        {
                            "key": "TEST123",
                            "title": "Mock Research Paper",
                            "authors": ["Mock Author"],
                            "year": 2024
                        }
                    ],
                    "total": 1
                }
            elif tool == "get_item" or tool == "get_item_fulltext":
                return {
                    "status": "success",
                    "content": "Mock PDF content..." if arguments.get("include_pdf") else "Mock paper content",
                    "metadata": {
                        "title": "Mock Item Details",
                        "authors": ["Mock Author"]
                    }
                }
        
        # Default response
        return {
            "status": "success",
            "message": f"Mock response from {server}.{tool}"
        }
    
    async def close(self):
        """
        Close all MCP server connections.
        """
        logger.info("Closing all MCP server connections")
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()
        self.initialized_servers.clear()
        self._initialized = False

    async def shutdown(self):
        """
        Shutdown the MCP manager (alias for close).
        """
        await self.close()


def get_mcp_manager() -> MCPManager:
    """
    Get the singleton MCP manager instance.
    """
    global _mcp_manager_instance
    if _mcp_manager_instance is None:
        _mcp_manager_instance = MCPManager()
    return _mcp_manager_instance


# --- External configuration loading ---------------------------------------------------------
# Allow users to override or extend server configurations via a JSON file. The lookup order is:
# 1. Environment variable MCP_CONFIG_PATH (explicit path)
# 2. Default path at "~/.cursor/mcp.json".
# The JSON structure should match:
# {
#   "zotero": {
#     "command": "uvx",
#     "args": ["zotero-mcp"],
#     "env": {"ZOTERO_API_KEY": "...", ...}
#   },
#   ...
# }
# Each key represents the server name. Missing fields fall back to defaults or empty values.

def _load_external_server_configs():
    """Load external server configuration from JSON and merge with SERVER_CONFIGS.

    This enables user-specific overrides (e.g., different command runners, env vars)
    without modifying the codebase. It is executed during module import so that both
    ``MCPManager`` and any other modules importing ``MCPManager.SERVER_CONFIGS`` get
    the updated values.
    """
    potential_paths = []

    # 1. Explicit path via environment variable
    env_path = os.getenv("MCP_CONFIG_PATH")
    if env_path:
        potential_paths.append(Path(env_path).expanduser())

    # 2. Default location under the user's home directory
    potential_paths.append(Path.home() / ".cursor" / "mcp.json")

    for config_path in potential_paths:
        if config_path and config_path.is_file():
            try:
                with config_path.open("r", encoding="utf-8") as fp:
                    external_cfg = json.load(fp)

                if not isinstance(external_cfg, dict):
                    logger.warning("External MCP config %s is not a JSON object. Skipping.", config_path)
                    continue

                for server_name, cfg in external_cfg.items():
                    if not isinstance(cfg, dict):
                        logger.warning("Invalid configuration for server '%s' in %s. Skipping.", server_name, config_path)
                        continue

                    # Merge or create a new MCPServerConfig
                    command = cfg.get("command", "npx")
                    args = cfg.get("args", [])
                    env_vars = cfg.get("env", {}) or {}

                    MCPManager.SERVER_CONFIGS[server_name] = MCPServerConfig(
                        name=server_name,
                        command=command,
                        args=args,
                        env=env_vars,
                    )

                logger.info("Loaded external MCP server configurations from %s", config_path)
                # Stop at the first valid config file found
                break
            except Exception as e:
                logger.error("Failed to load external MCP configuration from %s: %s", config_path, e)
                # Continue to next potential path if available
                continue


# Execute after class definition so SERVER_CONFIGS exists
_load_external_server_configs()


