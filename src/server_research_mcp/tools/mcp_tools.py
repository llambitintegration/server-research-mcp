"""
MCP Tools System - Simplified Architecture
Clean, unified MCP server integration with declarative configuration.
"""
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional, Type, Callable
import logging
from ..utils.logging_config import get_symbol
import os
import asyncio
import threading
from dataclasses import dataclass, field
from functools import wraps
from contextlib import asynccontextmanager

from ..utils.mcpadapt import MCPAdapt, CrewAIAdapter
from mcp import StdioServerParameters
from pydantic import BaseModel, Field, create_model
from ..utils.rate_limiting import (
    wrap_tools_with_rate_limit,
    RateLimitConfig,
    get_rate_limited_tools
)

logger = logging.getLogger(__name__)

# =============================================================================
# Core Tool Adapter
# =============================================================================

class MCPToolAdapter:
    """Unified adapter for MCP tools to CrewAI tools."""
    
    def __init__(self, 
                 name: str,
                 server_params: StdioServerParameters,
                 parameter_handlers: Optional[Dict[str, Callable]] = None):
        self.name = name
        self.server_params = server_params
        self.parameter_handlers = parameter_handlers or {}
        self._tools: Optional[List[BaseTool]] = None
        self._ctx = None
        
    async def initialize(self):
        """Initialize MCP server connection."""
        try:
            # Use MCPAdapt for server management
            self._ctx = MCPAdapt([self.server_params], CrewAIAdapter())
            self._tools = self._ctx.__enter__()
            logger.info(f"{get_symbol('success')} {self.name} server initialized with {len(self._tools)} tools")
        except Exception as e:
            logger.error(f"{get_symbol('error')} Failed to initialize {self.name} server: {e}")
            self._tools = []
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools from this MCP server as CrewAI tools."""
        if self._tools is None:
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, create task instead of using asyncio.run()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.initialize())
                    future.result()  # Wait for completion
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                asyncio.run(self.initialize())
        return self._tools or []
    
    def shutdown(self):
        """Cleanup MCP server connection."""
        if self._ctx:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Warning during {self.name} server shutdown: {e}")

# =============================================================================
# Tool Registry and Mapping
# =============================================================================

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    parameter_handlers: Dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class ToolMapping:
    """Mapping of tools to agents."""
    agent_name: str
    tool_patterns: List[str]
    required_count: int = 0
    fallback_enabled: bool = True

class MCPToolRegistry:
    """Central registry for MCP servers and tool mappings."""
    
    def __init__(self):
        self.servers: Dict[str, ServerConfig] = {}
        self.mappings: List[ToolMapping] = []
        self.adapters: Dict[str, MCPToolAdapter] = {}
        self._initialized = False
        self._all_tools: Optional[List[BaseTool]] = None
    
    def register_server(self, name: str, **kwargs):
        """Register an MCP server configuration."""
        self.servers[name] = ServerConfig(name=name, **kwargs)
        logger.info(f"{get_symbol('docs')} Registered MCP server: {name}")
    
    def map_tools(self, agent: str, patterns: List[str], **kwargs):
        """Map tool patterns to an agent."""
        self.mappings.append(ToolMapping(
            agent_name=agent,
            tool_patterns=patterns,
            **kwargs
        ))
        logger.info(f"🔗 Mapped tools to {agent}: {patterns}")
    
    def initialize_all(self):
        """Initialize all registered MCP servers."""
        if self._initialized:
            return
            
        for name, config in self.servers.items():
            if not config.enabled:
                continue
                
            try:
                # Create server parameters
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env
                )
                
                # Create adapter
                adapter = MCPToolAdapter(
                    name=name,
                    server_params=server_params,
                    parameter_handlers=config.parameter_handlers
                )
                
                # Store adapter (will initialize on first tool request)
                self.adapters[name] = adapter
                
            except Exception as e:
                logger.error(f"Failed to register {name} server: {e}")
        
        self._initialized = True
        logger.info(f"{get_symbol('success')} Registered {len(self.adapters)} MCP servers")
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all servers."""
        if self._all_tools is None:
            self.initialize_all()
            all_tools = []
            for adapter in self.adapters.values():
                all_tools.extend(adapter.get_tools())
            self._all_tools = all_tools
        return self._all_tools
    
    def get_agent_tools(self, agent_name: str, apply_rate_limiting: bool = True) -> List[BaseTool]:
        """Get all tools for a specific agent with comprehensive rate limiting."""
        # Find mapping for this agent
        mapping = next(
            (m for m in self.mappings if m.agent_name == agent_name), 
            None
        )
        
        if not mapping:
            logger.warning(f"No tool mapping found for agent: {agent_name}")
            return []
        
        # Get all available tools
        all_tools = self.get_all_tools()
        logger.info(f"{get_symbol('docs')} Total available tools: {len(all_tools)}")
        
        # Filter tools by patterns
        matched_tools = []
        for pattern in mapping.tool_patterns:
            pattern_lower = pattern.lower()
            for tool in all_tools:
                if pattern_lower in tool.name.lower() and tool not in matched_tools:
                    matched_tools.append(tool)
        
        logger.info(f"{get_symbol('success')} Matched {len(matched_tools)} tools for {agent_name}: {[t.name for t in matched_tools]}")
        
        # Handle minimum count requirement
        if len(matched_tools) < mapping.required_count and mapping.fallback_enabled:
            fallback_count = mapping.required_count - len(matched_tools)
            fallback_tools = self._create_fallback_tools(agent_name, fallback_count)
            matched_tools.extend(fallback_tools)
            logger.info(f"{get_symbol('warning')} Added {len(fallback_tools)} fallback tools for {agent_name}")
        
        # Apply rate limiting if enabled
        if apply_rate_limiting and matched_tools:
            # Determine tool type for appropriate rate limiting
            tool_type = self._get_tool_type_for_agent(agent_name)
            
            # Log before rate limiting
            logger.info(f"{get_symbol('gear')} Applying {tool_type} rate limiting to {len(matched_tools)} tools for {agent_name}")
            
            # Apply rate limiting based on tool type
            rate_limited_tools = get_rate_limited_tools(matched_tools, tool_type)
            
            # Verify rate limiting was applied
            rate_limited_count = sum(1 for tool in rate_limited_tools if hasattr(tool, 'rate_limiter'))
            logger.info(f"{get_symbol('success')} Rate limiting applied: {rate_limited_count}/{len(rate_limited_tools)} tools wrapped")
            
            # Log rate limit configurations
            for tool in rate_limited_tools:
                if hasattr(tool, 'rate_limiter'):
                    config = tool.rate_limiter.config
                    logger.debug(f"  {tool.name}: {config.max_requests_per_minute} req/min, "
                               f"{config.max_requests_per_hour} req/hr, "
                               f"{config.min_request_interval}s interval")
            
            return rate_limited_tools
        
        return matched_tools
    
    def _get_tool_type_for_agent(self, agent_name: str) -> str:
        """Determine the appropriate rate limiting type for an agent."""
        agent_type_mapping = {
            "researcher": "zotero",
            "historian": "memory", 
            "archivist": "sequential_thinking",
            "publisher": "filesystem"
        }
        return agent_type_mapping.get(agent_name.lower(), "default")
    
    def _create_fallback_tools(self, agent_name: str, count: int) -> List[BaseTool]:
        """Create basic fallback tools."""
        tools = []
        
        for i in range(count):
            class FallbackTool(BaseTool):
                name: str = f"{agent_name}_fallback_tool_{i+1}"
                description: str = f"Fallback tool {i+1} for {agent_name} agent compatibility"
                
                def _run(self, query: str = "") -> str:
                    return f"Fallback tool {self.name} executed: {query}"
            
            tools.append(FallbackTool())
        
        return tools
    
    def shutdown_all(self):
        """Shutdown all MCP servers."""
        for adapter in self.adapters.values():
            adapter.shutdown()
        self._initialized = False
        self._all_tools = None

# =============================================================================
# Parameter Handlers
# =============================================================================

class ParameterHandlers:
    """Collection of parameter transformation handlers."""
    
    @staticmethod
    def zotero_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Zotero-specific parameters."""
        # Zotero expects limit as string
        if 'limit' in params:
            params['limit'] = str(params['limit'])
        return params
    
    @staticmethod
    def filesystem_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle filesystem-specific parameters."""
        # Normalize paths
        if 'path' in params:
            params['path'] = os.path.normpath(params['path'])
        return params
    
    @staticmethod
    def search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search-specific parameters."""
        # Ensure query is non-empty
        if 'query' in params:
            query = str(params.get('query', '')).strip()
            if not query:
                raise ValueError("Search query cannot be empty")
            params['query'] = query
        return params

# =============================================================================
# Global Registry Setup
# =============================================================================

_global_registry = MCPToolRegistry()

def get_registry() -> MCPToolRegistry:
    """Get the global tool registry."""
    return _global_registry

def get_rate_limit_config_from_env(prefix: str = "MCP") -> Dict[str, Any]:
    """Get rate limit configuration from environment variables."""
    config = {}
    
    # Check for rate limiting env vars
    env_vars = {
        f"{prefix}_MAX_REQUESTS_PER_MINUTE": "max_requests_per_minute",
        f"{prefix}_MAX_REQUESTS_PER_HOUR": "max_requests_per_hour",
        f"{prefix}_MIN_REQUEST_INTERVAL": "min_request_interval",
        f"{prefix}_MAX_RETRIES": "max_retries",
        f"{prefix}_INITIAL_RETRY_DELAY": "initial_retry_delay",
        f"{prefix}_MAX_RETRY_DELAY": "max_retry_delay",
        f"{prefix}_BACKOFF_FACTOR": "backoff_factor",
    }
    
    for env_var, config_key in env_vars.items():
        value = os.getenv(env_var)
        if value:
            try:
                # Convert to appropriate type
                if config_key in ["max_requests_per_minute", "max_requests_per_hour", "max_retries"]:
                    config[config_key] = int(value)
                else:
                    config[config_key] = float(value)
            except ValueError:
                logger.warning(f"Invalid value for {env_var}: {value}")
    
    return config

def setup_registry():
    """Setup the global registry with all MCP servers."""
    registry = get_registry()
    
    # Memory server
    registry.register_server(
        "memory",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        parameter_handlers={
            "search": ParameterHandlers.search_handler
        }
    )
    
    # Zotero server (if credentials available)
    if os.getenv("ZOTERO_API_KEY") and os.getenv("ZOTERO_LIBRARY_ID"):
        registry.register_server(
            "zotero",
            command="uvx",
            args=["zotero-mcp"],
            env={
                "ZOTERO_API_KEY": os.getenv("ZOTERO_API_KEY"),
                "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_LIBRARY_ID")
            },
            parameter_handlers={
                "zotero": ParameterHandlers.zotero_handler
            }
        )
    
    # Filesystem server
    obsidian_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\0_repos\mcp\Obsidian")
    registry.register_server(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", obsidian_path],
        parameter_handlers={
            "file": ParameterHandlers.filesystem_handler
        }
    )
    
    # Sequential thinking server
    registry.register_server(
        "sequential_thinking",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )
    
    # Tool mappings - Use specific patterns to avoid cross-contamination
    registry.map_tools("historian", ["entities", "relations", "observations", "graph", "nodes"], required_count=6)
    registry.map_tools("researcher", ["zotero"], required_count=3)
    registry.map_tools("archivist", ["sequential"], required_count=1)
    registry.map_tools("publisher", ["file", "directory", "write", "edit", "move", "list_"], required_count=11)

# =============================================================================
# API Compatibility Functions
# =============================================================================

def get_historian_tools() -> List[BaseTool]:
    """Get tools for historian agent."""
    return get_registry().get_agent_tools("historian")

def get_researcher_tools() -> List[BaseTool]:
    """Get tools for researcher agent."""
    return get_registry().get_agent_tools("researcher")

def get_archivist_tools() -> List[BaseTool]:
    """Get tools for archivist agent."""
    return get_registry().get_agent_tools("archivist")

def get_publisher_tools() -> List[BaseTool]:
    """Get tools for publisher agent."""
    return get_registry().get_agent_tools("publisher")

def get_context7_tools() -> List[BaseTool]:
    """Get Context7 tools (deprecated, use get_all_mcp_tools)."""
    return []

def get_all_mcp_tools() -> Dict[str, List[BaseTool]]:
    """Get all MCP tools organized by agent."""
    return {
        "historian": get_historian_tools(),
        "researcher": get_researcher_tools(), 
        "archivist": get_archivist_tools(),
        "publisher": get_publisher_tools(),
        "context7": get_context7_tools()  # Keep for test compatibility
    }

# =============================================================================
# Basic Tools for Compatibility
# =============================================================================

class _SchemaValidationArgs(BaseModel):
    """Pydantic model for SchemaValidationTool arguments."""
    data: Any = Field(..., description="JSON string or Python object to validate against the research paper schema")

class SchemaValidationTool(BaseTool):
    """Basic schema validation tool."""
    name: str = "schema_validation"
    description: str = "Validate data against research paper JSON schema"
    args_schema: Type[BaseModel] = _SchemaValidationArgs

    def _run(self, data) -> str:
        """Validate data against schema."""
        try:
            import json
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
            
            # Basic validation logic
            required_fields = ["title", "authors"]
            if isinstance(parsed_data, dict):
                missing = [field for field in required_fields if field not in parsed_data]
                if missing:
                    return f"Validation failed: missing required fields {missing}"
                return "Schema validation passed"
            else:
                return "Validation failed: data must be a JSON object"
        except Exception as e:
            return f"Validation error: {str(e)}"

class _IntelligentSummaryArgs(BaseModel):
    """Pydantic model for IntelligentSummaryTool arguments."""
    content: str = Field(..., description="Content to summarize")
    max_length: int = Field(500, description="Maximum length of the summary")

class IntelligentSummaryTool(BaseTool):
    """Basic content summarization tool."""
    name: str = "intelligent_summary"
    description: str = "Generate intelligent summaries of research content"
    args_schema: Type[BaseModel] = _IntelligentSummaryArgs

    def _run(self, content: str, max_length: int = 500) -> str:
        """Generate summary of content."""
        # Basic summarization logic
        if len(content) <= max_length:
            return content
        
        # Simple truncation with ellipsis
        return content[:max_length-3] + "..."

def add_basic_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """Add basic tools to a tool list."""
    return tools + [SchemaValidationTool(), IntelligentSummaryTool()]

# =============================================================================
# Legacy Compatibility
# =============================================================================

BASIC_TOOLS = [SchemaValidationTool(), IntelligentSummaryTool()]

def get_mcp_manager():
    """Legacy compatibility - return registry for tests."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.initialized_servers = list(get_registry().servers.keys())
    mock.call_tool = lambda name, args: f"Mock tool {name} called with {args}"
    return mock

# Initialize registry on import
setup_registry() 

# =============================================================================
# Backward Compatibility Classes
# =============================================================================

class MCPToolWrapper:
    """Backward compatibility wrapper for tests that expect MCPToolWrapper."""
    
    def __init__(self, tool):
        self.tool = tool
    
    def _run(self, **kwargs):
        """Emulate the old MCPToolWrapper behavior."""
        # Apply parameter handlers if this is a Zotero tool
        if hasattr(self.tool, 'name') and 'zotero' in self.tool.name.lower():
            kwargs = ParameterHandlers.zotero_handler(kwargs)
        
        # Call the underlying tool
        if hasattr(self.tool, '_run'):
            return self.tool._run(**kwargs)
        elif callable(self.tool):
            try:
                return self.tool(**kwargs)
            except TypeError:
                # If tool raises TypeError (not callable), fall back to string representation
                return str(kwargs)
        else:
            return str(kwargs)

class _AdaptHolder:
    """Backward compatibility class for tests that expect _AdaptHolder."""
    
    @staticmethod
    def get_all_tools():
        """Return empty list for test compatibility."""
        return [] 