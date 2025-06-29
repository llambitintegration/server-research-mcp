"""
MCPAdapt-based Tools System for Server Research MCP
Clean, simple MCP server integration using mcpadapt.
"""
from crewai.tools import BaseTool
from typing import List, Dict, Any
import logging
import os
from mcpadapt.core import MCPAdapt
from mcpadapt.crewai_adapter import CrewAIAdapter
from mcp import StdioServerParameters
import asyncio
import threading
import functools
import inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MCP Server Configuration
# =============================================================================

def get_mcp_server_configs() -> List[StdioServerParameters]:
    """Get MCP server configurations for all available servers."""
    
    servers = []
    
    # Memory server
    servers.append(StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        env={}
    ))
    
    # Sequential thinking server
    servers.append(StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
        env={}
    ))
    
    # Context7 server
    servers.append(StdioServerParameters(
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
        env={}
    ))
    
    # Zotero server (if credentials available)
    zotero_api_key = os.getenv("ZOTERO_API_KEY")
    zotero_library_id = os.getenv("ZOTERO_LIBRARY_ID")
    
    if zotero_api_key and zotero_library_id:
        servers.append(StdioServerParameters(
            command="uvx",
            args=["zotero-mcp"],
            env={
                "ZOTERO_LOCAL": "false",
                "ZOTERO_API_KEY": zotero_api_key,
                "ZOTERO_LIBRARY_ID": zotero_library_id
            }
        ))
        logger.info("🔗 Zotero server configured with API credentials")
    else:
        logger.warning("⚠️ Zotero server skipped - missing ZOTERO_API_KEY or ZOTERO_LIBRARY_ID")
    
    return servers

# =============================================================================
# Tool Filtering Functions
# =============================================================================

def filter_tools_by_keywords(tools: List[BaseTool], keywords: List[str]) -> List[BaseTool]:
    """Filter tools by keywords in their names."""
    filtered = []
    for tool in tools:
        tool_name = getattr(tool, 'name', '').lower()
        if any(keyword.lower() in tool_name for keyword in keywords):
            filtered.append(tool)
    return filtered

# =============================================================================
# Persistent MCPAdapt Holder (prevents premature event-loop shutdown)
# =============================================================================

class _AdaptHolder:
    """Singleton-style holder that initialises MCPAdapt once and keeps it open.

    We *do not* exit the context manager so the async resources (event loop,
    aiohttp sessions, etc.) stay active for the entire process lifetime.
    """
    _tools: List[BaseTool] | None = None
    _ctx: Any = None  # Holds the active context manager so it is not garbage-collected

    @classmethod
    def _initialise(cls) -> List[BaseTool]:
        def _setup_sync():
            server_configs = get_mcp_server_configs()
            ctx_local = MCPAdapt(server_configs, CrewAIAdapter())
            tools_local = ctx_local.__enter__()
            return ctx_local, tools_local

        async def _setup_async():
            return _setup_sync()

        try:
            # Ensure we build inside the background loop to bind IO resources
            ctx, tools = _AsyncWorker.run(_setup_async())
            
            # Wrap tools with enhanced error handling and async bridging
            wrapped_tools = []
            for tool in tools:
                patched_tool = _patch_tool(tool)
                wrapped_tool = MCPToolWrapper(patched_tool)
                wrapped_tools.append(wrapped_tool)
            
            cls._ctx = ctx
            return wrapped_tools
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            # Return basic tools as fallback
            return [SchemaValidationTool(), IntelligentSummaryTool()]

    @classmethod
    def get_all_tools(cls) -> List[BaseTool]:
        if cls._tools is None:
            cls._tools = cls._initialise()
        return cls._tools

# =============================================================================
# Updated Tool Accessors (reuse persistent tool list)
# =============================================================================

# Replace previous get_*_tools implementations

def _filter_tools(keywords: List[str]) -> List[BaseTool]:
    all_tools = _AdaptHolder.get_all_tools()
    return filter_tools_by_keywords(all_tools, keywords)


def get_historian_tools() -> List[BaseTool]:
    """Memory/knowledge management tools"""
    memory_keywords = [
        'memory', 'search', 'create', 'add', 'node', 'observation',
        'relation', 'entities', 'graph'
    ]
    return _filter_tools(memory_keywords)


def get_researcher_tools() -> List[BaseTool]:
    """Research/Zotero/Context tools"""
    research_keywords = [
        'zotero', 'search', 'item', 'fulltext', 'metadata',
        'resolve', 'docs', 'library', 'context'
    ]
    return _filter_tools(research_keywords)


def get_archivist_tools() -> List[BaseTool]:
    analysis_keywords = ['thinking', 'sequential', 'thought', 'analyze']
    return _filter_tools(analysis_keywords)


def get_publisher_tools() -> List[BaseTool]:
    publish_keywords = ['create', 'write', 'publish', 'document', 'format']
    return _filter_tools(publish_keywords)


def get_context7_tools() -> List[BaseTool]:
    context7_keywords = ['resolve', 'docs', 'library', 'context']
    return _filter_tools(context7_keywords)

def get_all_mcp_tools() -> Dict[str, List[BaseTool]]:
    """Get all MCP tools organized by agent type."""
    return {
        "historian": get_historian_tools(),
        "researcher": get_researcher_tools(),
        "archivist": get_archivist_tools(),
        "publisher": get_publisher_tools(),
        "context7": get_context7_tools()
    }

# =============================================================================
# Backwards Compatibility
# =============================================================================

# Keep some basic tools for compatibility
class SchemaValidationTool(BaseTool):
    """Basic schema validation tool."""
    name: str = "schema_validation"
    description: str = "Validate data against research paper JSON schema"
    
    def _run(self, data: str) -> str:
        """Validate JSON data structure."""
        try:
            import json
            parsed = json.loads(data)
            return f"✅ Valid JSON with {len(parsed)} top-level keys"
        except Exception as e:
            return f"❌ Invalid JSON: {str(e)}"

class IntelligentSummaryTool(BaseTool):
    """Basic content summarization tool."""
    name: str = "intelligent_summary"
    description: str = "Generate intelligent summaries of research content"
    
    def _run(self, content: str, max_length: int = 500) -> str:
        """Generate a summary of the content."""
        # Simple summarization - truncate with ellipsis
        if len(content) <= max_length:
            return content
        
        # Find a good break point near max_length
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        break_point = max(last_period, last_space) if last_period > 0 or last_space > 0 else max_length
        
        return content[:break_point] + "..."

# Add basic tools to all agent toolsets
BASIC_TOOLS = [SchemaValidationTool(), IntelligentSummaryTool()]

def add_basic_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """Add basic tools to any toolset."""
    return tools + BASIC_TOOLS 

# =============================================================================
# Thread-safe Async Worker (sync-async bridge)
# =============================================================================

class _AsyncWorker:
    """Run a dedicated asyncio event loop in a background thread."""
    _loop: Any = None
    _thread: Any = None

    @classmethod
    def _run_loop(cls, loop):  # pragma: no cover
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            # Clean shutdown
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.close()

    @classmethod
    def get_loop(cls):
        if cls._loop is None or cls._loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=cls._run_loop, args=(loop,), daemon=True)
            thread.start()
            cls._loop = loop
            cls._thread = thread
        return cls._loop

    @classmethod
    def run(cls, coro):
        """Synchronously execute the coroutine in the background loop."""
        loop = cls.get_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)  # 30 second timeout
        except Exception as e:
            logger.error(f"AsyncWorker execution failed: {e}")
            raise


def _patch_tool(tool: BaseTool) -> BaseTool:
    """Ensure the BaseTool's synchronous interface uses background loop."""
    # Check if tool has async methods that need patching
    if hasattr(tool, "_arun") and callable(getattr(tool, "_arun")):
        original_arun = tool._arun

        def _run_sync(*args, **kwargs):  # type: ignore
            try:
                return _AsyncWorker.run(original_arun(*args, **kwargs))
            except Exception as e:
                logger.error(f"Tool {tool.name} execution failed: {e}")
                return f"Tool execution failed: {str(e)}"

        tool._run = _run_sync  # type: ignore
    
    # Also patch any other async methods that might be called
    for attr_name in dir(tool):
        if attr_name.startswith('_') and not attr_name.startswith('__'):
            attr = getattr(tool, attr_name)
            if callable(attr) and inspect.iscoroutinefunction(attr):
                # Create a sync wrapper for this async method
                async_method = attr
                
                def make_sync_wrapper(async_func):
                    @functools.wraps(async_func)
                    def sync_wrapper(*args, **kwargs):
                        try:
                            return _AsyncWorker.run(async_func(*args, **kwargs))
                        except Exception as e:
                            logger.error(f"Async method {async_func.__name__} failed: {e}")
                            return f"Method execution failed: {str(e)}"
                    return sync_wrapper
                
                setattr(tool, attr_name, make_sync_wrapper(async_method))
    
    return tool

# =============================================================================
# Enhanced Tool Validation and Error Handling
# =============================================================================

class MCPToolWrapper(BaseTool):
    """Wrapper for MCP tools with enhanced error handling."""
    
    def __init__(self, original_tool: BaseTool):
        super().__init__()
        self.original_tool = original_tool
        self.name = getattr(original_tool, 'name', 'unknown_tool')
        self.description = getattr(original_tool, 'description', 'MCP tool')
        
    def _run(self, *args, **kwargs) -> str:
        """Execute the wrapped tool with enhanced error handling."""
        try:
            # Try to call the original tool
            if hasattr(self.original_tool, '_run'):
                result = self.original_tool._run(*args, **kwargs)
            elif hasattr(self.original_tool, '_arun'):
                # Use async worker for async tools
                result = _AsyncWorker.run(self.original_tool._arun(*args, **kwargs))
            else:
                return f"Error: Tool {self.name} has no executable method"
            
            # Ensure result is a string
            if result is None:
                return f"Tool {self.name} executed successfully (no output)"
            
            return str(result)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool {self.name} failed: {error_msg}")
            
            # Handle specific MCP errors
            if "Event loop is closed" in error_msg:
                return f"MCP server connection error for {self.name}: Event loop closed"
            elif "Connection refused" in error_msg:
                return f"MCP server unavailable for {self.name}: Connection refused"
            elif "coroutine" in error_msg and "never awaited" in error_msg:
                return f"Async execution error for {self.name}: {error_msg}"
            else:
                return f"Tool {self.name} execution failed: {error_msg}"

# Apply patch to all tools retrieved from MCPAdapt
    