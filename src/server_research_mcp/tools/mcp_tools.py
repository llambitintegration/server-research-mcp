"""
MCPAdapt-based Tools System for Server Research MCP
Clean, simple MCP server integration using mcpadapt.
"""
from crewai.tools import BaseTool
from typing import List, Dict, Any, Optional, Type
import logging
import os
from ..utils.mcpadapt import MCPAdapt, CrewAIAdapter
from mcp import StdioServerParameters
import asyncio
import threading
import functools
import inspect
import re
from pydantic import BaseModel, Field, create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MCP Server Configuration
# =============================================================================

def get_mcp_server_configs() -> List[StdioServerParameters]:
    """Get MCP server configurations for all available servers."""
    
    servers = []
    
    # Core servers that should always work
    core_servers = [
        {
            "name": "Memory",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": {}
        },
        {
            "name": "Sequential Thinking", 
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "env": {}
        }
    ]
    
    # Add core servers
    for server in core_servers:
        try:
            servers.append(StdioServerParameters(
                command=server["command"],
                args=server["args"],
                env=server["env"]
            ))
            logger.info(f"✅ {server['name']} server configured")
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure {server['name']} server: {e}")
    
    # Filesystem server for publishing (essential for publisher)
    try:
        obsidian_vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
        publish_directory = obsidian_vault_path or os.path.join(os.getcwd(), "outputs")
        
        # Ensure directory exists
        os.makedirs(publish_directory, exist_ok=True)
        
        servers.append(StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", publish_directory],
            env={}
        ))
        logger.info(f"📁 Filesystem server configured for: {publish_directory}")
    except Exception as e:
        logger.error(f"❌ Filesystem server configuration failed: {e}")
    
    # Zotero server (if credentials available)
    try:
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
    except Exception as e:
        logger.warning(f"⚠️ Zotero server configuration failed: {e}")
    
    # Optional servers that might fail
    optional_servers = []
    
    # Context7 server (requires authentication)
    context7_token = os.getenv("CONTEXT7_TOKEN") or os.getenv("UPSTASH_TOKEN")
    if context7_token:
        optional_servers.append({
            "name": "Context7",
            "command": "npx", 
            "args": ["-y", "@upstash/context7-mcp"],
            "env": {"UPSTASH_TOKEN": context7_token}
        })
    else:
        logger.warning("⚠️ Context7 server skipped - missing CONTEXT7_TOKEN or UPSTASH_TOKEN")
    
    # Add optional servers with error handling
    for server in optional_servers:
        try:
            servers.append(StdioServerParameters(
                command=server["command"],
                args=server["args"], 
                env=server["env"]
            ))
            logger.info(f"✅ {server['name']} server configured")
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure {server['name']} server: {e}")
    
    logger.info(f"🔧 Total MCP servers configured: {len(servers)}")
    return servers

def validate_zotero_credentials() -> Dict[str, Any]:
    """Validate Zotero API credentials and configuration."""
    diagnosis = {
        "credentials_valid": False,
        "api_key_set": False,
        "library_id_set": False,
        "credentials_format_valid": False,
        "environment_status": {},
        "recommendations": []
    }
    
    # Check if credentials are set
    api_key = os.getenv("ZOTERO_API_KEY")
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    
    diagnosis["api_key_set"] = bool(api_key)
    diagnosis["library_id_set"] = bool(library_id)
    
    # Validate credential formats
    if api_key:
        # Zotero API keys are typically 40-character alphanumeric strings
        api_key_valid = len(api_key) >= 20 and api_key.isalnum()
        diagnosis["environment_status"]["api_key_format"] = "valid" if api_key_valid else "invalid_format"
    else:
        diagnosis["environment_status"]["api_key_format"] = "missing"
    
    if library_id:
        # Library IDs are typically numeric strings
        library_id_valid = library_id.isdigit() and len(library_id) > 0
        diagnosis["environment_status"]["library_id_format"] = "valid" if library_id_valid else "invalid_format"
    else:
        diagnosis["environment_status"]["library_id_format"] = "missing"
    
    diagnosis["credentials_format_valid"] = (
        diagnosis["environment_status"]["api_key_format"] == "valid" and
        diagnosis["environment_status"]["library_id_format"] == "valid"
    )
    
    diagnosis["credentials_valid"] = diagnosis["api_key_set"] and diagnosis["library_id_set"] and diagnosis["credentials_format_valid"]
    
    # Generate recommendations
    if not diagnosis["api_key_set"]:
        diagnosis["recommendations"].append("Set ZOTERO_API_KEY environment variable with your Zotero API key")
    elif diagnosis["environment_status"]["api_key_format"] != "valid":
        diagnosis["recommendations"].append("ZOTERO_API_KEY format appears invalid - should be 40-character alphanumeric string")
    
    if not diagnosis["library_id_set"]:
        diagnosis["recommendations"].append("Set ZOTERO_LIBRARY_ID environment variable with your library ID")
    elif diagnosis["environment_status"]["library_id_format"] != "valid":
        diagnosis["recommendations"].append("ZOTERO_LIBRARY_ID format appears invalid - should be numeric string")
    
    if not diagnosis["credentials_valid"]:
        diagnosis["recommendations"].append("Visit https://www.zotero.org/settings/keys to create API credentials")
    
    return diagnosis

async def test_zotero_api_connectivity() -> Dict[str, Any]:
    """Test direct Zotero API connectivity."""
    try:
        import aiohttp
    except ImportError:
        return {
            "api_reachable": False,
            "auth_valid": False,
            "library_accessible": False,
            "response_details": {},
            "error_details": {"import_error": "aiohttp not available for direct API testing"}
        }
    
    connectivity = {
        "api_reachable": False,
        "auth_valid": False,
        "library_accessible": False,
        "response_details": {},
        "error_details": {}
    }
    
    api_key = os.getenv("ZOTERO_API_KEY")
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    
    if not api_key or not library_id:
        connectivity["error_details"]["missing_credentials"] = "API key or library ID not set"
        return connectivity
    
    # Test basic API connectivity
    try:
        async with aiohttp.ClientSession() as session:
            # Test 1: Basic API endpoint
            async with session.get("https://api.zotero.org/") as response:
                connectivity["api_reachable"] = response.status == 200
                connectivity["response_details"]["base_api"] = response.status
            
            # Test 2: Authentication with user library access
            headers = {"Authorization": f"Bearer {api_key}"}
            library_url = f"https://api.zotero.org/users/{library_id}/collections"
            
            async with session.get(library_url, headers=headers) as response:
                connectivity["auth_valid"] = response.status in [200, 404]  # 404 is valid for empty library
                connectivity["library_accessible"] = response.status == 200
                connectivity["response_details"]["library_access"] = response.status
                
                if response.status == 401:
                    connectivity["error_details"]["auth_error"] = "Invalid API key or insufficient permissions"
                elif response.status == 403:
                    connectivity["error_details"]["auth_error"] = "API key lacks required permissions"
                elif response.status >= 400:
                    error_text = await response.text()
                    connectivity["error_details"]["api_error"] = f"HTTP {response.status}: {error_text}"
    
    except aiohttp.ClientError as e:
        connectivity["error_details"]["connection_error"] = str(e)
    except Exception as e:
        connectivity["error_details"]["unexpected_error"] = str(e)
    
    return connectivity

async def test_zotero_mcp_server_startup() -> Dict[str, Any]:
    """Test Zotero MCP server startup and tool availability."""
    server_test = {
        "server_starts": False,
        "tools_loaded": False,
        "tool_count": 0,
        "available_tools": [],
        "startup_error": None,
        "execution_test": {}
    }
    
    try:
        # Create Zotero server configuration
        api_key = os.getenv("ZOTERO_API_KEY")
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        
        if not api_key or not library_id:
            server_test["startup_error"] = "Missing Zotero credentials"
            return server_test
        
        zotero_config = StdioServerParameters(
            command="uvx",
            args=["zotero-mcp"],
            env={
                "ZOTERO_LOCAL": "false",
                "ZOTERO_API_KEY": api_key,
                "ZOTERO_LIBRARY_ID": library_id
            }
        )
        
        # Test server startup with MCPAdapt
        with MCPAdapt([zotero_config], CrewAIAdapter()) as tools:
            server_test["server_starts"] = True
            server_test["tools_loaded"] = len(tools) > 0
            server_test["tool_count"] = len(tools)
            server_test["available_tools"] = [tool.name for tool in tools]
            
            # Test a basic tool execution if tools are available
            if tools:
                zotero_tools = [tool for tool in tools if 'zotero' in tool.name.lower()]
                if zotero_tools:
                    try:
                        # Try a simple search operation
                        search_tool = zotero_tools[0]
                        result = search_tool._run("test")
                        server_test["execution_test"] = {
                            "tool_executed": True,
                            "tool_name": search_tool.name,
                            "result_length": len(str(result)),
                            "execution_error": None
                        }
                    except Exception as e:
                        server_test["execution_test"] = {
                            "tool_executed": False,
                            "execution_error": str(e)
                        }
    
    except Exception as e:
        server_test["startup_error"] = str(e)
    
    return server_test

def diagnose_mcp_servers() -> Dict[str, Any]:
    """Diagnose MCP server connectivity and return status report."""
    import subprocess
    import sys
    
    diagnosis = {
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd()
        },
        "environment_vars": {},
        "server_checks": {},
        "recommendations": []
    }
    
    # Check environment variables
    env_vars = [
        "OBSIDIAN_VAULT_PATH", "ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID", 
        "CONTEXT7_TOKEN", "UPSTASH_TOKEN"
    ]
    for var in env_vars:
        diagnosis["environment_vars"][var] = "SET" if os.getenv(var) else "MISSING"
    
    # Check if key commands are available
    commands_to_check = [
        ("npx", ["--version"]),
        ("uvx", ["--version"]),
        ("node", ["--version"])
    ]
    
    for cmd, args in commands_to_check:
        try:
            result = subprocess.run([cmd] + args, capture_output=True, text=True, timeout=10)
            diagnosis["server_checks"][cmd] = {
                "available": True,
                "version": result.stdout.strip() if result.returncode == 0 else "Error",
                "error": result.stderr.strip() if result.stderr else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            diagnosis["server_checks"][cmd] = {
                "available": False,
                "error": str(e)
            }
    
    # Generate recommendations
    if not diagnosis["server_checks"].get("npx", {}).get("available"):
        diagnosis["recommendations"].append("Install Node.js and npm to enable npx")
    
    if not diagnosis["server_checks"].get("uvx", {}).get("available"):
        diagnosis["recommendations"].append("Install uv package manager for uvx support")
    
    if diagnosis["environment_vars"]["ZOTERO_API_KEY"] == "MISSING":
        diagnosis["recommendations"].append("Set ZOTERO_API_KEY for research functionality")
    
    if diagnosis["environment_vars"]["OBSIDIAN_VAULT_PATH"] == "MISSING":
        diagnosis["recommendations"].append("Set OBSIDIAN_VAULT_PATH for publishing functionality")
    
    return diagnosis

# =============================================================================
# Tool Filtering Functions
# =============================================================================

def normalize_tool_name(name: str) -> str:
    """Normalize tool name by removing all non-alphanumeric characters and converting to lowercase."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def filter_tools_by_keywords(tools: List[BaseTool], keywords: List[str]) -> List[BaseTool]:
    """Filter tools by keywords in their names using normalized matching."""
    filtered = []
    normalized_keywords = [normalize_tool_name(keyword) for keyword in keywords]
    
    for tool in tools:
        tool_name_normalized = normalize_tool_name(getattr(tool, 'name', ''))
        if any(keyword in tool_name_normalized for keyword in normalized_keywords):
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
            
            # Try with all servers first
            try:
                ctx_local = MCPAdapt(server_configs, CrewAIAdapter())
                tools_local = ctx_local.__enter__()
                logger.info(f"✅ All MCP servers connected successfully ({len(tools_local)} tools)")
                return ctx_local, tools_local
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect all MCP servers: {e}")
                
                # Try with reduced server set (core servers only)
                try:
                    core_configs = [
                        server for server in server_configs 
                        if any(arg in server.args for arg in ["server-memory", "server-sequential-thinking", "server-filesystem"])
                    ]
                    
                    if core_configs:
                        logger.info(f"🔄 Retrying with {len(core_configs)} core servers...")
                        ctx_local = MCPAdapt(core_configs, CrewAIAdapter())
                        tools_local = ctx_local.__enter__()
                        logger.info(f"✅ Core MCP servers connected ({len(tools_local)} tools)")
                        return ctx_local, tools_local
                    else:
                        raise Exception("No core servers available")
                        
                except Exception as e2:
                    logger.error(f"❌ Failed to connect even core MCP servers: {e2}")
                    raise e2

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
            
            # Step 5a: Guarantee Zotero tool availability
            zotero_credentials_exist = bool(os.getenv("ZOTERO_API_KEY")) and bool(os.getenv("ZOTERO_LIBRARY_ID"))
            zotero_tools = [tool for tool in wrapped_tools if normalize_tool_name('zotero') in normalize_tool_name(tool.name)]
            
            if zotero_credentials_exist and not zotero_tools:
                raise Exception("Zotero credentials provided but no Zotero tools loaded - aborting fallback to basic tools")
            
            cls._ctx = ctx
            logger.info(f"🔧 MCP tools initialized: {len(wrapped_tools)} tools wrapped and ready")
            if zotero_credentials_exist:
                logger.info(f"✅ Zotero integration: {len(zotero_tools)} Zotero tools available")
            return wrapped_tools
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP tools: {e}")
            
            # Step 5b: Log explicit warning if fallback engaged
            zotero_credentials_exist = bool(os.getenv("ZOTERO_API_KEY")) and bool(os.getenv("ZOTERO_LIBRARY_ID"))
            if zotero_credentials_exist:
                logger.warning("⚠️ Zotero credentials detected but MCP tools failed to load - this may indicate a configuration issue")
            
            logger.info("🔄 Falling back to basic tools...")
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
        'memory', 'create_entities', 'read_graph', 'search_nodes', 'open_nodes'
    ]
    filtered_tools = _filter_tools(memory_keywords)
    
    # Ensure historian has some tools for basic functionality
    if not filtered_tools:
        logger.warning("⚠️ No MCP memory tools available for historian")
        filtered_tools = BASIC_TOOLS
    
    return filtered_tools


def get_researcher_tools() -> List[BaseTool]:
    """Research/Zotero/Context tools"""
    research_keywords = [
        'zotero_search_items', 'zotero_item_metadata', 'zotero_item_fulltext'
    ]
    filtered_tools = _filter_tools(research_keywords)
    
    # Ensure researcher has some tools for basic functionality
    if not filtered_tools:
        logger.warning("⚠️ No MCP research tools available for researcher")
        filtered_tools = BASIC_TOOLS
    
    return filtered_tools


def get_archivist_tools() -> List[BaseTool]:
    """Sequential thinking and analysis tools"""
    analysis_keywords = ['sequentialthinking', 'sequential_thinking']
    filtered_tools = _filter_tools(analysis_keywords)
    
    # Ensure archivist has some tools for basic functionality
    if not filtered_tools:
        logger.warning("⚠️ No MCP analysis tools available for archivist")
        filtered_tools = BASIC_TOOLS
    
    return filtered_tools


def get_publisher_tools() -> List[BaseTool]:
    """Obsidian/publishing tools with filesystem fallback"""
    publish_keywords = [
        'obsidian', 'note', 'vault', 'markdown', 'publish', 'create_note',
        'write_file', 'read_file', 'create_directory', 'list_directory', 'write_text_file'
    ]
    filtered_tools = _filter_tools(publish_keywords)
    
    # Always ensure we have basic publishing capability
    if not filtered_tools:
        logger.warning("⚠️ No MCP publishing tools available, adding basic tools")
        filtered_tools = BASIC_TOOLS
    
    return filtered_tools


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

def debug_mcp_setup() -> str:
    """Debug MCP setup and return diagnostic information."""
    try:
        diagnosis = diagnose_mcp_servers()
        
        report = ["🔍 MCP Setup Diagnostic Report", "=" * 40]
        
        # System info
        report.append("\n📋 System Information:")
        report.append(f"  Python: {diagnosis['system_info']['python_version'].split()[0]}")  
        report.append(f"  Platform: {diagnosis['system_info']['platform']}")
        report.append(f"  Working Directory: {diagnosis['system_info']['working_directory']}")
        
        # Environment variables
        report.append("\n🔧 Environment Variables:")
        for var, status in diagnosis["environment_vars"].items():
            icon = "✅" if status == "SET" else "❌"
            report.append(f"  {icon} {var}: {status}")
        
        # Command availability  
        report.append("\n⚙️ Command Availability:")
        for cmd, info in diagnosis["server_checks"].items():
            if info["available"]:
                report.append(f"  ✅ {cmd}: {info.get('version', 'Available')}")
            else:
                report.append(f"  ❌ {cmd}: {info.get('error', 'Not available')}")
        
        # Current tool status
        report.append("\n🔧 Current Tool Status:")
        try:
            all_tools = _AdaptHolder.get_all_tools()
            report.append(f"  ✅ Total tools loaded: {len(all_tools)}")
            
            tool_counts = get_all_mcp_tools()
            for agent, tools in tool_counts.items():
                report.append(f"    {agent.capitalize()}: {len(tools)} tools")
                
        except Exception as e:
            report.append(f"  ❌ Tool loading error: {str(e)}")
        
        # Recommendations
        if diagnosis["recommendations"]:
            report.append("\n💡 Recommendations:")
            for rec in diagnosis["recommendations"]:
                report.append(f"  • {rec}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"❌ Diagnostic failed: {str(e)}" 

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
    # Preserve args_schema before patching
    original_args_schema = getattr(tool, 'args_schema', None)
    
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
    
    # Restore args_schema after patching
    if original_args_schema is not None:
        tool.args_schema = original_args_schema
    
    return tool

# =============================================================================
# Enhanced Tool Validation and Error Handling
# =============================================================================

class MCPToolWrapper(BaseTool):
    """Wrapper for MCP tools with enhanced error handling and schema propagation."""
    original_tool: Any  # Declare the field for Pydantic
    args_schema: Type[BaseModel]  # Expose schema as class attribute
    
    def __init__(self, original_tool: BaseTool):
        # Extract name and description before calling super
        name = getattr(original_tool, 'name', 'unknown_tool')
        description = getattr(original_tool, 'description', 'MCP tool')
        
        # Step 1a: Read original_tool.args_schema if it exists
        if hasattr(original_tool, 'args_schema') and original_tool.args_schema is not None:
            args_schema = original_tool.args_schema
        else:
            # Step 1b: Build Pydantic model from _run signature
            args_schema = self._build_schema_from_run_signature(original_tool)
        
        # Initialize parent with required fields
        super().__init__(name=name, description=description, original_tool=original_tool, args_schema=args_schema)
    
    def _build_schema_from_run_signature(self, tool: BaseTool) -> Type[BaseModel]:
        """Build a Pydantic model from the tool's _run method signature."""
        if hasattr(tool, '_run') and callable(tool._run):
            try:
                sig = inspect.signature(tool._run)
                field_definitions = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name in ('self', 'args', 'kwargs'):
                        continue
                    
                    # Determine type from annotation or default to str
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                    
                    # Handle default values
                    if param.default != inspect.Parameter.empty:
                        field_definitions[param_name] = (param_type, Field(default=param.default))
                    else:
                        field_definitions[param_name] = (param_type, Field(...))
                
                # Create model with field definitions
                if field_definitions:
                    return create_model(f"{tool.name}Schema", **field_definitions)
                else:
                    # Fallback: create simple schema with common parameters
                    return create_model(f"{tool.name}Schema", 
                                      query=(str, Field(default=None, description="Query parameter")))
            except Exception as e:
                logger.debug(f"Could not build schema from {tool.name} signature: {e}")
        
        # Fallback: create a basic schema
        return create_model(f"{tool.name}Schema", 
                          query=(str, Field(default=None, description="Query parameter")))
    
    def _original_tool_needs_positional_args(self) -> bool:
        """Check if the original tool expects positional arguments."""
        if not hasattr(self.original_tool, '_run'):
            return False
        
        try:
            sig = inspect.signature(self.original_tool._run)
            # Check if there are positional parameters (excluding self)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    if param_name not in ('args', 'kwargs'):
                        return True
            return False
        except Exception:
            return False
    
    def _convert_kwargs_to_positional(self, kwargs: dict) -> tuple:
        """Convert kwargs to positional args based on the original tool's signature."""
        if not hasattr(self.original_tool, '_run'):
            return (), kwargs
        
        try:
            sig = inspect.signature(self.original_tool._run)
            args = []
            remaining_kwargs = kwargs.copy()
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    if param_name in remaining_kwargs:
                        args.append(remaining_kwargs.pop(param_name))
                    elif param.default == inspect.Parameter.empty:
                        # Required parameter missing
                        break
            
            return tuple(args), remaining_kwargs
        except Exception:
            return (), kwargs
        
    def _run(self, **kwargs) -> str:
        """Execute the wrapped tool with enhanced error handling. Step 2a: kwargs-only signature."""
        try:
            import json  # Local import to avoid global dependency
            
            # Enhanced logging for debugging
            logger.info(f"🔧 MCPToolWrapper {self.name} ENTRY - kwargs={kwargs}")
            original_kwargs = kwargs.copy()
            
            # Step 1: CrewAI sometimes wraps kwargs in a single "properties" dict
            if len(kwargs) == 1 and 'properties' in kwargs:
                logger.info(f"🔧 {self.name} - Step 1: Unwrapping 'properties' dict")
                kwargs = kwargs['properties']
                logger.info(f"🔧 {self.name} - After Step 1: kwargs={kwargs}")

            # Step 2: Handle legacy positional arg calls by converting first string arg to kwargs
            # This preserves backward compatibility while enforcing kwargs-only interface
            if len(kwargs) == 1 and list(kwargs.keys())[0] not in ['query', 'properties']:
                # Check if the value looks like a JSON string
                first_key = list(kwargs.keys())[0]
                first_value = kwargs[first_key]
                if isinstance(first_value, str):
                    try:
                        parsed = json.loads(first_value)
                        if isinstance(parsed, dict):
                            logger.info(f"🔧 {self.name} - Converting JSON string to kwargs: {parsed}")
                            kwargs = parsed
                    except json.JSONDecodeError:
                        # Not JSON, treat as query parameter
                        kwargs = {'query': first_value}

            # After preprocessing, log the final payload in debug mode
            logger.info(f"🔧 {self.name} AFTER PROCESSING - kwargs={kwargs}")

            # Step 2b: Convert kwargs to positional tuple only for tools whose original signature requires it
            if self._original_tool_needs_positional_args():
                args, remaining_kwargs = self._convert_kwargs_to_positional(kwargs)
                logger.info(f"🔧 {self.name} - Converting to positional: args={args}, kwargs={remaining_kwargs}")
                kwargs = remaining_kwargs
            else:
                args = ()

            # Validate that we have meaningful parameters for tools that require them
            if 'search' in self.name.lower() and not args and not kwargs:
                logger.error(f"🔧 VALIDATION FAILED for {self.name}:")
                logger.error(f"   Original input: kwargs={original_kwargs}")
                logger.error(f"   Final processed: args={args}, kwargs={kwargs}")
                return f"Error: {self.name} requires search parameters but none were provided"
            
            # Debug logging for search tools
            if 'search' in self.name.lower():
                logger.debug(f"Search tool {self.name} - args: {args}, kwargs: {kwargs}")
            
            # Special handling for search_nodes - ensure query parameter is present and valid
            if 'search_nodes' in self.name.lower():
                query_value = None
                if args:
                    query_value = str(args[0]) if args[0] is not None else None
                elif 'query' in kwargs:
                    query_value = str(kwargs['query']) if kwargs['query'] is not None else None
                
                if not query_value or not query_value.strip():
                    return f"Error: {self.name} requires a non-empty 'query' parameter for searching. Received args={args}, kwargs={kwargs}"

            # Try to call the original tool
            if hasattr(self.original_tool, '_run'):
                if args:
                    result = self.original_tool._run(*args, **kwargs)
                else:
                    result = self.original_tool._run(**kwargs)
            elif hasattr(self.original_tool, '_arun'):
                # Use async worker for async tools
                if args:
                    result = _AsyncWorker.run(self.original_tool._arun(*args, **kwargs))
                else:
                    result = _AsyncWorker.run(self.original_tool._arun(**kwargs))
            else:
                return f"Error: Tool {self.name} has no executable method"
            
            # Ensure result is a string
            if result is None:
                return f"Tool {self.name} executed successfully (no output)"
            
            return str(result)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool {self.name} failed: {error_msg}")
            
            # Handle specific MCP errors with more detailed diagnostics
            if "Cannot read properties of undefined" in error_msg and "toLowerCase" in error_msg:
                return f"Parameter validation error for {self.name}: A required string parameter was undefined. This typically happens when a 'query' parameter is missing or empty. Args provided: {args if 'args' in locals() else []}, Kwargs provided: {kwargs}. Please ensure all required text parameters are provided."
            elif "toLowerCase" in error_msg:
                return f"String parameter error for {self.name}: A parameter expected to be a string was undefined or null. Args: {args if 'args' in locals() else []}, Kwargs: {kwargs}. Check that all text parameters are properly set."
            elif "Event loop is closed" in error_msg:
                return f"MCP server connection error for {self.name}: Event loop closed"
            elif "Connection refused" in error_msg:
                return f"MCP server unavailable for {self.name}: Connection refused"
            elif "coroutine" in error_msg and "never awaited" in error_msg:
                return f"Async execution error for {self.name}: {error_msg}"
            else:
                return f"Tool {self.name} execution failed: {error_msg}"

# Export debug function for easy import
__all__ = [
    'get_all_mcp_tools', 'get_historian_tools', 'get_researcher_tools', 
    'get_archivist_tools', 'get_publisher_tools', 'get_context7_tools',
    'debug_mcp_setup', 'diagnose_mcp_servers', 'validate_zotero_credentials',
    'test_zotero_api_connectivity', 'test_zotero_mcp_server_startup'
]

# Apply patch to all tools retrieved from MCPAdapt
    