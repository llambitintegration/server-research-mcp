"""
Enhanced MCP Manager for Server Research MCP
==========================================

Production-ready MCP server integration with schema fixing, context management,
and robust error handling based on proven patterns from obsidianCrew-mcp.

Features:
- Schema fixing for MCP tool compatibility issues
- Context manager patterns for proper server lifecycle
- Enhanced error handling and timeout protection
- Support for both official and custom MCP adapters
- Performance monitoring and logging
- Graceful fallbacks and recovery patterns
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from pathlib import Path

# Performance monitoring
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Environment setup for non-interactive mode
def setup_non_interactive_environment():
    """Set environment variables to prevent interactive prompts during testing."""
    if any(os.environ.get(var) for var in ["AUTOMATED_TESTING", "CI", "PYTEST_CURRENT_TEST"]) or "pytest" in sys.modules:
        os.environ["CLICK_CONFIRM_DEFAULT"] = "n"
        os.environ["AUTOMATED_TESTING"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["CI"] = "1"


# ============================================================================
# SCHEMA FIXING UTILITIES (from obsidianCrew-mcp reference)
# ============================================================================

def fix_schema_anyof_const_issue(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix MCP schema anyOf const issues that cause Union[tuple([])] errors.
    Converts anyOf arrays with only const values to enum types.
    """
    if not isinstance(schema, dict):
        return schema

    fixed_schema: Dict[str, Any] = {}

    for key, value in schema.items():
        if key == "anyOf" and isinstance(value, list):
            if not value:
                fixed_schema["type"] = "string"
                continue

            const_values = []
            has_non_const = False
            valid_items = []

            for item in value:
                if isinstance(item, dict) and "const" in item and len(item) == 1:
                    const_values.append(item["const"])
                elif isinstance(item, dict) and item:
                    has_non_const = True
                    valid_items.append(fix_schema_anyof_const_issue(item))
                elif isinstance(item, dict):
                    continue
                else:
                    has_non_const = True
                    valid_items.append(item)

            if not has_non_const and const_values:
                if all(isinstance(v, str) for v in const_values):
                    fixed_schema["type"] = "string"
                    fixed_schema["enum"] = const_values
                else:
                    fixed_schema[key] = value
            elif valid_items:
                fixed_schema[key] = valid_items
            else:
                fixed_schema["type"] = "string"

        elif isinstance(value, dict):
            fixed_schema[key] = fix_schema_anyof_const_issue(value)
        elif isinstance(value, list):
            fixed_schema[key] = [
                fix_schema_anyof_const_issue(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            fixed_schema[key] = value

    return fixed_schema


def find_node_command(command: str) -> str:
    """Find the full path to a Node.js command (npx, npm, node)."""
    # First try shutil.which which handles Windows PATH properly
    full_path = shutil.which(command)
    if full_path:
        logger.debug(f"Found {command} via PATH: {full_path}")
        return full_path

    # Common Windows paths for Node.js
    common_paths = [
        r"C:\Program Files\nodejs",
        r"C:\Program Files (x86)\nodejs",
        os.path.expanduser(r"~\AppData\Roaming\npm"),
        os.path.expanduser(r"~\AppData\Local\Programs\nodejs"),
        r"C:\Users\ackin\miniconda3\envs\mcp2\Scripts",
        r"C:\Users\ackin\miniconda3\Scripts",
    ]

    for path in common_paths:
        for ext in [".cmd", ".exe", ""]:  # Include no extension too
            full_path = os.path.join(path, f"{command}{ext}")
            if os.path.exists(full_path):
                logger.debug(f"Found {command} at: {full_path}")
                return full_path

    # Last resort: try the command as-is and let subprocess handle it
    logger.warning(f"Could not find {command} in PATH or common locations, trying as-is")
    return command


# ============================================================================
# ENHANCED MCP ADAPTER WITH SCHEMA FIXING
# ============================================================================

class EnhancedMCPAdapter:
    """Enhanced MCP adapter with schema fixing and production features."""

    def __init__(self, server_config, use_schema_fixing: bool = True):
        self.server_config = server_config
        self.use_schema_fixing = use_schema_fixing
        self.process = None
        self._tools = None
        self._is_context_managed = False
        self.start_time = None
        self.performance_metrics = {}

    def __enter__(self):
        """Context manager entry."""
        self._is_context_managed = True
        self.start()
        return self.tools

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self._is_context_managed = False

    def start(self):
        """Start the MCP server with enhanced error handling."""
        setup_non_interactive_environment()
        self.start_time = datetime.now()
        
        logger.info(f"Starting Enhanced MCP Adapter for {self.server_config.name}")

        try:
            # Prepare environment with UTF-8 encoding
            env = os.environ.copy()
            env.update(self.server_config.env)
            # Force UTF-8 encoding to prevent charmap codec issues
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LC_ALL'] = 'C.UTF-8'
            env['LANG'] = 'C.UTF-8'

            # Resolve command path properly on Windows
            resolved_command = find_node_command(self.server_config.command)
            logger.debug(f"Resolved command: {resolved_command}")

            # Start server process with timeout protection
            self.process = subprocess.Popen(
                [resolved_command] + self.server_config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # Explicitly set encoding
                errors='replace',  # Handle encoding errors gracefully
                env=env,
            )

            if not self.process.stdin or not self.process.stdout:
                raise RuntimeError("Failed to create MCP server process")

            # Initialize server with timeout
            self._initialize_server_with_timeout()
            
            # Record performance metrics
            self.performance_metrics['startup_time'] = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Server {self.server_config.name} started in {self.performance_metrics['startup_time']:.2f}s")

        except Exception as e:
            logger.error(f"Failed to start {self.server_config.name}: {e}")
            if self.process:
                self._cleanup_process()
            raise RuntimeError(f"Failed to start Enhanced MCP Adapter: {e}")

    def _initialize_server_with_timeout(self, timeout: int = 30):
        """Initialize server with timeout protection."""
        def initialize():
            try:
                # Send initialize message
                init_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "enhanced-mcp-adapter", "version": "2.0.0"},
                    },
                }

                self.process.stdin.write(json.dumps(init_msg) + "\n")
                self.process.stdin.flush()
                response = json.loads(self.process.stdout.readline())
                logger.debug(f"Initialize response: {response}")

                # Send initialized notification
                self.process.stdin.write(
                    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
                )
                self.process.stdin.flush()

                # Get tools
                tools_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
                self.process.stdin.write(json.dumps(tools_msg) + "\n")
                self.process.stdin.flush()

                tools_response = json.loads(self.process.stdout.readline())
                
                if "result" in tools_response and "tools" in tools_response["result"]:
                    original_tools = tools_response["result"]["tools"]
                    logger.info(f"Found {len(original_tools)} tools from {self.server_config.name}")

                    # Special initialization for memory server
                    if self.server_config.name == "memory":
                        logger.info("Initializing memory server with empty graph...")
                        try:
                            # Try to read the current graph to initialize memory state
                            read_graph_msg = {
                                "jsonrpc": "2.0",
                                "id": 3,
                                "method": "tools/call",
                                "params": {"name": "read_graph", "arguments": {}}
                            }
                            self.process.stdin.write(json.dumps(read_graph_msg) + "\n")
                            self.process.stdin.flush()
                            
                            read_response = json.loads(self.process.stdout.readline())
                            logger.debug(f"Memory graph read response: {read_response}")
                        except Exception as e:
                            logger.warning(f"Memory graph initialization warning: {e}")

                    # Special initialization for context7 server
                    elif self.server_config.name == "context7":
                        logger.info("Initializing context7 server...")
                        try:
                            # Log available tools for debugging
                            tool_names = [tool.get("name", "unknown") for tool in original_tools]
                            logger.info(f"Context7 available tools: {', '.join(tool_names)}")
                            
                            # Test basic connectivity with a simple call if available
                            if any("resolve" in tool.get("name", "") for tool in original_tools):
                                logger.debug("Context7 server ready for library resolution")
                            else:
                                logger.warning("Context7 server missing expected resolve tools")
                                
                        except Exception as e:
                            logger.warning(f"Context7 initialization warning: {e}")

                    # Apply schema fixes if enabled
                    if self.use_schema_fixing:
                        fixed_tools = []
                        for tool in original_tools:
                            fixed_tool = tool.copy()
                            if "inputSchema" in fixed_tool:
                                fixed_tool["inputSchema"] = fix_schema_anyof_const_issue(
                                    fixed_tool["inputSchema"]
                                )
                            fixed_tools.append(fixed_tool)
                        self._create_enhanced_tools(fixed_tools)
                    else:
                        self._create_enhanced_tools(original_tools)
                        
                else:
                    raise RuntimeError("No tools found in MCP server response")

            except Exception as e:
                logger.error(f"Server initialization failed: {e}")
                raise

        # Run initialization with timeout
        init_thread = threading.Thread(target=initialize)
        init_thread.daemon = True
        init_thread.start()
        init_thread.join(timeout=timeout)

        if init_thread.is_alive():
            logger.error(f"Server initialization timed out after {timeout}s")
            self._cleanup_process()
            raise RuntimeError(f"Server initialization timed out")

    def _create_enhanced_tools(self, tool_definitions):
        """Create enhanced tool wrappers with monitoring."""
        # For now, store tool definitions - in production would create CrewAI tools
        self._tools = tool_definitions
        logger.info(f"Created {len(tool_definitions)} enhanced tools")

    def _call_mcp_tool_with_monitoring(self, tool_name: str, kwargs: Dict[str, Any]):
        """Enhanced MCP tool execution with monitoring and error handling."""
        # Validate arguments based on tool type
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Apply server-specific parameter validation
        if self.server_config.name == "memory":
            try:
                filtered_kwargs = self._validate_memory_parameters(tool_name, filtered_kwargs)
            except Exception as e:
                logger.error(f"Memory parameter validation failed for {tool_name}: {e}")
                raise
        elif self.server_config.name == "context7":
            try:
                filtered_kwargs = self._validate_context7_parameters(tool_name, filtered_kwargs)
            except Exception as e:
                logger.error(f"Context7 parameter validation failed for {tool_name}: {e}")
                raise
        
        start_time = datetime.now()
        logger.info(f"Calling {self.server_config.name}.{tool_name} with args: {filtered_kwargs}")
        
        try:
            # Prepare request
            call_msg = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": filtered_kwargs},
            }
            
            # Execute with timeout
            response = self._send_request_with_timeout(call_msg, timeout=30)
            
            # Check for MCP-level errors first
            if "error" in response:
                error_code = response["error"].get("code", -1)
                error_message = response["error"].get("message", "Unknown error")
                
                # Handle specific memory server toLowerCase error
                if (self.server_config.name == "memory" and 
                    tool_name == "search_nodes" and 
                    "toLowerCase" in error_message and "undefined" in error_message):
                    
                    logger.error(f"Memory server toLowerCase() bug detected: {error_message}")
                    
                    # Try fallback with non-empty query if original was empty
                    if not filtered_kwargs.get("query"):
                        logger.info("Attempting fallback with non-empty query to work around memory server bug")
                        fallback_kwargs = filtered_kwargs.copy()
                        fallback_kwargs["query"] = " "  # Single space as minimal non-empty query
                        
                        try:
                            fallback_msg = {
                                "jsonrpc": "2.0", 
                                "id": 4,
                                "method": "tools/call",
                                "params": {"name": tool_name, "arguments": fallback_kwargs},
                            }
                            fallback_response = self._send_request_with_timeout(fallback_msg, timeout=30)
                            
                            if "result" in fallback_response:
                                logger.info("Memory server fallback query succeeded")
                                return fallback_response["result"]
                        except Exception as fallback_error:
                            logger.warning(f"Memory server fallback also failed: {fallback_error}")
                    
                    # If fallback failed or wasn't applicable, provide helpful error
                    raise ValueError(
                        f"Memory server has a toLowerCase() bug that prevents searching. "
                        f"This is a known issue with the @modelcontextprotocol/server-memory package. "
                        f"Original error: {error_message}. "
                        f"Suggested fix: Update the memory server or use a different implementation."
                    )
                
                # Handle specific context7 server errors
                elif self.server_config.name == "context7":
                    logger.error(f"Context7 server error detected: {error_message}")
                    
                    # Check if we're using incorrect tool names
                    if tool_name in ["context7_resolve", "context7_get_docs"]:
                        correct_tool = "resolve-library-id" if tool_name == "context7_resolve" else "get-library-docs"
                        logger.warning(f"Using legacy tool name '{tool_name}', actual Context7 API uses '{correct_tool}'")
                    
                    # Provide helpful context7-specific error information
                    if "invalid json" in error_message.lower() or "parse" in error_message.lower():
                        raise ValueError(
                            f"Context7 server communication error. Server may be sending non-JSON output. "
                            f"Original error: {error_message}. "
                            f"Suggested fix: Check if you're using the correct Context7 package (@upstash/context7-mcp) and tool names (resolve-library-id, get-library-docs)."
                        )
                    elif "timeout" in error_message.lower():
                        raise ValueError(
                            f"Context7 server timed out. This may indicate network issues or server overload. "
                            f"Original error: {error_message}. "
                            f"Suggested fix: Check network connectivity or try again later."
                        )
                    elif "not found" in error_message.lower():
                        raise ValueError(
                            f"Context7 library or resource not found. "
                            f"Original error: {error_message}. "
                            f"Suggested fix: Verify the library name is correct and available in Context7."
                        )
                    elif "invalid argument" in error_message.lower():
                        raise ValueError(
                            f"Context7 server rejected arguments. Check parameter names and formats. "
                            f"Original error: {error_message}. "
                            f"Expected tools: resolve-library-id (libraryName), get-library-docs (context7CompatibleLibraryID, topic, tokens)."
                        )
                    else:
                        raise ValueError(
                            f"Context7 server error: {error_message}. "
                            f"This may be due to server configuration, connectivity issues, or incorrect tool/parameter names."
                        )
                
                # Handle other MCP errors
                raise ValueError(f"MCP tool error: {response['error']}")
            
            # Return successful result
            if "result" in response:
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ {tool_name} completed successfully in {duration:.2f}s")
                return response["result"]
            else:
                raise ValueError("No result or error in MCP response")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error calling {tool_name} after {duration:.2f}s: {e}")
            raise

    def _validate_memory_parameters(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and transform parameters for memory server tools."""
        if tool_name == "search_nodes":
            # Ensure query is a non-empty string and handle the toLowerCase() undefined bug
            query = kwargs.get("query", "")
            if not query or not isinstance(query, str):
                # Use empty string fallback instead of passing undefined/None
                kwargs["query"] = ""
                logger.warning(f"search_nodes: Invalid or missing query parameter, using empty string fallback")
            else:
                # Ensure the query is properly formatted as a string
                kwargs["query"] = str(query).strip()
                
            # Add additional safety check for the memory server bug
            # Some memory servers crash on empty queries due to toLowerCase() on undefined
            if not kwargs["query"]:
                logger.info("search_nodes: Empty query provided, this may cause memory server toLowerCase() errors")
                
            return kwargs
        
        elif tool_name == "create_entities":
            # Handle both single entity and batch creation modes
            if "entities" in kwargs:
                # Batch mode - ensure entities is a proper list
                entities = kwargs.get("entities", [])
                if not isinstance(entities, list):
                    raise ValueError("entities parameter must be a list")
                
                # Validate each entity in the batch (API expects name, entityType, observations)
                validated_entities = []
                for entity in entities:
                    if isinstance(entity, dict):
                        validated_entity = {
                            "name": str(entity.get("name", "")),
                            "entityType": str(entity.get("entity_type", entity.get("entityType", "concept"))),
                            "observations": entity.get("observations", [])
                        }
                        if validated_entity["name"]:  # Only add if name is not empty
                            validated_entities.append(validated_entity)
                
                return {"entities": validated_entities}
            
            else:
                # Single entity mode - convert to batch format that the API expects
                name = kwargs.get("name", "")
                entity_type = kwargs.get("entity_type", "concept")
                observations = kwargs.get("observations", [])
                
                if not name:
                    raise ValueError("name parameter is required for single entity creation")
                
                return {
                    "entities": [{
                        "name": str(name),
                        "entityType": str(entity_type),
                        "observations": observations if isinstance(observations, list) else []
                    }]
                }
        
        elif tool_name == "add_observations":
            # API expects observations array with entityName and contents
            entity_name = kwargs.get("entity_name", kwargs.get("entityName", ""))
            observations = kwargs.get("observations", [])
            
            if not entity_name:
                raise ValueError("entity_name parameter is required")
            
            return {
                "observations": [{
                    "entityName": str(entity_name),
                    "contents": observations if isinstance(observations, list) else [str(observations)]
                }]
            }
        
        return kwargs

    def _validate_context7_parameters(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and transform parameters for context7 server tools."""
        if tool_name == "resolve-library-id":
            # Convert library_name to libraryName for the actual Context7 API
            library_name = kwargs.get("library_name", kwargs.get("libraryName", ""))
            if not library_name or not isinstance(library_name, str):
                logger.warning(f"resolve-library-id: Invalid or missing libraryName parameter")
                kwargs["libraryName"] = ""
            else:
                kwargs["libraryName"] = str(library_name).strip()
                
            # Remove old parameter names
            if "library_name" in kwargs:
                kwargs.pop("library_name")
            if "query" in kwargs:
                kwargs.pop("query")
                
            logger.info(f"resolve-library-id: Validated libraryName parameter: '{kwargs['libraryName']}'")
            return kwargs
            
        elif tool_name == "get-library-docs":
            # Ensure context7CompatibleLibraryID is a valid string
            library_id = kwargs.get("context7_library_id", kwargs.get("context7CompatibleLibraryID", ""))
            if not library_id or not isinstance(library_id, str):
                raise ValueError("context7CompatibleLibraryID parameter is required and must be a non-empty string")
                
            kwargs["context7CompatibleLibraryID"] = str(library_id)
            
            # Remove old parameter names
            if "context7_library_id" in kwargs:
                kwargs.pop("context7_library_id")
            
            # Ensure topic is a string (optional)
            topic = kwargs.get("topic", "")
            kwargs["topic"] = str(topic) if topic else ""
            
            # Ensure tokens is a valid integer (minimum 10000 according to docs)
            tokens = kwargs.get("tokens", 10000)
            try:
                kwargs["tokens"] = max(int(tokens), 10000)  # Context7 enforces minimum 10000
            except (ValueError, TypeError):
                kwargs["tokens"] = 10000
                
            logger.info(f"get-library-docs: Validated parameters - libraryID: '{library_id}', topic: '{kwargs['topic']}', tokens: {kwargs['tokens']}")
            return kwargs
            
        return kwargs

    def _send_request_with_timeout(self, request: Dict[str, Any], timeout: int = 30):
        """Send request with timeout protection."""
        response_container = [None]
        error_container = [None]

        def send_and_receive():
            try:
                message = json.dumps(request) + "\n"
                logger.debug(f"Sending to MCP server: {message.strip()}")
                
                self.process.stdin.write(message)
                self.process.stdin.flush()
                
                line = self.process.stdout.readline()
                logger.debug(f"Received from MCP server: {line.strip()}")
                
                if line:
                    try:
                        response_container[0] = json.loads(line)
                    except json.JSONDecodeError as json_err:
                        # Server sent non-JSON output - likely error messages or logs
                        logger.error(f"MCP server sent non-JSON response: {line.strip()}")
                        error_container[0] = f"Server sent invalid JSON: {line.strip()[:200]}... (JSON error: {json_err})"
                else:
                    error_container[0] = "Empty response from MCP server"
            except BrokenPipeError:
                error_container[0] = "MCP server process terminated unexpectedly"
            except OSError as os_err:
                if os_err.errno == 22:  # Invalid argument
                    error_container[0] = f"Communication error with MCP server - server may have crashed or be misconfigured: {os_err}"
                else:
                    error_container[0] = f"OS error communicating with MCP server: {os_err}"
            except Exception as e:
                error_container[0] = f"Failed to send/receive: {e}"

        thread = threading.Thread(target=send_and_receive)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            error_container[0] = f"Request timed out after {timeout}s"

        if error_container[0]:
            raise RuntimeError(error_container[0])

        if response_container[0] is None:
            raise RuntimeError("No response received")

        return response_container[0]

    def _cleanup_process(self):
        """Clean up server process."""
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error during process cleanup: {e}")
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None

    @property
    def tools(self):
        """Get available tools."""
        return self._tools or []

    def stop(self):
        """Stop the MCP server with comprehensive cleanup."""
        if self.process:
            logger.info(f"Stopping {self.server_config.name}")
            self._cleanup_process()
            
            # Log performance metrics
            if self.start_time:
                total_time = (datetime.now() - self.start_time).total_seconds()
                logger.info(f"Server {self.server_config.name} ran for {total_time:.2f}s")
                
            if self.performance_metrics:
                logger.debug(f"Performance metrics for {self.server_config.name}: {self.performance_metrics}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            'server_name': self.server_config.name,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'tool_metrics': self.performance_metrics
        }


# ============================================================================
# ENHANCED MCP MANAGER
# ============================================================================

class EnhancedMCPManager:
    """Enhanced MCP Manager with production features and monitoring."""

    def __init__(self, use_schema_fixing: bool = True, enable_monitoring: bool = True):
        """Initialize Enhanced MCP Manager."""
        setup_non_interactive_environment()
        
        self.use_schema_fixing = use_schema_fixing
        self.enable_monitoring = enable_monitoring
        self.adapters: Dict[str, EnhancedMCPAdapter] = {}
        self.connection_status: Dict[str, bool] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self._initialized = False
        
        # Import server configs from existing manager for compatibility
        from .mcp_manager import MCPManager
        self.server_configs = MCPManager.SERVER_CONFIGS
        
        logger.info("Enhanced MCP Manager initialized")

    async def initialize(self, servers: Optional[List[str]] = None, timeout_per_server: int = 30):
        """Initialize MCP servers with enhanced error handling."""
        # Allow incremental initialization of additional servers even after the first call.
        servers_to_init = servers or list(self.server_configs.keys())

        # Filter out servers that are already connected unless we want to re-init them.
        pending_servers = [
            s for s in servers_to_init
            if s not in self.connection_status or s not in self.adapters
        ]

        if not pending_servers:
            logger.info("Enhanced MCP Manager already initialized for requested servers")
            return self.connection_status

        if self._initialized and pending_servers:
            logger.info(f"Initializing additional MCP servers: {pending_servers}")

        # Use pending_servers list for subsequent logic.
        servers_to_init = pending_servers

        initialization_results = {}
        
        for server_name in servers_to_init:
            if server_name not in self.server_configs:
                logger.warning(f"Unknown server: {server_name}")
                initialization_results[server_name] = False
                continue

            try:
                success = await self._initialize_server(server_name, timeout_per_server)
                initialization_results[server_name] = success
                self.connection_status[server_name] = success
                
                if success:
                    logger.info(f"✅ {server_name}: Connected successfully")
                else:
                    logger.warning(f"❌ {server_name}: Failed to connect")
                    
            except Exception as e:
                logger.error(f"Error initializing {server_name}: {e}")
                initialization_results[server_name] = False
                self.connection_status[server_name] = False

        # Mark as initialized when we have at least one server connected.
        if any(self.connection_status.values()):
            self._initialized = True
        
        # Log summary
        successful = sum(initialization_results.values())
        total = len(initialization_results)
        logger.info(f"Enhanced MCP initialization complete: {successful}/{total} servers connected")
        
        return initialization_results

    async def _initialize_server(self, server_name: str, timeout: int) -> bool:
        """Initialize individual server with timeout and error handling."""
        try:
            config = self.server_configs[server_name]
            
            import os
            require_real = os.getenv("REQUIRE_REAL_MCP", "false").lower() == "true"

            # Test server availability first; if unavailable, optionally fall back to MockMCPClient
            try:
                if not await self._test_server_availability(config, timeout=10):
                    raise RuntimeError(f"Server {server_name} not available for connection")
                
                # Create enhanced adapter
                adapter = EnhancedMCPAdapter(config, use_schema_fixing=self.use_schema_fixing)
                
                # Try to start the real MCP server
                adapter.start()
                self.adapters[server_name] = adapter
                
                if self.enable_monitoring:
                    self.performance_metrics[server_name] = adapter.get_performance_metrics()
                
                logger.info(f"Successfully initialized real MCP server: {server_name}")
                return True
                
            except Exception as real_server_error:
                logger.warning("⚠️ Failed to initialize real MCP server %s: %s", server_name, real_server_error, exc_info=True)
                
                if require_real:
                    logger.error(f"Server {server_name} not available and REQUIRE_REAL_MCP=true; aborting initialization.")
                    return False

                logger.warning(f"Server {server_name} not available – falling back to mock client")

                # Fallback to lightweight in-process mock client
                from .mcp_manager import MockMCPClient  # Local import to avoid circular deps

                mock_client = MockMCPClient(server_name)
                await mock_client.connect()

                # Store mock client as adapter for compatibility
                self.adapters[server_name] = mock_client

                if self.enable_monitoring:
                    self.performance_metrics[server_name] = {
                        "server_name": server_name,
                        "uptime": 0,
                        "tool_metrics": {},
                        "mock_mode": True
                    }

                logger.info(f"Successfully initialized mock MCP client: {server_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize server {server_name}: {e}")
            return False

    async def _test_server_availability(self, config, timeout: int = 10) -> bool:
        """Test if server command is available by checking if the command can be found."""
        try:
            # For MCP servers, just check if the command (npx) exists and the package can be resolved
            # This is much more reliable than trying to start/stop the server
            import shutil, subprocess, os

            cmd_path = shutil.which(config.command)
            logger.debug("_test_server_availability: command=%s resolved_path=%s PATH=%s", config.command, cmd_path, os.environ.get("PATH"))

            # Check if base command exists
            if not cmd_path:
                logger.warning("🔍 %s not found on PATH → will fall back to mock if real server is required", config.command)
                return False

            # For npx-based servers, we can assume they're available if npx works
            # MCP servers are designed to be run-on-demand via npx
            if config.command == "npx":
                return True
                
            # For other commands, do a quick executable check
            try:
                process = subprocess.Popen(
                    [config.command, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=5)
                logger.debug("%s --version stdout=%s stderr=%s returncode=%s", config.command, stdout.strip(), stderr.strip(), process.returncode)
                return process.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.warning("Timeout or file-not-found while probing %s: %s", config.command, exc)
                return False
                
        except Exception as e:
            logger.debug(f"Server availability test failed: {e}")
            return False

    async def call_tool(self, server: str, tool: str, arguments: Dict[str, Any]) -> Any:
        """Call tool with enhanced error handling and monitoring."""
        if server not in self.adapters:
            raise ValueError(f"Server {server} not initialized")
        
        if not self.connection_status.get(server, False):
            raise RuntimeError(f"Server {server} not connected")
        
        adapter = self.adapters[server]

        # If adapter is an EnhancedMCPAdapter use its monitoring method
        if hasattr(adapter, "_call_mcp_tool_with_monitoring"):
            return adapter._call_mcp_tool_with_monitoring(tool, arguments)

        # Fallback for MockMCPClient or other adapters that expose 'call_tool'
        if hasattr(adapter, "call_tool"):
            return await adapter.call_tool(tool, arguments)

        raise RuntimeError("Adapter does not support tool calls")

    @contextmanager
    def server_connection(self, server_name: str):
        """Context manager for individual server connections."""
        if server_name not in self.server_configs:
            raise ValueError(f"Server {server_name} not configured")
        
        config = self.server_configs[server_name]
        adapter = EnhancedMCPAdapter(config, use_schema_fixing=self.use_schema_fixing)
        
        try:
            adapter.start()
            yield adapter
        finally:
            adapter.stop()

    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive health check with performance metrics."""
        health_report = {}
        
        for server_name, adapter in self.adapters.items():
            try:
                is_healthy = self.connection_status.get(server_name, False)
                metrics = adapter.get_performance_metrics() if self.enable_monitoring else {}
                
                health_report[server_name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'connected': is_healthy,
                    'metrics': metrics
                }
                
            except Exception as e:
                health_report[server_name] = {
                    'status': 'error',
                    'connected': False,
                    'error': str(e),
                    'metrics': {}
                }
        
        return health_report

    async def shutdown(self):
        """Shutdown all servers with comprehensive cleanup."""
        logger.info("Shutting down Enhanced MCP Manager...")
        
        for server_name, adapter in self.adapters.items():
            try:
                adapter.stop()
                logger.info(f"Stopped {server_name}")
            except Exception as e:
                logger.warning(f"Error stopping {server_name}: {e}")
        
        self.adapters.clear()
        self.connection_status.clear()
        self.performance_metrics.clear()
        self._initialized = False
        
        logger.info("Enhanced MCP Manager shutdown complete")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.enable_monitoring:
            return {"monitoring": "disabled"}
        
        return {
            'total_servers': len(self.adapters),
            'connected_servers': sum(self.connection_status.values()),
            'server_metrics': self.performance_metrics,
            'generated_at': datetime.now().isoformat()
        }


# ============================================================================
# COMPATIBILITY AND FACTORY FUNCTIONS
# ============================================================================

# Global instance for backward compatibility
_enhanced_manager: Optional[EnhancedMCPManager] = None

def get_enhanced_mcp_manager() -> EnhancedMCPManager:
    """Get enhanced MCP manager instance."""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedMCPManager()
    return _enhanced_manager

def enable_enhanced_mcp_manager():
    """Enable enhanced MCP manager for the session."""
    global _enhanced_manager
    _enhanced_manager = EnhancedMCPManager()
    logger.info("Enhanced MCP Manager enabled")

# For backward compatibility, update the standard get_mcp_manager
def get_mcp_manager_with_enhancement(use_enhanced: bool = False):
    """Get MCP manager with optional enhancement."""
    if use_enhanced or os.getenv("USE_ENHANCED_MCP", "false").lower() == "true":
        return get_enhanced_mcp_manager()
    else:
        # Import and return standard manager
        from .mcp_manager import get_mcp_manager
        return get_mcp_manager() 