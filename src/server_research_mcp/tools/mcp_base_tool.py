"""
Base class for MCP tools that eliminates boilerplate code.
"""
from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import Any, Dict, Type, Optional
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# Module-level import for test patching
def get_mcp_manager():
    """Return the MCPServerAdapterManager instance using official CrewAI patterns.
    
    This replaces the previous mock/enhanced manager selection with the official
    MCPServerAdapter-based implementation from crewai-tools.
    """
    logger.info("🔧 Using MCPServerAdapterManager with official CrewAI patterns")
    
    # Use the new MCPServerAdapterManager
    from .mcp_adapter_manager import get_mcp_adapter_manager
    return get_mcp_adapter_manager()


class MCPBaseTool(BaseTool):
    """
    Base class for all MCP tools that handles common functionality:
    - MCP manager initialization
    - Async to sync bridge for CrewAI
    - Error handling and logging
    - Result serialization
    
    Subclasses only need to define:
    - server_name: str
    - mcp_tool_name: str  
    - args_schema: Type[BaseModel]
    - name: str (optional)
    - description: str (optional)
    """
    
    # Subclasses must override these attributes
    server_name: str               # e.g. "zotero", "memory", "context7"
    mcp_tool_name: str             # e.g. "search", "create_entity", "get_docs"
    args_schema: Type[BaseModel]   # Pydantic model for tool arguments
    
    # Optional attributes that can be overridden
    # Remove these since they're already defined in BaseTool with proper annotations
    
    def _run(self, **kwargs) -> str:
        """
        Synchronous entry point required by CrewAI.
        Bridges to async MCP operations.
        """
        logger.info(f"🔧 Starting MCP tool {self.server_name}.{self.mcp_tool_name}")
        logger.debug(f"📊 Tool arguments: {kwargs}")
        
        try:
            result = asyncio.run(self._async_run(**kwargs))
            logger.info(f"✅ MCP tool {self.server_name}.{self.mcp_tool_name} completed successfully")
            logger.debug(f"📋 Tool result: {result}")
            
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.error(
                f"❌ MCP tool {self.server_name}.{self.mcp_tool_name} failed: {exc}",
                exc_info=True
            )
            error_result = {
                "error": str(exc),
                "tool": f"{self.server_name}.{self.mcp_tool_name}",
                "status": "failed"
            }
            return json.dumps(error_result)
    
    async def _async_run(self, **kwargs) -> Dict[str, Any]:
        """
        Internal async method that performs the actual MCP call.
        """
        logger.debug(f"🔄 Starting async execution for {self.server_name}.{self.mcp_tool_name}")
        
        # Use module-level function for consistent patching
        manager = get_mcp_manager()
        logger.debug(f"📡 Got MCP manager: {type(manager)}")
        
        try:
            # Initialize the required server
            logger.info(f"🚀 Initializing MCP server: {self.server_name}")
            await manager.initialize([self.server_name])
            logger.info(f"✅ Server {self.server_name} initialized successfully")
            
            # Transform arguments if the input schema has a to_mcp_format method
            if hasattr(self.args_schema, 'to_mcp_format'):
                # Create an instance of the schema to access the method
                try:
                    schema_instance = self.args_schema(**kwargs)
                    if hasattr(schema_instance, 'to_mcp_format'):
                        original_kwargs = kwargs.copy()
                        kwargs = schema_instance.to_mcp_format()
                        logger.debug(f"🔄 Transformed arguments: {original_kwargs} -> {kwargs}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to transform arguments using to_mcp_format: {e}")
            
            # Call the MCP tool with provided arguments
            logger.info(f"📞 Calling MCP tool {self.server_name}.{self.mcp_tool_name}")
            result = await manager.call_tool(
                server=self.server_name,
                tool=self.mcp_tool_name,
                arguments=kwargs
            )
            logger.info(f"✅ MCP tool call successful")
            logger.debug(f"📋 Raw MCP result: {result}")
            
            # Override the "tool" field with CrewAI tool name for test compatibility
            if isinstance(result, dict) and "tool" in result:
                result["tool"] = self.name or f"{self.server_name}_{self.mcp_tool_name}"
                logger.debug(f"🔧 Overrode tool name to: {result['tool']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ MCP server call failed: {e}")
            
            # With MCPServerAdapter, we rely on real servers - no mock fallback
            error_result = {
                "status": "error",
                "error": str(e),
                "server": self.server_name,
                "tool": self.mcp_tool_name,
                "message": "MCP server call failed. Ensure MCP servers are properly installed and configured."
            }
            
            return error_result
    

    
    def __init_subclass__(cls, **kwargs):
        """
        Validate that subclasses define required attributes.
        """
        super().__init_subclass__(**kwargs)
        
        # These checks will happen at class definition time
        required_attrs = ['server_name', 'mcp_tool_name', 'args_schema']
        for attr in required_attrs:
            if not hasattr(cls, attr) or getattr(cls, attr) is None:
                # Don't raise during class definition, but log warning
                logger.warning(
                    "MCPBaseTool subclass %s should define '%s' attribute",
                    cls.__name__, 
                    attr
                )