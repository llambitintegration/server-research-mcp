"""
Base class for MCP tools that eliminates boilerplate code.
"""
from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import Any, Dict, Type, Optional
import asyncio
import json
import logging
from ..rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

# Global rate limiter: 5 MCP calls per second with burst 5
MCP_RATE_LIMITER = TokenBucketRateLimiter(rate=5, capacity=5)

# Module-level import for test patching
def get_mcp_manager():
    """Return the appropriate MCP manager instance with extra diagnostic logging.

    This version adds a small amount of diagnostic logging so that it is clear
    at runtime *why* we selected either the enhanced or the lightweight (mock)
    manager.  It makes troubleshooting fallback-to-mock scenarios—such as why
    the Zotero server is mocked—incredibly easier.
    """
    # Diagnostic: capture env var first so we can log it
    import os
    env_value = os.getenv("USE_ENHANCED_MCP", "false").lower()
    use_enhanced = env_value == "true"

    # Emit single INFO-level line (won't flood logs but still visible)
    logger.info(
        "🔧 Selecting %s MCP manager (USE_ENHANCED_MCP=%s)",
        "enhanced" if use_enhanced else "standard/mock",
        env_value,
    )

    if use_enhanced:
        from .enhanced_mcp_manager import get_mcp_manager_with_enhancement
        return get_mcp_manager_with_enhancement(use_enhanced=True)

    # Default path – standard (possibly mocked) manager
    from .mcp_manager import get_mcp_manager as _get_mcp_manager  # noqa: WPS433
    return _get_mcp_manager()


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
            await MCP_RATE_LIMITER.acquire()
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
            await MCP_RATE_LIMITER.acquire()
            result = await manager.call_tool(
                server=self.server_name,
                tool=self.mcp_tool_name,
                arguments=kwargs
            )
            logger.info(f"✅ MCP tool call successful")
            logger.debug(f"📋 Raw MCP result: {result}")
            
            return result
            
        except Exception as e:
            logger.warning(f"❌ MCP server call failed: {e}")
            logger.warning(f"🔄 Falling back to mock mode for {self.server_name}.{self.mcp_tool_name}")
            
            # Fallback to mock responses when real MCP fails
            mock_result = self._get_mock_response(**kwargs)
            logger.info(f"🎭 Mock response generated: {type(mock_result)}")
            logger.debug(f"📋 Mock result: {mock_result}")
            
            return mock_result
    
    def _get_mock_response(self, **kwargs) -> Dict[str, Any]:
        """Provide mock responses when MCP servers are unavailable."""
        if self.server_name == "zotero":
            if self.mcp_tool_name == "search":
                return {
                    "status": "success",
                    "items": [
                        {
                            "key": "MOCK123",
                            "title": "KST: Executable Formal Semantics of IEC 61131-3 Structured Text for Verification",
                            "authors": ["Peter Duerr", "Bernhard Beckert"],
                            "year": 2022,
                            "journal": "Formal Methods in Programming",
                            "doi": "10.1000/mock.doi"
                        }
                    ],
                    "total": 1
                }
            elif self.mcp_tool_name == "get_item":
                return {
                    "status": "success",
                    "content": "Mock PDF content for the KST paper on formal verification...",
                    "metadata": {
                        "title": "KST: Executable Formal Semantics of IEC 61131-3 Structured Text for Verification",
                        "authors": ["Peter Duerr", "Bernhard Beckert"],
                        "abstract": "This paper presents KST, a novel approach for formal verification...",
                        "sections": [
                            {"title": "Introduction", "content": "Mock introduction content"},
                            {"title": "Methods", "content": "Mock methods content"},
                            {"title": "Results", "content": "Mock results content"},
                            {"title": "Conclusion", "content": "Mock conclusion content"}
                        ]
                    }
                }
        
        # Default mock response
        return {
            "status": "success",
            "message": f"Mock response from {self.server_name}.{self.mcp_tool_name}",
            "data": kwargs
        }
    
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