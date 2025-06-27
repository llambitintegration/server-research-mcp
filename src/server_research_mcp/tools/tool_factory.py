"""
Factory and decorator for creating MCP tools with zero boilerplate.
"""
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, create_model
from .mcp_base_tool import MCPBaseTool
import json


def mcp_tool(
    *,                       # Force keyword arguments
    tool_name: str,
    server: str,
    mcp_method: str,
    schema: Type[BaseModel],
    description: str = ""
):
    """
    Factory function to create MCP tool classes with minimal code.
    
    Can be used as a decorator:
    ```python
    @mcp_tool(
        tool_name="zotero_search",
        server="zotero", 
        mcp_method="search",
        schema=ZoteroSearchInput
    )
    class ZoteroSearchTool:
        pass
    ```
    
    Or as a direct factory:
    ```python
    ZoteroSearchTool = mcp_tool(
        tool_name="zotero_search",
        server="zotero",
        mcp_method="search", 
        schema=ZoteroSearchInput
    )(None)
    ```
    
    Args:
        tool_name: Name of the tool for CrewAI
        server: MCP server name (e.g., "zotero", "memory", "context7")
        mcp_method: Method name on the MCP server
        schema: Pydantic BaseModel for input validation
        description: Tool description (optional)
    """
    def decorator(cls: Optional[Type] = None):
        # Extract description from decorated class docstring if available
        final_description = description
        if cls and cls.__doc__ and not description:
            final_description = cls.__doc__.strip()
        elif not final_description:
            final_description = f"MCP tool: {server}.{mcp_method}"
        
        # Create custom __init__ method that properly sets Pydantic fields
        def __init__(self, **kwargs):
            # Set the name and description using proper Pydantic initialization
            kwargs.setdefault('name', tool_name)
            kwargs.setdefault('description', final_description)
            super(tool_class, self).__init__(**kwargs)
        
        # Create the tool class attributes without overriding Pydantic fields
        attrs = {
            '__annotations__': {
                'server_name': str,
                'mcp_tool_name': str,
                'args_schema': Type[BaseModel],
            },
            '__module__': cls.__module__ if cls else __name__,
            '__init__': __init__,
            'server_name': server,
            'mcp_tool_name': mcp_method,
            'args_schema': schema,
        }
        
        # Create new class inheriting from MCPBaseTool
        tool_class = type(
            f"{tool_name.title().replace('_', '')}Tool",
            (MCPBaseTool,),
            attrs
        )
        
        return tool_class
    
    return decorator


def json_schema_to_pydantic(
    json_schema: Dict[str, Any],
    class_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    Convert a JSON Schema to a Pydantic model.
    
    This is a simplified version that handles basic types.
    For production, consider using datamodel-code-generator.
    
    Args:
        json_schema: JSON Schema dictionary
        class_name: Name for the generated Pydantic class
        
    Returns:
        Pydantic model class
    """
    # Map JSON Schema types to Python types
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict[str, Any]
    }
    
    # Extract properties
    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))
    
    # Build field definitions
    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        python_type = type_mapping.get(prop_type, str)
        
        # Handle arrays with items
        if prop_type == "array" and "items" in prop_schema:
            item_type = prop_schema["items"].get("type", "string")
            python_type = List[type_mapping.get(item_type, str)]
        
        # Determine if field is required
        default = ... if prop_name in required else None
        
        # Get description
        field_description = prop_schema.get("description", "")
        
        # Create field with description
        if field_description:
            fields[prop_name] = (python_type, default)
        else:
            fields[prop_name] = (python_type, default)
    
    # Create the model
    return create_model(class_name, **fields)


def create_tools_from_mcp_schema(
    server_name: str,
    tool_definitions: List[Dict[str, Any]]
) -> List[MCPBaseTool]:
    """
    Automatically create tool instances from MCP server schema.
    
    Args:
        server_name: Name of the MCP server
        tool_definitions: List of tool definitions from MCP schema
        
    Returns:
        List of instantiated MCPBaseTool objects
    """
    tools = []
    
    for tool_def in tool_definitions:
        # Generate Pydantic model from input schema
        model_name = f"{tool_def['name'].title().replace('-', '').replace('_', '')}Input"
        input_model = json_schema_to_pydantic(
            tool_def.get("inputSchema", {"properties": {}}),
            class_name=model_name
        )
        
        # Create tool class using factory
        tool_class = mcp_tool(
            tool_name=tool_def["name"],
            server=server_name,
            mcp_method=tool_def["name"],
            schema=input_model,
            description=tool_def.get("description", "")
        )(None)
        
        # Instantiate and add to list
        tools.append(tool_class())
    
    return tools