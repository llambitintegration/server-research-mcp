"""This module implements the CrewAI adapter.

CrewAI tools support only sync functions for their tools.

Example Usage:
>>> with MCPAdapt(StdioServerParameters(command="uv", args=["run", "src/echo.py"]), CrewAIAdapter()) as tools:
>>>     print(tools)
"""

from typing import Any, Callable, Coroutine, Type

import jsonref  # type: ignore
import mcp
from crewai.tools import BaseTool  # type: ignore
from pydantic import BaseModel, Field, create_model

from .core import ToolAdapter
from .utils.modeling import (
    resolve_refs_and_remove_defs,
)

json_type_mapping: dict[str, Type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class PositionalArgsModel(BaseModel):
    """Base model that handles both positional and keyword arguments."""
    
    def __init__(self, *args, **kwargs):
        # Convert positional args to kwargs based on schema property order
        if args and hasattr(self.__class__, '_mcp_schema_properties'):
            schema_properties = getattr(self.__class__, '_mcp_schema_properties')
            required_props = getattr(self.__class__, '_mcp_required_props', [])
            
            # Create ordered list: required properties first, then optional
            all_prop_names = list(schema_properties.keys())
            ordered_props = required_props + [name for name in all_prop_names if name not in required_props]
            
            # Map positional args to property names
            for i, arg_value in enumerate(args):
                if i < len(ordered_props):
                    prop_name = ordered_props[i]
                    # Only use positional arg if not overridden by kwargs
                    if prop_name not in kwargs:
                        kwargs[prop_name] = arg_value
        
        super().__init__(**kwargs)


def create_positional_model_from_json_schema(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> Type[PositionalArgsModel]:
    """Create a Pydantic model that supports positional arguments from a JSON schema."""
    # Get properties and required fields from schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Build field definitions for create_model
    field_definitions = {}
    for prop_name, prop_info in properties.items():
        python_type = _get_python_type_from_property(prop_info)
        is_required = prop_name in required
        
        if is_required:
            field_definitions[prop_name] = (python_type, Field(description=prop_info.get("description", "")))
        else:
            field_definitions[prop_name] = (python_type, Field(default=None, description=prop_info.get("description", "")))
    
    # Create model class that inherits from PositionalArgsModel
    model_class = create_model(
        model_name,
        __base__=PositionalArgsModel,
        **field_definitions
    )
    
    # Attach schema info as class attributes for positional arg mapping
    model_class._mcp_schema_properties = properties
    model_class._mcp_required_props = required
    
    return model_class


def _get_python_type_from_property(prop_info: dict) -> Type:
    """
    Map a JSON-schema property to a Python type.
    Handles  ✦ type                        ─ "string" / "integer" / …
             ✦ union types in a list       ─ ["integer","null"]
             ✦ anyOf / oneOf constructs    ─ anyOf:[{type:int},{type:null}]
    Falls back to typing.Any if the combination is too complex.
    """
    # 1. simple  "type": "integer"
    if "type" in prop_info:
        t = prop_info["type"]
        if isinstance(t, list):
            # choose the first non-null entry
            t = next((x for x in t if x != "null"), "string")
        return json_type_mapping.get(t, Any)

    # 2. complex schema – look at anyOf / oneOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in prop_info and isinstance(prop_info[key], list):
            for option in prop_info[key]:
                if "type" in option and option["type"] != "null":
                    return json_type_mapping.get(option["type"], Any)

    # 3. default fallback
    return Any


class CrewAIAdapter(ToolAdapter):
    """Adapter for `crewai`.

    Note that `crewai` support only sync tools so we write adapt for sync tools only.

    Warning: if the mcp tool name is a python keyword, starts with digits or contains
    dashes, the tool name will be sanitized to become a valid python function name.

    """

    def adapt(
        self,
        func: Callable[[dict | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ) -> BaseTool:
        """Adapt a MCP tool to a CrewAI tool.

        Args:
            func: The function to adapt.
            mcp_tool: The MCP tool to adapt.

        Returns:
            A CrewAI tool.
        """
        mcp_tool.inputSchema = resolve_refs_and_remove_defs(mcp_tool.inputSchema)
        ToolInput = create_positional_model_from_json_schema(mcp_tool.inputSchema)

        class CrewAIMCPTool(BaseTool):
            name: str = mcp_tool.name
            description: str = mcp_tool.description or ""
            args_schema: Type[BaseModel] = ToolInput

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                """Execute the MCP tool with the provided arguments."""
                try:
                    # Convert positional args to kwargs using our PositionalArgsModel logic
                    if args and hasattr(ToolInput, '_mcp_schema_properties'):
                        schema_properties = getattr(ToolInput, '_mcp_schema_properties')
                        required_props = getattr(ToolInput, '_mcp_required_props', [])
                        
                        # Create ordered list: required properties first, then optional
                        all_prop_names = list(schema_properties.keys())
                        ordered_props = required_props + [name for name in all_prop_names if name not in required_props]
                        
                        # Map positional args to property names
                        for i, arg_value in enumerate(args):
                            if i < len(ordered_props):
                                prop_name = ordered_props[i]
                                # Only use positional arg if not overridden by kwargs
                                if prop_name not in kwargs:
                                    kwargs[prop_name] = arg_value
                    
                    schema_properties = mcp_tool.inputSchema.get("properties", {})
                    required_props = mcp_tool.inputSchema.get("required", [])
                    
                    # Filter out None values if the schema doesn't allow null
                    filtered_kwargs: dict[str, Any] = {}
                    for key, value in kwargs.items():
                        if value is None and key in schema_properties:
                            prop_schema = schema_properties[key]
                            # Check if the property allows null
                            if isinstance(prop_schema.get("type"), list):
                                if "null" in prop_schema["type"]:
                                    filtered_kwargs[key] = value
                            elif "anyOf" in prop_schema:
                                # Check if any option allows null
                                if any(
                                    opt.get("type") == "null"
                                    for opt in prop_schema["anyOf"]
                                ):
                                    filtered_kwargs[key] = value
                            # If neither case allows null, skip the None value
                        else:
                            filtered_kwargs[key] = value
                    
                    # CRITICAL FIX: Validate required parameters are present and not empty
                    # Before validation, check if we have JSON strings that might contain the required parameters
                    missing_required = []
                    
                    # First, try to parse any JSON strings in args or kwargs that might contain required parameters
                    potential_json_sources = []
                    
                    # Check positional args for JSON strings
                    if args:
                        for arg in args:
                            if isinstance(arg, str) and arg.strip().startswith('{'):
                                potential_json_sources.append(arg)
                    
                    # Check kwargs values for JSON strings
                    for key, value in filtered_kwargs.items():
                        if isinstance(value, str) and value.strip().startswith('{'):
                            potential_json_sources.append(value)
                    
                    # Try to parse JSON sources and merge them into filtered_kwargs
                    parsed_params = {}
                    for json_str in potential_json_sources:
                        try:
                            import json
                            parsed_data = json.loads(json_str)
                            if isinstance(parsed_data, dict):
                                parsed_params.update(parsed_data)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    # Merge parsed parameters (but don't override existing direct parameters)
                    for key, value in parsed_params.items():
                        if key not in filtered_kwargs:
                            filtered_kwargs[key] = value
                    
                    # Now validate required parameters
                    for required_prop in required_props:
                        if required_prop not in filtered_kwargs:
                            missing_required.append(required_prop)
                        elif filtered_kwargs[required_prop] is None:
                            missing_required.append(f"{required_prop} (is None)")
                        elif isinstance(filtered_kwargs[required_prop], str) and not filtered_kwargs[required_prop].strip():
                            missing_required.append(f"{required_prop} (is empty string)")
                    
                    if missing_required:
                        return f"Error: Missing required parameters for {mcp_tool.name}: {', '.join(missing_required)}. Schema requires: {required_props}"
                    
                    # Convert empty strings to meaningful defaults for known problematic parameters
                    if 'query' in filtered_kwargs and isinstance(filtered_kwargs['query'], str) and not filtered_kwargs['query'].strip():
                        filtered_kwargs['query'] = "search query"
                    
                    # Log the parameters being sent to help with debugging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Tool {mcp_tool.name} called with parameters: {filtered_kwargs}")
                    
                    result = func(filtered_kwargs)
                    
                    # Handle the response safely
                    if hasattr(result, 'content') and result.content:
                        if isinstance(result.content, list) and len(result.content) > 0:
                            if hasattr(result.content[0], 'text'):
                                return result.content[0].text
                            else:
                                return str(result.content[0])
                        else:
                            return str(result.content)
                    else:
                        return str(result)
                        
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle specific JavaScript errors from MCP servers
                    if "Cannot read properties of undefined" in error_msg and "toLowerCase" in error_msg:
                        return f"Parameter validation error for {mcp_tool.name}: A required string parameter was undefined. This usually means a 'query' or similar text field is missing or empty. Please ensure all required parameters are provided."
                    elif "toLowerCase" in error_msg:
                        return f"String parameter error for {mcp_tool.name}: A parameter expected to be a string was undefined or null. Check that all text parameters are properly set."
                    elif "Event loop is closed" in error_msg:
                        return f"MCP server connection error for {mcp_tool.name}: Event loop closed"
                    elif "Connection refused" in error_msg:
                        return f"MCP server unavailable for {mcp_tool.name}: Connection refused"
                    else:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Tool {mcp_tool.name} failed with args={args}, kwargs={kwargs}: {error_msg}")
                        return f"Tool {mcp_tool.name} execution failed: {error_msg}"

            def _generate_description(self):
                # Use the original MCP tool schema instead of the simplified Pydantic model schema
                # This preserves complex type information like enums and list items
                args_schema = {
                    k: v
                    for k, v in mcp_tool.inputSchema.items()
                    if k != "$defs"
                }
                self.description = f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nTool Description: {self.description}"

        return CrewAIMCPTool()

    async def async_adapt(
        self,
        afunc: Callable[[dict | None], Coroutine[Any, Any, mcp.types.CallToolResult]],
        mcp_tool: mcp.types.Tool,
    ) -> Any:
        raise NotImplementedError("async is not supported by the CrewAI framework.")


if __name__ == "__main__":
    from mcp import StdioServerParameters

    from .core import MCPAdapt

    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "src/echo.py"]),
        CrewAIAdapter(),
    ) as tools:
        print(tools)
        print(tools[0].run(text="hello"))
