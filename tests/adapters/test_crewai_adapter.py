import ast
import re

from textwrap import dedent

import pytest
from mcp import StdioServerParameters

from src.server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter


def extract_and_eval_dict(text):
    # Match the first outermost curly brace block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No dictionary-like structure found in the string.")

    dict_str = match.group(0)

    try:
        # Safer than eval for parsing literals
        parsed_dict = ast.literal_eval(dict_str)
        return parsed_dict
    except Exception as e:
        raise ValueError(f"Failed to evaluate dictionary: {e}")


@pytest.fixture
def echo_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


@pytest.fixture
def custom_script_with_custom_arguments():
    return dedent(
        """
        from mcp.server.fastmcp import FastMCP
        from typing import Literal
        from enum import Enum
        from pydantic import BaseModel

        class Animal(BaseModel):
            legs: int
            name: str

        mcp = FastMCP("Server")

        @mcp.tool()
        def custom_tool(
            text: Literal["ciao", "hello"],
            animal: Animal,
            env: str | None = None,

        ) -> str:
            pass

        mcp.run()
        """
    )


@pytest.fixture
def custom_script_with_custom_list():
    return dedent(
        """
        from mcp.server.fastmcp import FastMCP
        from pydantic import BaseModel

        class Point(BaseModel):
            x: float
            y: float

        mcp = FastMCP("Server")

        @mcp.tool()
        def custom_tool(
            points: list[Point],

        ) -> str:
            pass

        mcp.run()
        """
    )


@pytest.fixture
def echo_server_sse_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run("sse")
        '''
    )


@pytest.fixture
def echo_server_optional_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool_optional(text: str | None = None) -> str:
            """Echo the input text, or return a default message if no text is provided"""
            if text is None:
                return "No input provided"
            return f"Echo: {text}"

        @mcp.tool()
        def echo_tool_default_value(text: str = "empty") -> str:
            """Echo the input text, default to 'empty' if no text is provided"""
            return f"Echo: {text}"

        @mcp.tool()
        def echo_tool_union_none(text: str | None) -> str:
            """Echo the input text, but None is not specified by default."""
            if text is None:
                return "No input provided"
            return f"Echo: {text}"
        
        mcp.run()
        '''
    )


# Positional Arguments Test Fixtures
@pytest.fixture
def single_arg_server_script():
    return dedent('''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Single Arg Server")

        @mcp.tool()
        def simple_tool(text: str) -> str:
            """Simple tool with one required argument"""
            return f"Echo: {text}"
        
        mcp.run()
    ''')


@pytest.fixture
def multi_arg_server_script():
    return dedent('''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Multi Arg Server")

        @mcp.tool()
        def multi_tool(name: str, age: int, city: str = "Unknown") -> str:
            """Tool with required and optional arguments"""
            return f"Name: {name}, Age: {age}, City: {city}"
        
        mcp.run()
    ''')


@pytest.fixture
def optional_only_server_script():
    return dedent('''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Optional Server")

        @mcp.tool()
        def optional_tool(message: str = "default") -> str:
            """Tool with only optional argument"""
            return f"Message: {message}"
        
        mcp.run()
    ''')


@pytest.fixture
def mcp_server_that_rejects_none_script():
    return dedent(
        """
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
        import anyio

        app = Server("mcp-strict-server")

        @app.call_tool()
        async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            if name != "strict_tool":
                raise ValueError(f"Unknown tool: {name}")
            
            # Simulate GitHub MCP server behavior - reject None values
            if arguments:
                for key, value in arguments.items():
                    if value is None:
                        # MCP servers that reject None values raise an error
                        raise RuntimeError(f"parameter {key} is not of type string, is <nil>")
            
            required = arguments.get("required") if arguments else None
            optional = arguments.get("optional") if arguments else None  
            another_optional = arguments.get("another_optional") if arguments else None
            
            return [types.TextContent(
                type="text",
                text=f"Required: {required}, Optional: {optional}, Another: {another_optional}"
            )]

        @app.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="strict_tool",
                    description="A tool that expects only string values",
                    inputSchema={
                        "type": "object",
                        "required": ["required"],
                        "properties": {
                            "required": {
                                "type": "string",
                                "description": "Required parameter",
                            },
                            "optional": {
                                "type": "string", 
                                "description": "Optional parameter",
                            },
                            "another_optional": {
                                "type": "string",
                                "description": "Another optional parameter",
                            }
                        },
                    },
                )
            ]

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)
        """
    )


@pytest.fixture
async def echo_sse_server(echo_server_sse_script):
    import subprocess
    import time

    # Start the SSE server process with its own process group
    process = subprocess.Popen(
        ["python", "-c", echo_server_sse_script],
    )

    # Give the server a moment to start up
    time.sleep(1)

    try:
        yield {"url": "http://127.0.0.1:8000/sse"}
    finally:
        # Clean up the process when test is done
        process.kill()
        process.wait()


def test_basic_sync(echo_server_script):
    with MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_script]
        ),
        CrewAIAdapter(),
    ) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].run(text="hello") == "Echo: hello"


# Fails if enums, unions, or pydantic classes are not included in the
# generated schema
def test_basic_sync_custom_arguments(custom_script_with_custom_arguments):
    with MCPAdapt(
        StdioServerParameters(
            command="uv",
            args=["run", "python", "-c", custom_script_with_custom_arguments],
        ),
        CrewAIAdapter(),
    ) as tools:
        tools_dict = extract_and_eval_dict(tools[0].description)
        assert tools_dict != {}
        assert tools_dict["properties"] != {}
        # Enum tests
        assert "enum" in tools_dict["properties"]["text"]
        assert "hello" in tools_dict["properties"]["text"]["enum"]
        assert "ciao" in tools_dict["properties"]["text"]["enum"]
        # Pydantic class tests
        assert tools_dict["properties"]["animal"]["properties"] != {}
        assert tools_dict["properties"]["animal"]["properties"]["legs"] != {}
        assert tools_dict["properties"]["animal"]["properties"]["name"] != {}
        # Union tests
        assert "anyOf" in tools_dict["properties"]["env"]
        assert tools_dict["properties"]["env"]["anyOf"] != []
        types = [opt.get("type") for opt in tools_dict["properties"]["env"]["anyOf"]]
        assert "null" in types
        assert "string" in types


# Raises KeyError
# if the pydantic objects list is not correctly resolved with $ref handling
# within mcp_tool.inputSchema
def test_basic_sync_custom_list(custom_script_with_custom_list):
    with MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", custom_script_with_custom_list]
        ),
        CrewAIAdapter(),
    ) as tools:
        tools_dict = extract_and_eval_dict(tools[0].description)
        assert tools_dict != {}
        assert tools_dict["properties"] != {}
        # Pydantic class tests
        assert tools_dict["properties"]["points"]["items"] != {}





def test_optional_sync(echo_server_optional_script):
    with MCPAdapt(
        StdioServerParameters(
            command="uv", args=["run", "python", "-c", echo_server_optional_script]
        ),
        CrewAIAdapter(),
    ) as tools:
        assert len(tools) == 3
        assert tools[0].name == "echo_tool_optional"
        assert tools[0].run(text="hello") == "Echo: hello"
        assert tools[0].run() == "No input provided"
        assert tools[1].name == "echo_tool_default_value"
        assert tools[1].run(text="hello") == "Echo: hello"
        assert tools[1].run() == "Echo: empty"
        assert tools[2].name == "echo_tool_union_none"
        assert tools[2].run(text="hello") == "Echo: hello"


def test_none_values_filtered_from_kwargs(mcp_server_that_rejects_none_script):
    """Test that None values are filtered out before being sent to MCP tool.

    This test reproduces issue #46 where None values in kwargs cause MCP servers
    to reject the request with 'parameter is not of type string, is <nil>' error.
    """
    with MCPAdapt(
        StdioServerParameters(
            command="uv",
            args=["run", "python", "-c", mcp_server_that_rejects_none_script],
        ),
        CrewAIAdapter(),
    ) as tools:
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "strict_tool"

        # This should work - only required parameter
        result = tool.run(required="test")
        assert "Required: test" in result

        # This should work - explicit non-None values
        result = tool.run(required="test", optional="value1", another_optional="value2")
        assert "Required: test" in result
        assert "Optional: value1" in result
        assert "Another: value2" in result

        # After the fix - CrewAI passes None values but they are filtered out
        # before being sent to the MCP server if the schema doesn't allow null
        result = tool.run(required="test", optional=None, another_optional=None)

        # The fix filters out None values, so the server receives only {"required": "test"}
        # and returns a successful response with None for the optional parameters
        assert "Required: test" in result
        assert "Optional: None" in result
        assert "Another: None" in result
        # Most importantly, no error message about "is not of type string, is <nil>"
        assert "parameter" not in result or "is <nil>" not in result


# Positional Arguments Tests
def test_single_positional_argument(single_arg_server_script):
    """Test that single positional argument works correctly."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", single_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run("test message")
        assert result == "Echo: test message"


def test_positional_backward_compatibility(single_arg_server_script):
    """Test that existing keyword argument usage still works."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", single_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run(text="keyword test")
        assert result == "Echo: keyword test"


def test_multiple_positional_arguments(multi_arg_server_script):
    """Test multiple positional arguments with required parameters."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", multi_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run("Alice", 25)
        assert "Name: Alice" in result
        assert "Age: 25" in result
        assert "City: Unknown" in result  # Default value


def test_mixed_positional_and_keyword_arguments(multi_arg_server_script):
    """Test mixing positional and keyword arguments."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", multi_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run("Bob", 30, city="New York")
        assert "Name: Bob" in result
        assert "Age: 30" in result
        assert "City: New York" in result


def test_keyword_override_positional(multi_arg_server_script):
    """Test that keyword arguments override positional arguments."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", multi_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        # Pass name positionally but override with keyword
        result = tool.run("Charlie", age=35, name="David")
        assert "Name: David" in result  # Keyword should override positional
        assert "Age: 35" in result


def test_optional_argument_with_positional(optional_only_server_script):
    """Test positional argument with optional parameter."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", optional_only_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run("custom message")
        assert result == "Message: custom message"


def test_optional_argument_uses_default(optional_only_server_script):
    """Test that optional arguments work with no parameters."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", optional_only_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        result = tool.run()
        assert result == "Message: default"


def test_excess_positional_arguments_ignored(single_arg_server_script):
    """Test that excess positional arguments are ignored."""
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", single_arg_server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        # Pass extra arguments - should only use the first one
        result = tool.run("first", "second", "third")
        assert result == "Echo: first"


def test_positional_args_ordering():
    """Test that positional arguments are mapped in correct order (required first)."""
    server_script = dedent('''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Order Server")

        @mcp.tool()
        def order_tool(required1: str, optional1: str = "opt1", required2: int = ..., optional2: str = "opt2") -> str:
            """Tool to test argument ordering"""
            return f"R1:{required1}, O1:{optional1}, R2:{required2}, O2:{optional2}"
        
        mcp.run()
    ''')
    
    with MCPAdapt(
        StdioServerParameters(command="uv", args=["run", "python", "-c", server_script]),
        CrewAIAdapter(),
    ) as tools:
        tool = tools[0]
        # Required args should come first in positional mapping
        result = tool.run("req1_value", 42)
        assert "R1:req1_value" in result
        assert "R2:42" in result
        assert "O1:opt1" in result  # Default values
        assert "O2:opt2" in result


def test_args_schema_propagation():
    """Test that every wrapped tool has non-None args_schema (Step 6a)."""
    from src.server_research_mcp.tools.mcp_tools import _AdaptHolder
    
    # Get all wrapped tools
    tools = _AdaptHolder.get_all_tools()
    
    # Verify every tool has args_schema
    for tool in tools:
        assert hasattr(tool, 'args_schema'), f"Tool {tool.name} missing args_schema attribute"
        assert tool.args_schema is not None, f"Tool {tool.name} has None args_schema"
        
        # Verify args_schema is a valid Pydantic model class
        from pydantic import BaseModel
        assert issubclass(tool.args_schema, BaseModel), f"Tool {tool.name} args_schema is not a BaseModel subclass"
        
    print(f"✅ Schema propagation test passed: All {len(tools)} tools have valid args_schema")
