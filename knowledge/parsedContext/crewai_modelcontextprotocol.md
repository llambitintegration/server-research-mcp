# MCP Servers as Tools in CrewAI

> Learn how to integrate MCP servers as tools in your CrewAI agents using the `crewai-tools` library.

## Overview

The [Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) provides a standardized way for AI agents to provide context to LLMs by communicating with external services, known as MCP Servers.
The `crewai-tools` library extends CrewAI's capabilities by allowing you to seamlessly integrate tools from these MCP servers into your agents.
This gives your crews access to a vast ecosystem of functionalities.

We currently support the following transport mechanisms:

* **Stdio**: for local servers (communication via standard input/output between processes on the same machine)
* **Server-Sent Events (SSE)**: for remote servers (unidirectional, real-time data streaming from server to client over HTTP)
* **Streamable HTTP**: for remote servers (flexible, potentially bi-directional communication over HTTP, often utilizing SSE for server-to-client streams)

## Video Tutorial

Watch this video tutorial for a comprehensive guide on MCP integration with CrewAI:

<iframe width="100%" height="400" src="https://www.youtube.com/embed/TpQ45lAZh48" title="CrewAI MCP Integration Guide" frameborder="0" style={{ borderRadius: '10px' }} allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen />

## Installation

Before you start using MCP with `crewai-tools`, you need to install the `mcp` extra `crewai-tools` dependency with the following command:

```shell
uv pip install 'crewai-tools[mcp]'
```

## Key Concepts & Getting Started

The `MCPServerAdapter` class from `crewai-tools` is the primary way to connect to an MCP server and make its tools available to your CrewAI agents. It supports different transport mechanisms and simplifies connection management.

Using a Python context manager (`with` statement) is the **recommended approach** for `MCPServerAdapter`. It automatically handles starting and stopping the connection to the MCP server.

```python
from crewai import Agent
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters # For Stdio Server

# Example server_params (choose one based on your server type):
# 1. Stdio Server:
server_params=StdioServerParameters(
    command="python3", 
    args=["servers/your_server.py"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

# 2. SSE Server:
server_params = {
    "url": "http://localhost:8000/sse", 
    "transport": "sse"
}

# 3. Streamable HTTP Server:
server_params = {
    "url": "http://localhost:8001/mcp", 
    "transport": "streamable-http"
}

# Example usage (uncomment and adapt once server_params is set):
with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")
    
    my_agent = Agent(
        role="MCP Tool User",
        goal="Utilize tools from an MCP server.",
        backstory="I can connect to MCP servers and use their tools.",
        tools=mcp_tools, # Pass the loaded tools to your agent
        reasoning=True,
        verbose=True
    )
    # ... rest of your crew setup ...
```

This general pattern shows how to integrate tools. For specific examples tailored to each transport, refer to the detailed guides below.

## Filtering Tools

```python
with MCPServerAdapter(server_params) as mcp_tools:
    print(f"Available tools: {[tool.name for tool in mcp_tools]}")

    my_agent = Agent(
        role="MCP Tool User",
        goal="Utilize tools from an MCP server.",
        backstory="I can connect to MCP servers and use their tools.",
        tools=mcp_tools["tool_name"], # Pass the loaded tools to your agent
        reasoning=True,
        verbose=True
    )
    # ... rest of your crew setup ...
```

## Explore MCP Integrations

<CardGroup cols={2}>
  <Card title="Stdio Transport" icon="server" href="/mcp/stdio" color="#3B82F6">
    Connect to local MCP servers via standard input/output. Ideal for scripts and local executables.
  </Card>

  <Card title="SSE Transport" icon="wifi" href="/mcp/sse" color="#10B981">
    Integrate with remote MCP servers using Server-Sent Events for real-time data streaming.
  </Card>

  <Card title="Streamable HTTP Transport" icon="globe" href="/mcp/streamable-http" color="#F59E0B">
    Utilize flexible Streamable HTTP for robust communication with remote MCP servers.
  </Card>

  <Card title="Connecting to Multiple Servers" icon="layer-group" href="/mcp/multiple-servers" color="#8B5CF6">
    Aggregate tools from several MCP servers simultaneously using a single adapter.
  </Card>

  <Card title="Security Considerations" icon="lock" href="/mcp/security" color="#EF4444">
    Review important security best practices for MCP integration to keep your agents safe.
  </Card>
</CardGroup>

Checkout this repository for full demos and examples of MCP integration with CrewAI! 👇

<Card title="GitHub Repository" icon="github" href="https://github.com/tonykipkemboi/crewai-mcp-demo" target="_blank">
  CrewAI MCP Demo
</Card>

## Staying Safe with MCP

<Warning>
  Always ensure that you trust an MCP Server before using it.
</Warning>

#### Security Warning: DNS Rebinding Attacks

SSE transports can be vulnerable to DNS rebinding attacks if not properly secured.
To prevent this:

1. **Always validate Origin headers** on incoming SSE connections to ensure they come from expected sources
2. **Avoid binding servers to all network interfaces** (0.0.0.0) when running locally - bind only to localhost (127.0.0.1) instead
3. **Implement proper authentication** for all SSE connections

Without these protections, attackers could use DNS rebinding to interact with local MCP servers from remote websites.

For more details, see the [Anthropic's MCP Transport Security docs](https://modelcontextprotocol.io/docs/concepts/transports#security-considerations).

### Limitations

* **Supported Primitives**: Currently, `MCPServerAdapter` primarily supports adapting MCP `tools`.
  Other MCP primitives like `prompts` or `resources` are not directly integrated as CrewAI components through this adapter at this time.
* **Output Handling**: The adapter typically processes the primary text output from an MCP tool (e.g., `.content[0].text`). Complex or multi-modal outputs might require custom handling if not fitting this pattern.

# Stdio Transport

> Learn how to connect CrewAI to local MCP servers using the Stdio (Standard Input/Output) transport mechanism.

## Overview

The Stdio (Standard Input/Output) transport is designed for connecting `MCPServerAdapter` to local MCP servers that communicate over their standard input and output streams. This is typically used when the MCP server is a script or executable running on the same machine as your CrewAI application.

## Key Concepts

* **Local Execution**: Stdio transport manages a locally running process for the MCP server.
* **`StdioServerParameters`**: This class from the `mcp` library is used to configure the command, arguments, and environment variables for launching the Stdio server.

## Connecting via Stdio

You can connect to an Stdio-based MCP server using two main approaches for managing the connection lifecycle:

### 1. Fully Managed Connection (Recommended)

Using a Python context manager (`with` statement) is the recommended approach. It automatically handles starting the MCP server process and stopping it when the context is exited.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

# Create a StdioServerParameters object
server_params=StdioServerParameters(
    command="python3", 
    args=["servers/your_stdio_server.py"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPServerAdapter(server_params) as tools:
    print(f"Available tools from Stdio MCP server: {[tool.name for tool in tools]}")

    # Example: Using the tools from the Stdio MCP server in a CrewAI Agent
    research_agent = Agent(
        role="Local Data Processor",
        goal="Process data using a local Stdio-based tool.",
        backstory="An AI that leverages local scripts via MCP for specialized tasks.",
        tools=tools,
        reasoning=True,
        verbose=True,
    )
    
    processing_task = Task(
        description="Process the input data file 'data.txt' and summarize its contents.",
        expected_output="A summary of the processed data.",
        agent=research_agent,
        markdown=True
    )
    
    data_crew = Crew(
        agents=[research_agent],
        tasks=[processing_task],
        verbose=True,
        process=Process.sequential 
    )
   
    result = data_crew.kickoff()
    print("\nCrew Task Result (Stdio - Managed):\n", result)

```

### 2. Manual Connection Lifecycle

If you need finer-grained control over when the Stdio MCP server process is started and stopped, you can manage the `MCPServerAdapter` lifecycle manually.

<Info>
  You **MUST** call `mcp_server_adapter.stop()` to ensure the server process is terminated and resources are released. Using a `try...finally` block is highly recommended.
</Info>

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

# Create a StdioServerParameters object
stdio_params=StdioServerParameters(
    command="python3", 
    args=["servers/your_stdio_server.py"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

mcp_server_adapter = MCPServerAdapter(server_params=stdio_params)
try:
    mcp_server_adapter.start()  # Manually start the connection and server process
    tools = mcp_server_adapter.tools
    print(f"Available tools (manual Stdio): {[tool.name for tool in tools]}")

    # Example: Using the tools with your Agent, Task, Crew setup
    manual_agent = Agent(
        role="Local Task Executor",
        goal="Execute a specific local task using a manually managed Stdio tool.",
        backstory="An AI proficient in controlling local processes via MCP.",
        tools=tools,
        verbose=True
    )
    
    manual_task = Task(
        description="Execute the 'perform_analysis' command via the Stdio tool.",
        expected_output="Results of the analysis.",
        agent=manual_agent
    )
    
    manual_crew = Crew(
        agents=[manual_agent],
        tasks=[manual_task],
        verbose=True,
        process=Process.sequential
    )
        
       
    result = manual_crew.kickoff() # Actual inputs depend on your tool
    print("\nCrew Task Result (Stdio - Manual):\n", result)
            
except Exception as e:
    print(f"An error occurred during manual Stdio MCP integration: {e}")
finally:
    if mcp_server_adapter and mcp_server_adapter.is_connected: # Check if connected before stopping
        print("Stopping Stdio MCP server connection (manual)...")
        mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
    elif mcp_server_adapter: # If adapter exists but not connected (e.g. start failed)
        print("Stdio MCP server adapter was not connected. No stop needed or start failed.")

```

Remember to replace placeholder paths and commands with your actual Stdio server details. The `env` parameter in `StdioServerParameters` can
be used to set environment variables for the server process, which can be useful for configuring its behavior or providing necessary paths (like `PYTHONPATH`).

# SSE Transport

> Learn how to connect CrewAI to remote MCP servers using Server-Sent Events (SSE) for real-time communication.

## Overview

Server-Sent Events (SSE) provide a standard way for a web server to send updates to a client over a single, long-lived HTTP connection. In the context of MCP, SSE is used for remote servers to stream data (like tool responses) to your CrewAI application in real-time.

## Key Concepts

* **Remote Servers**: SSE is suitable for MCP servers hosted remotely.
* **Unidirectional Stream**: Typically, SSE is a one-way communication channel from server to client.
* **`MCPServerAdapter` Configuration**: For SSE, you'll provide the server's URL and specify the transport type.

## Connecting via SSE

You can connect to an SSE-based MCP server using two main approaches for managing the connection lifecycle:

### 1. Fully Managed Connection (Recommended)

Using a Python context manager (`with` statement) is the recommended approach. It automatically handles establishing and closing the connection to the SSE MCP server.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8000/sse", # Replace with your actual SSE server URL
    "transport": "sse" 
}

# Using MCPServerAdapter with a context manager
try:
    with MCPServerAdapter(server_params) as tools:
        print(f"Available tools from SSE MCP server: {[tool.name for tool in tools]}")

        # Example: Using a tool from the SSE MCP server
        sse_agent = Agent(
            role="Remote Service User",
            goal="Utilize a tool provided by a remote SSE MCP server.",
            backstory="An AI agent that connects to external services via SSE.",
            tools=tools,
            reasoning=True,
            verbose=True,
        )

        sse_task = Task(
            description="Fetch real-time stock updates for 'AAPL' using an SSE tool.",
            expected_output="The latest stock price for AAPL.",
            agent=sse_agent,
            markdown=True
        )

        sse_crew = Crew(
            agents=[sse_agent],
            tasks=[sse_task],
            verbose=True,
            process=Process.sequential
        )
        
        if tools: # Only kickoff if tools were loaded
            result = sse_crew.kickoff() # Add inputs={'stock_symbol': 'AAPL'} if tool requires it
            print("\nCrew Task Result (SSE - Managed):\n", result)
        else:
            print("Skipping crew kickoff as tools were not loaded (check server connection).")

except Exception as e:
    print(f"Error connecting to or using SSE MCP server (Managed): {e}")
    print("Ensure the SSE MCP server is running and accessible at the specified URL.")

```

<Note>
  Replace `"http://localhost:8000/sse"` with the actual URL of your SSE MCP server.
</Note>

### 2. Manual Connection Lifecycle

If you need finer-grained control, you can manage the `MCPServerAdapter` connection lifecycle manually.

<Info>
  You **MUST** call `mcp_server_adapter.stop()` to ensure the connection is closed and resources are released. Using a `try...finally` block is highly recommended.
</Info>

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8000/sse", # Replace with your actual SSE server URL
    "transport": "sse"
}

mcp_server_adapter = None 
try:
    mcp_server_adapter = MCPServerAdapter(server_params)
    mcp_server_adapter.start()
    tools = mcp_server_adapter.tools
    print(f"Available tools (manual SSE): {[tool.name for tool in tools]}")

    manual_sse_agent = Agent(
        role="Remote Data Analyst",
        goal="Analyze data fetched from a remote SSE MCP server using manual connection management.",
        backstory="An AI skilled in handling SSE connections explicitly.",
        tools=tools,
        verbose=True
    )
    
    analysis_task = Task(
        description="Fetch and analyze the latest user activity trends from the SSE server.",
        expected_output="A summary report of user activity trends.",
        agent=manual_sse_agent
    )
    
    analysis_crew = Crew(
        agents=[manual_sse_agent],
        tasks=[analysis_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = analysis_crew.kickoff()
    print("\nCrew Task Result (SSE - Manual):\n", result)

except Exception as e:
    print(f"An error occurred during manual SSE MCP integration: {e}")
    print("Ensure the SSE MCP server is running and accessible.")
finally:
    if mcp_server_adapter and mcp_server_adapter.is_connected:
        print("Stopping SSE MCP server connection (manual)...")
        mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
    elif mcp_server_adapter:
        print("SSE MCP server adapter was not connected. No stop needed or start failed.")

```

## Security Considerations for SSE

<Warning>
  **DNS Rebinding Attacks**: SSE transports can be vulnerable to DNS rebinding attacks if the MCP server is not properly secured. This could allow malicious websites to interact with local or intranet-based MCP servers.
</Warning>

To mitigate this risk:

* MCP server implementations should **validate `Origin` headers** on incoming SSE connections.
* When running local SSE MCP servers for development, **bind only to `localhost` (`127.0.0.1`)** rather than all network interfaces (`0.0.0.0`).
* Implement **proper authentication** for all SSE connections if they expose sensitive tools or data.

For a comprehensive overview of security best practices, please refer to our [Security Considerations](./security.mdx) page and the official [MCP Transport Security documentation](https://modelcontextprotocol.io/docs/concepts/transports#security-considerations).

# Streamable HTTP Transport

> Learn how to connect CrewAI to remote MCP servers using the flexible Streamable HTTP transport.

## Overview

Streamable HTTP transport provides a flexible way to connect to remote MCP servers. It's often built upon HTTP and can support various communication patterns, including request-response and streaming, sometimes utilizing Server-Sent Events (SSE) for server-to-client streams within a broader HTTP interaction.

## Key Concepts

* **Remote Servers**: Designed for MCP servers hosted remotely.
* **Flexibility**: Can support more complex interaction patterns than plain SSE, potentially including bi-directional communication if the server implements it.
* **`MCPServerAdapter` Configuration**: You'll need to provide the server's base URL for MCP communication and specify `"streamable-http"` as the transport type.

## Connecting via Streamable HTTP

You have two primary methods for managing the connection lifecycle with a Streamable HTTP MCP server:

### 1. Fully Managed Connection (Recommended)

The recommended approach is to use a Python context manager (`with` statement), which handles the connection's setup and teardown automatically.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

try:
    with MCPServerAdapter(server_params) as tools:
        print(f"Available tools from Streamable HTTP MCP server: {[tool.name for tool in tools]}")

        http_agent = Agent(
            role="HTTP Service Integrator",
            goal="Utilize tools from a remote MCP server via Streamable HTTP.",
            backstory="An AI agent adept at interacting with complex web services.",
            tools=tools,
            verbose=True,
        )

        http_task = Task(
            description="Perform a complex data query using a tool from the Streamable HTTP server.",
            expected_output="The result of the complex data query.",
            agent=http_agent,
        )

        http_crew = Crew(
            agents=[http_agent],
            tasks=[http_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = http_crew.kickoff() 
        print("\nCrew Task Result (Streamable HTTP - Managed):\n", result)

except Exception as e:
    print(f"Error connecting to or using Streamable HTTP MCP server (Managed): {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible at the specified URL.")

```

**Note:** Replace `"http://localhost:8001/mcp"` with the actual URL of your Streamable HTTP MCP server.

### 2. Manual Connection Lifecycle

For scenarios requiring more explicit control, you can manage the `MCPServerAdapter` connection manually.

<Info>
  It is **critical** to call `mcp_server_adapter.stop()` when you are done to close the connection and free up resources. A `try...finally` block is the safest way to ensure this.
</Info>

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

mcp_server_adapter = None 
try:
    mcp_server_adapter = MCPServerAdapter(server_params)
    mcp_server_adapter.start()
    tools = mcp_server_adapter.tools
    print(f"Available tools (manual Streamable HTTP): {[tool.name for tool in tools]}")

    manual_http_agent = Agent(
        role="Advanced Web Service User",
        goal="Interact with an MCP server using manually managed Streamable HTTP connections.",
        backstory="An AI specialist in fine-tuning HTTP-based service integrations.",
        tools=tools,
        verbose=True
    )
    
    data_processing_task = Task(
        description="Submit data for processing and retrieve results via Streamable HTTP.",
        expected_output="Processed data or confirmation.",
        agent=manual_http_agent
    )
    
    data_crew = Crew(
        agents=[manual_http_agent],
        tasks=[data_processing_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = data_crew.kickoff()
    print("\nCrew Task Result (Streamable HTTP - Manual):\n", result)

except Exception as e:
    print(f"An error occurred during manual Streamable HTTP MCP integration: {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible.")
finally:
    if mcp_server_adapter and mcp_server_adapter.is_connected:
        print("Stopping Streamable HTTP MCP server connection (manual)...")
        mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
    elif mcp_server_adapter:
        print("Streamable HTTP MCP server adapter was not connected. No stop needed or start failed.")
```

## Security Considerations

When using Streamable HTTP transport, general web security best practices are paramount:

* **Use HTTPS**: Always prefer HTTPS (HTTP Secure) for your MCP server URLs to encrypt data in transit.
* **Authentication**: Implement robust authentication mechanisms if your MCP server exposes sensitive tools or data.
* **Input Validation**: Ensure your MCP server validates all incoming requests and parameters.

For a comprehensive guide on securing your MCP integrations, please refer to our [Security Considerations](./security.mdx) page and the official [MCP Transport Security documentation](https://modelcontextprotocol.io/docs/concepts/transports#security-considerations).

# Streamable HTTP Transport

> Learn how to connect CrewAI to remote MCP servers using the flexible Streamable HTTP transport.

## Overview

Streamable HTTP transport provides a flexible way to connect to remote MCP servers. It's often built upon HTTP and can support various communication patterns, including request-response and streaming, sometimes utilizing Server-Sent Events (SSE) for server-to-client streams within a broader HTTP interaction.

## Key Concepts

* **Remote Servers**: Designed for MCP servers hosted remotely.
* **Flexibility**: Can support more complex interaction patterns than plain SSE, potentially including bi-directional communication if the server implements it.
* **`MCPServerAdapter` Configuration**: You'll need to provide the server's base URL for MCP communication and specify `"streamable-http"` as the transport type.

## Connecting via Streamable HTTP

You have two primary methods for managing the connection lifecycle with a Streamable HTTP MCP server:

### 1. Fully Managed Connection (Recommended)

The recommended approach is to use a Python context manager (`with` statement), which handles the connection's setup and teardown automatically.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

try:
    with MCPServerAdapter(server_params) as tools:
        print(f"Available tools from Streamable HTTP MCP server: {[tool.name for tool in tools]}")

        http_agent = Agent(
            role="HTTP Service Integrator",
            goal="Utilize tools from a remote MCP server via Streamable HTTP.",
            backstory="An AI agent adept at interacting with complex web services.",
            tools=tools,
            verbose=True,
        )

        http_task = Task(
            description="Perform a complex data query using a tool from the Streamable HTTP server.",
            expected_output="The result of the complex data query.",
            agent=http_agent,
        )

        http_crew = Crew(
            agents=[http_agent],
            tasks=[http_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = http_crew.kickoff() 
        print("\nCrew Task Result (Streamable HTTP - Managed):\n", result)

except Exception as e:
    print(f"Error connecting to or using Streamable HTTP MCP server (Managed): {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible at the specified URL.")

```

**Note:** Replace `"http://localhost:8001/mcp"` with the actual URL of your Streamable HTTP MCP server.

### 2. Manual Connection Lifecycle

For scenarios requiring more explicit control, you can manage the `MCPServerAdapter` connection manually.

<Info>
  It is **critical** to call `mcp_server_adapter.stop()` when you are done to close the connection and free up resources. A `try...finally` block is the safest way to ensure this.
</Info>

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

mcp_server_adapter = None 
try:
    mcp_server_adapter = MCPServerAdapter(server_params)
    mcp_server_adapter.start()
    tools = mcp_server_adapter.tools
    print(f"Available tools (manual Streamable HTTP): {[tool.name for tool in tools]}")

    manual_http_agent = Agent(
        role="Advanced Web Service User",
        goal="Interact with an MCP server using manually managed Streamable HTTP connections.",
        backstory="An AI specialist in fine-tuning HTTP-based service integrations.",
        tools=tools,
        verbose=True
    )
    
    data_processing_task = Task(
        description="Submit data for processing and retrieve results via Streamable HTTP.",
        expected_output="Processed data or confirmation.",
        agent=manual_http_agent
    )
    
    data_crew = Crew(
        agents=[manual_http_agent],
        tasks=[data_processing_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = data_crew.kickoff()
    print("\nCrew Task Result (Streamable HTTP - Manual):\n", result)

except Exception as e:
    print(f"An error occurred during manual Streamable HTTP MCP integration: {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible.")
finally:
    if mcp_server_adapter and mcp_server_adapter.is_connected:
        print("Stopping Streamable HTTP MCP server connection (manual)...")
        mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
    elif mcp_server_adapter:
        print("Streamable HTTP MCP server adapter was not connected. No stop needed or start failed.")
```

## Security Considerations

When using Streamable HTTP transport, general web security best practices are paramount:

* **Use HTTPS**: Always prefer HTTPS (HTTP Secure) for your MCP server URLs to encrypt data in transit.
* **Authentication**: Implement robust authentication mechanisms if your MCP server exposes sensitive tools or data.
* **Input Validation**: Ensure your MCP server validates all incoming requests and parameters.

For a comprehensive guide on securing your MCP integrations, please refer to our [Security Considerations](./security.mdx) page and the official [MCP Transport Security documentation](https://modelcontextprotocol.io/docs/concepts/transports#security-considerations).

# Streamable HTTP Transport

> Learn how to connect CrewAI to remote MCP servers using the flexible Streamable HTTP transport.

## Overview

Streamable HTTP transport provides a flexible way to connect to remote MCP servers. It's often built upon HTTP and can support various communication patterns, including request-response and streaming, sometimes utilizing Server-Sent Events (SSE) for server-to-client streams within a broader HTTP interaction.

## Key Concepts

* **Remote Servers**: Designed for MCP servers hosted remotely.
* **Flexibility**: Can support more complex interaction patterns than plain SSE, potentially including bi-directional communication if the server implements it.
* **`MCPServerAdapter` Configuration**: You'll need to provide the server's base URL for MCP communication and specify `"streamable-http"` as the transport type.

## Connecting via Streamable HTTP

You have two primary methods for managing the connection lifecycle with a Streamable HTTP MCP server:

### 1. Fully Managed Connection (Recommended)

The recommended approach is to use a Python context manager (`with` statement), which handles the connection's setup and teardown automatically.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

try:
    with MCPServerAdapter(server_params) as tools:
        print(f"Available tools from Streamable HTTP MCP server: {[tool.name for tool in tools]}")

        http_agent = Agent(
            role="HTTP Service Integrator",
            goal="Utilize tools from a remote MCP server via Streamable HTTP.",
            backstory="An AI agent adept at interacting with complex web services.",
            tools=tools,
            verbose=True,
        )

        http_task = Task(
            description="Perform a complex data query using a tool from the Streamable HTTP server.",
            expected_output="The result of the complex data query.",
            agent=http_agent,
        )

        http_crew = Crew(
            agents=[http_agent],
            tasks=[http_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = http_crew.kickoff() 
        print("\nCrew Task Result (Streamable HTTP - Managed):\n", result)

except Exception as e:
    print(f"Error connecting to or using Streamable HTTP MCP server (Managed): {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible at the specified URL.")

```

**Note:** Replace `"http://localhost:8001/mcp"` with the actual URL of your Streamable HTTP MCP server.

### 2. Manual Connection Lifecycle

For scenarios requiring more explicit control, you can manage the `MCPServerAdapter` connection manually.

<Info>
  It is **critical** to call `mcp_server_adapter.stop()` when you are done to close the connection and free up resources. A `try...finally` block is the safest way to ensure this.
</Info>

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter

server_params = {
    "url": "http://localhost:8001/mcp", # Replace with your actual Streamable HTTP server URL
    "transport": "streamable-http"
}

mcp_server_adapter = None 
try:
    mcp_server_adapter = MCPServerAdapter(server_params)
    mcp_server_adapter.start()
    tools = mcp_server_adapter.tools
    print(f"Available tools (manual Streamable HTTP): {[tool.name for tool in tools]}")

    manual_http_agent = Agent(
        role="Advanced Web Service User",
        goal="Interact with an MCP server using manually managed Streamable HTTP connections.",
        backstory="An AI specialist in fine-tuning HTTP-based service integrations.",
        tools=tools,
        verbose=True
    )
    
    data_processing_task = Task(
        description="Submit data for processing and retrieve results via Streamable HTTP.",
        expected_output="Processed data or confirmation.",
        agent=manual_http_agent
    )
    
    data_crew = Crew(
        agents=[manual_http_agent],
        tasks=[data_processing_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = data_crew.kickoff()
    print("\nCrew Task Result (Streamable HTTP - Manual):\n", result)

except Exception as e:
    print(f"An error occurred during manual Streamable HTTP MCP integration: {e}")
    print("Ensure the Streamable HTTP MCP server is running and accessible.")
finally:
    if mcp_server_adapter and mcp_server_adapter.is_connected:
        print("Stopping Streamable HTTP MCP server connection (manual)...")
        mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
    elif mcp_server_adapter:
        print("Streamable HTTP MCP server adapter was not connected. No stop needed or start failed.")
```

## Security Considerations

When using Streamable HTTP transport, general web security best practices are paramount:

* **Use HTTPS**: Always prefer HTTPS (HTTP Secure) for your MCP server URLs to encrypt data in transit.
* **Authentication**: Implement robust authentication mechanisms if your MCP server exposes sensitive tools or data.
* **Input Validation**: Ensure your MCP server validates all incoming requests and parameters.

For a comprehensive guide on securing your MCP integrations, please refer to our [Security Considerations](./security.mdx) page and the official [MCP Transport Security documentation](https://modelcontextprotocol.io/docs/concepts/transports#security-considerations).

