# Agents

> Detailed guide on creating and managing agents within the CrewAI framework.

## Overview of an Agent

In the CrewAI framework, an `Agent` is an autonomous unit that can:

* Perform specific tasks
* Make decisions based on its role and goal
* Use tools to accomplish objectives
* Communicate and collaborate with other agents
* Maintain memory of interactions
* Delegate tasks when allowed

<Tip>
  Think of an agent as a specialized team member with specific skills, expertise, and responsibilities. For example, a `Researcher` agent might excel at gathering and analyzing information, while a `Writer` agent might be better at creating content.
</Tip>

<Note type="info" title="Enterprise Enhancement: Visual Agent Builder">
  CrewAI Enterprise includes a Visual Agent Builder that simplifies agent creation and configuration without writing code. Design your agents visually and test them in real-time.

  ![Visual Agent Builder Screenshot](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/enterprise/crew-studio-interface.png)

  The Visual Agent Builder enables:

  * Intuitive agent configuration with form-based interfaces
  * Real-time testing and validation
  * Template library with pre-configured agent types
  * Easy customization of agent attributes and behaviors
</Note>

## Agent Attributes

| Attribute                               | Parameter                | Type                                  | Description                                                                                              |
| :-------------------------------------- | :----------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------------------- |
| **Role**                                | `role`                   | `str`                                 | Defines the agent's function and expertise within the crew.                                              |
| **Goal**                                | `goal`                   | `str`                                 | The individual objective that guides the agent's decision-making.                                        |
| **Backstory**                           | `backstory`              | `str`                                 | Provides context and personality to the agent, enriching interactions.                                   |
| **LLM** *(optional)*                    | `llm`                    | `Union[str, LLM, Any]`                | Language model that powers the agent. Defaults to the model specified in `OPENAI_MODEL_NAME` or "gpt-4". |
| **Tools** *(optional)*                  | `tools`                  | `List[BaseTool]`                      | Capabilities or functions available to the agent. Defaults to an empty list.                             |
| **Function Calling LLM** *(optional)*   | `function_calling_llm`   | `Optional[Any]`                       | Language model for tool calling, overrides crew's LLM if specified.                                      |
| **Max Iterations** *(optional)*         | `max_iter`               | `int`                                 | Maximum iterations before the agent must provide its best answer. Default is 20.                         |
| **Max RPM** *(optional)*                | `max_rpm`                | `Optional[int]`                       | Maximum requests per minute to avoid rate limits.                                                        |
| **Max Execution Time** *(optional)*     | `max_execution_time`     | `Optional[int]`                       | Maximum time (in seconds) for task execution.                                                            |
| **Verbose** *(optional)*                | `verbose`                | `bool`                                | Enable detailed execution logs for debugging. Default is False.                                          |
| **Allow Delegation** *(optional)*       | `allow_delegation`       | `bool`                                | Allow the agent to delegate tasks to other agents. Default is False.                                     |
| **Step Callback** *(optional)*          | `step_callback`          | `Optional[Any]`                       | Function called after each agent step, overrides crew callback.                                          |
| **Cache** *(optional)*                  | `cache`                  | `bool`                                | Enable caching for tool usage. Default is True.                                                          |
| **System Template** *(optional)*        | `system_template`        | `Optional[str]`                       | Custom system prompt template for the agent.                                                             |
| **Prompt Template** *(optional)*        | `prompt_template`        | `Optional[str]`                       | Custom prompt template for the agent.                                                                    |
| **Response Template** *(optional)*      | `response_template`      | `Optional[str]`                       | Custom response template for the agent.                                                                  |
| **Allow Code Execution** *(optional)*   | `allow_code_execution`   | `Optional[bool]`                      | Enable code execution for the agent. Default is False.                                                   |
| **Max Retry Limit** *(optional)*        | `max_retry_limit`        | `int`                                 | Maximum number of retries when an error occurs. Default is 2.                                            |
| **Respect Context Window** *(optional)* | `respect_context_window` | `bool`                                | Keep messages under context window size by summarizing. Default is True.                                 |
| **Code Execution Mode** *(optional)*    | `code_execution_mode`    | `Literal["safe", "unsafe"]`           | Mode for code execution: 'safe' (using Docker) or 'unsafe' (direct). Default is 'safe'.                  |
| **Multimodal** *(optional)*             | `multimodal`             | `bool`                                | Whether the agent supports multimodal capabilities. Default is False.                                    |
| **Inject Date** *(optional)*            | `inject_date`            | `bool`                                | Whether to automatically inject the current date into tasks. Default is False.                           |
| **Date Format** *(optional)*            | `date_format`            | `str`                                 | Format string for date when inject\_date is enabled. Default is "%Y-%m-%d" (ISO format).                 |
| **Reasoning** *(optional)*              | `reasoning`              | `bool`                                | Whether the agent should reflect and create a plan before executing a task. Default is False.            |
| **Max Reasoning Attempts** *(optional)* | `max_reasoning_attempts` | `Optional[int]`                       | Maximum number of reasoning attempts before executing the task. If None, will try until ready.           |
| **Embedder** *(optional)*               | `embedder`               | `Optional[Dict[str, Any]]`            | Configuration for the embedder used by the agent.                                                        |
| **Knowledge Sources** *(optional)*      | `knowledge_sources`      | `Optional[List[BaseKnowledgeSource]]` | Knowledge sources available to the agent.                                                                |
| **Use System Prompt** *(optional)*      | `use_system_prompt`      | `Optional[bool]`                      | Whether to use system prompt (for o1 model support). Default is True.                                    |

## Creating Agents

There are two ways to create agents in CrewAI: using **YAML configuration (recommended)** or defining them **directly in code**.

### YAML Configuration (Recommended)

Using YAML configuration provides a cleaner, more maintainable way to define agents. We strongly recommend using this approach in your CrewAI projects.

After creating your CrewAI project as outlined in the [Installation](/installation) section, navigate to the `src/latest_ai_development/config/agents.yaml` file and modify the template to match your requirements.

<Note>
  Variables in your YAML files (like `{topic}`) will be replaced with values from your inputs when running the crew:

  ```python Code
  crew.kickoff(inputs={'topic': 'AI Agents'})
  ```
</Note>

Here's an example of how to configure agents using YAML:

```yaml agents.yaml
# src/latest_ai_development/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
```

To use this YAML configuration in your code, create a crew class that inherits from `CrewBase`:

```python Code
# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, crew
from crewai_tools import SerperDevTool

@CrewBase
class LatestAiDevelopmentCrew():
  """LatestAiDevelopment crew"""

  agents_config = "config/agents.yaml"

  @agent
  def researcher(self) -> Agent:
    return Agent(
      config=self.agents_config['researcher'], # type: ignore[index]
      verbose=True,
      tools=[SerperDevTool()]
    )

  @agent
  def reporting_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['reporting_analyst'], # type: ignore[index]
      verbose=True
    )
```

<Note>
  The names you use in your YAML files (`agents.yaml`) should match the method names in your Python code.
</Note>

### Direct Code Definition

You can create agents directly in code by instantiating the `Agent` class. Here's a comprehensive example showing all available parameters:

```python Code
from crewai import Agent
from crewai_tools import SerperDevTool

# Create an agent with all available parameters
agent = Agent(
    role="Senior Data Scientist",
    goal="Analyze and interpret complex datasets to provide actionable insights",
    backstory="With over 10 years of experience in data science and machine learning, "
              "you excel at finding patterns in complex datasets.",
    llm="gpt-4",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    verbose=False,  # Default: False
    allow_delegation=False,  # Default: False
    max_iter=20,  # Default: 20 iterations
    max_rpm=None,  # Optional: Rate limit for API calls
    max_execution_time=None,  # Optional: Maximum execution time in seconds
    max_retry_limit=2,  # Default: 2 retries on error
    allow_code_execution=False,  # Default: False
    code_execution_mode="safe",  # Default: "safe" (options: "safe", "unsafe")
    respect_context_window=True,  # Default: True
    use_system_prompt=True,  # Default: True
    multimodal=False,  # Default: False
    inject_date=False,  # Default: False
    date_format="%Y-%m-%d",  # Default: ISO format
    reasoning=False,  # Default: False
    max_reasoning_attempts=None,  # Default: None
    tools=[SerperDevTool()],  # Optional: List of tools
    knowledge_sources=None,  # Optional: List of knowledge sources
    embedder=None,  # Optional: Custom embedder configuration
    system_template=None,  # Optional: Custom system prompt template
    prompt_template=None,  # Optional: Custom prompt template
    response_template=None,  # Optional: Custom response template
    step_callback=None,  # Optional: Callback function for monitoring
)
```

Let's break down some key parameter combinations for common use cases:

#### Basic Research Agent

```python Code
research_agent = Agent(
    role="Research Analyst",
    goal="Find and summarize information about specific topics",
    backstory="You are an experienced researcher with attention to detail",
    tools=[SerperDevTool()],
    verbose=True  # Enable logging for debugging
)
```

#### Code Development Agent

```python Code
dev_agent = Agent(
    role="Senior Python Developer",
    goal="Write and debug Python code",
    backstory="Expert Python developer with 10 years of experience",
    allow_code_execution=True,
    code_execution_mode="safe",  # Uses Docker for safety
    max_execution_time=300,  # 5-minute timeout
    max_retry_limit=3  # More retries for complex code tasks
)
```

#### Long-Running Analysis Agent

```python Code
analysis_agent = Agent(
    role="Data Analyst",
    goal="Perform deep analysis of large datasets",
    backstory="Specialized in big data analysis and pattern recognition",
    memory=True,
    respect_context_window=True,
    max_rpm=10,  # Limit API calls
    function_calling_llm="gpt-4o-mini"  # Cheaper model for tool calls
)
```

#### Custom Template Agent

```python Code
custom_agent = Agent(
    role="Customer Service Representative",
    goal="Assist customers with their inquiries",
    backstory="Experienced in customer support with a focus on satisfaction",
    system_template="""<|start_header_id|>system<|end_header_id|>
                        {{ .System }}<|eot_id|>""",
    prompt_template="""<|start_header_id|>user<|end_header_id|>
                        {{ .Prompt }}<|eot_id|>""",
    response_template="""<|start_header_id|>assistant<|end_header_id|>
                        {{ .Response }}<|eot_id|>""",
)
```

#### Date-Aware Agent with Reasoning

```python Code
strategic_agent = Agent(
    role="Market Analyst",
    goal="Track market movements with precise date references and strategic planning",
    backstory="Expert in time-sensitive financial analysis and strategic reporting",
    inject_date=True,  # Automatically inject current date into tasks
    date_format="%B %d, %Y",  # Format as "May 21, 2025"
    reasoning=True,  # Enable strategic planning
    max_reasoning_attempts=2,  # Limit planning iterations
    verbose=True
)
```

#### Reasoning Agent

```python Code
reasoning_agent = Agent(
    role="Strategic Planner",
    goal="Analyze complex problems and create detailed execution plans",
    backstory="Expert strategic planner who methodically breaks down complex challenges",
    reasoning=True,  # Enable reasoning and planning
    max_reasoning_attempts=3,  # Limit reasoning attempts
    max_iter=30,  # Allow more iterations for complex planning
    verbose=True
)
```

#### Multimodal Agent

```python Code
multimodal_agent = Agent(
    role="Visual Content Analyst",
    goal="Analyze and process both text and visual content",
    backstory="Specialized in multimodal analysis combining text and image understanding",
    multimodal=True,  # Enable multimodal capabilities
    verbose=True
)
```

### Parameter Details

#### Critical Parameters

* `role`, `goal`, and `backstory` are required and shape the agent's behavior
* `llm` determines the language model used (default: OpenAI's GPT-4)

#### Memory and Context

* `memory`: Enable to maintain conversation history
* `respect_context_window`: Prevents token limit issues
* `knowledge_sources`: Add domain-specific knowledge bases

#### Execution Control

* `max_iter`: Maximum attempts before giving best answer
* `max_execution_time`: Timeout in seconds
* `max_rpm`: Rate limiting for API calls
* `max_retry_limit`: Retries on error

#### Code Execution

* `allow_code_execution`: Must be True to run code
* `code_execution_mode`:
  * `"safe"`: Uses Docker (recommended for production)
  * `"unsafe"`: Direct execution (use only in trusted environments)

<Note>
  This runs a default Docker image. If you want to configure the docker image, the checkout the Code Interpreter Tool in the tools section.
  Add the code interpreter tool as a tool in the agent as a tool parameter.
</Note>

#### Advanced Features

* `multimodal`: Enable multimodal capabilities for processing text and visual content
* `reasoning`: Enable agent to reflect and create plans before executing tasks
* `inject_date`: Automatically inject current date into task descriptions

#### Templates

* `system_template`: Defines agent's core behavior
* `prompt_template`: Structures input format
* `response_template`: Formats agent responses

<Note>
  When using custom templates, ensure that both `system_template` and `prompt_template` are defined. The `response_template` is optional but recommended for consistent output formatting.
</Note>

<Note>
  When using custom templates, you can use variables like `{role}`, `{goal}`, and `{backstory}` in your templates. These will be automatically populated during execution.
</Note>

## Agent Tools

Agents can be equipped with various tools to enhance their capabilities. CrewAI supports tools from:

* [CrewAI Toolkit](https://github.com/joaomdmoura/crewai-tools)
* [LangChain Tools](https://python.langchain.com/docs/integrations/tools)

Here's how to add tools to an agent:

```python Code
from crewai import Agent
from crewai_tools import SerperDevTool, WikipediaTools

# Create tools
search_tool = SerperDevTool()
wiki_tool = WikipediaTools()

# Add tools to agent
researcher = Agent(
    role="AI Technology Researcher",
    goal="Research the latest AI developments",
    tools=[search_tool, wiki_tool],
    verbose=True
)
```

## Agent Memory and Context

Agents can maintain memory of their interactions and use context from previous tasks. This is particularly useful for complex workflows where information needs to be retained across multiple tasks.

```python Code
from crewai import Agent

analyst = Agent(
    role="Data Analyst",
    goal="Analyze and remember complex data patterns",
    memory=True,  # Enable memory
    verbose=True
)
```

<Note>
  When `memory` is enabled, the agent will maintain context across multiple interactions, improving its ability to handle complex, multi-step tasks.
</Note>

## Context Window Management

CrewAI includes sophisticated automatic context window management to handle situations where conversations exceed the language model's token limits. This powerful feature is controlled by the `respect_context_window` parameter.

### How Context Window Management Works

When an agent's conversation history grows too large for the LLM's context window, CrewAI automatically detects this situation and can either:

1. **Automatically summarize content** (when `respect_context_window=True`)
2. **Stop execution with an error** (when `respect_context_window=False`)

### Automatic Context Handling (`respect_context_window=True`)

This is the **default and recommended setting** for most use cases. When enabled, CrewAI will:

```python Code
# Agent with automatic context management (default)
smart_agent = Agent(
    role="Research Analyst",
    goal="Analyze large documents and datasets",
    backstory="Expert at processing extensive information",
    respect_context_window=True,  # 🔑 Default: auto-handle context limits
    verbose=True
)
```

**What happens when context limits are exceeded:**

* ⚠️ **Warning message**: `"Context length exceeded. Summarizing content to fit the model context window."`
* 🔄 **Automatic summarization**: CrewAI intelligently summarizes the conversation history
* ✅ **Continued execution**: Task execution continues seamlessly with the summarized context
* 📝 **Preserved information**: Key information is retained while reducing token count

### Strict Context Limits (`respect_context_window=False`)

When you need precise control and prefer execution to stop rather than lose any information:

```python Code
# Agent with strict context limits
strict_agent = Agent(
    role="Legal Document Reviewer",
    goal="Provide precise legal analysis without information loss",
    backstory="Legal expert requiring complete context for accurate analysis",
    respect_context_window=False,  # ❌ Stop execution on context limit
    verbose=True
)
```

**What happens when context limits are exceeded:**

* ❌ **Error message**: `"Context length exceeded. Consider using smaller text or RAG tools from crewai_tools."`
* 🛑 **Execution stops**: Task execution halts immediately
* 🔧 **Manual intervention required**: You need to modify your approach

### Choosing the Right Setting

#### Use `respect_context_window=True` (Default) when:

* **Processing large documents** that might exceed context limits
* **Long-running conversations** where some summarization is acceptable
* **Research tasks** where general context is more important than exact details
* **Prototyping and development** where you want robust execution

```python Code
# Perfect for document processing
document_processor = Agent(
    role="Document Analyst", 
    goal="Extract insights from large research papers",
    backstory="Expert at analyzing extensive documentation",
    respect_context_window=True,  # Handle large documents gracefully
    max_iter=50,  # Allow more iterations for complex analysis
    verbose=True
)
```

#### Use `respect_context_window=False` when:

* **Precision is critical** and information loss is unacceptable
* **Legal or medical tasks** requiring complete context
* **Code review** where missing details could introduce bugs
* **Financial analysis** where accuracy is paramount

```python Code
# Perfect for precision tasks
precision_agent = Agent(
    role="Code Security Auditor",
    goal="Identify security vulnerabilities in code",
    backstory="Security expert requiring complete code context",
    respect_context_window=False,  # Prefer failure over incomplete analysis
    max_retry_limit=1,  # Fail fast on context issues
    verbose=True
)
```

### Alternative Approaches for Large Data

When dealing with very large datasets, consider these strategies:

#### 1. Use RAG Tools

```python Code
from crewai_tools import RagTool

# Create RAG tool for large document processing
rag_tool = RagTool()

rag_agent = Agent(
    role="Research Assistant",
    goal="Query large knowledge bases efficiently",
    backstory="Expert at using RAG tools for information retrieval",
    tools=[rag_tool],  # Use RAG instead of large context windows
    respect_context_window=True,
    verbose=True
)
```

#### 2. Use Knowledge Sources

```python Code
# Use knowledge sources instead of large prompts
knowledge_agent = Agent(
    role="Knowledge Expert",
    goal="Answer questions using curated knowledge",
    backstory="Expert at leveraging structured knowledge sources",
    knowledge_sources=[your_knowledge_sources],  # Pre-processed knowledge
    respect_context_window=True,
    verbose=True
)
```

### Context Window Best Practices

1. **Monitor Context Usage**: Enable `verbose=True` to see context management in action
2. **Design for Efficiency**: Structure tasks to minimize context accumulation
3. **Use Appropriate Models**: Choose LLMs with context windows suitable for your tasks
4. **Test Both Settings**: Try both `True` and `False` to see which works better for your use case
5. **Combine with RAG**: Use RAG tools for very large datasets instead of relying solely on context windows

### Troubleshooting Context Issues

**If you're getting context limit errors:**

```python Code
# Quick fix: Enable automatic handling
agent.respect_context_window = True

# Better solution: Use RAG tools for large data
from crewai_tools import RagTool
agent.tools = [RagTool()]

# Alternative: Break tasks into smaller pieces
# Or use knowledge sources instead of large prompts
```

**If automatic summarization loses important information:**

```python Code
# Disable auto-summarization and use RAG instead
agent = Agent(
    role="Detailed Analyst",
    goal="Maintain complete information accuracy",
    backstory="Expert requiring full context",
    respect_context_window=False,  # No summarization
    tools=[RagTool()],  # Use RAG for large data
    verbose=True
)
```

<Note>
  The context window management feature works automatically in the background. You don't need to call any special functions - just set `respect_context_window` to your preferred behavior and CrewAI handles the rest!
</Note>

## Important Considerations and Best Practices

### Security and Code Execution

* When using `allow_code_execution`, be cautious with user input and always validate it
* Use `code_execution_mode: "safe"` (Docker) in production environments
* Consider setting appropriate `max_execution_time` limits to prevent infinite loops

### Performance Optimization

* Use `respect_context_window: true` to prevent token limit issues
* Set appropriate `max_rpm` to avoid rate limiting
* Enable `cache: true` to improve performance for repetitive tasks
* Adjust `max_iter` and `max_retry_limit` based on task complexity

### Memory and Context Management

* Leverage `knowledge_sources` for domain-specific information
* Configure `embedder` when using custom embedding models
* Use custom templates (`system_template`, `prompt_template`, `response_template`) for fine-grained control over agent behavior

### Advanced Features

* Enable `reasoning: true` for agents that need to plan and reflect before executing complex tasks
* Set appropriate `max_reasoning_attempts` to control planning iterations (None for unlimited attempts)
* Use `inject_date: true` to provide agents with current date awareness for time-sensitive tasks
* Customize the date format with `date_format` using standard Python datetime format codes
* Enable `multimodal: true` for agents that need to process both text and visual content

### Agent Collaboration

* Enable `allow_delegation: true` when agents need to work together
* Use `step_callback` to monitor and log agent interactions
* Consider using different LLMs for different purposes:
  * Main `llm` for complex reasoning
  * `function_calling_llm` for efficient tool usage

### Date Awareness and Reasoning

* Use `inject_date: true` to provide agents with current date awareness for time-sensitive tasks
* Customize the date format with `date_format` using standard Python datetime format codes
* Valid format codes include: %Y (year), %m (month), %d (day), %B (full month name), etc.
* Invalid date formats will be logged as warnings and will not modify the task description
* Enable `reasoning: true` for complex tasks that benefit from upfront planning and reflection

### Model Compatibility

* Set `use_system_prompt: false` for older models that don't support system messages
* Ensure your chosen `llm` supports the features you need (like function calling)

## Troubleshooting Common Issues

1. **Rate Limiting**: If you're hitting API rate limits:
   * Implement appropriate `max_rpm`
   * Use caching for repetitive operations
   * Consider batching requests

2. **Context Window Errors**: If you're exceeding context limits:
   * Enable `respect_context_window`
   * Use more efficient prompts
   * Clear agent memory periodically

3. **Code Execution Issues**: If code execution fails:
   * Verify Docker is installed for safe mode
   * Check execution permissions
   * Review code sandbox settings

4. **Memory Issues**: If agent responses seem inconsistent:
   * Check knowledge source configuration
   * Review conversation history management

Remember that agents are most effective when configured according to their specific use case. Take time to understand your requirements and adjust these parameters accordingly.


# Custom Manager Agent

> Learn how to set a custom agent as the manager in CrewAI, providing more control over task management and coordination.

# Setting a Specific Agent as Manager in CrewAI

CrewAI allows users to set a specific agent as the manager of the crew, providing more control over the management and coordination of tasks.
This feature enables the customization of the managerial role to better fit your project's requirements.

## Using the `manager_agent` Attribute

### Custom Manager Agent

The `manager_agent` attribute allows you to define a custom agent to manage the crew. This agent will oversee the entire process, ensuring that tasks are completed efficiently and to the highest standard.

### Example

```python Code
import os
from crewai import Agent, Task, Crew, Process

# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research and analysis on AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Create compelling content about AI and AI agents",
    backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
    allow_delegation=False,
)

# Define your task
task = Task(
    description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
    expected_output="5 bullet points, each with a paragraph and accompanying notes.",
)

# Define the manager agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
)

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
)

# Start the crew's work
result = crew.kickoff()
```

## Benefits of a Custom Manager Agent

* **Enhanced Control**: Tailor the management approach to fit the specific needs of your project.
* **Improved Coordination**: Ensure efficient task coordination and management by an experienced agent.
* **Customizable Management**: Define managerial roles and responsibilities that align with your project's goals.

## Setting a Manager LLM

If you're using the hierarchical process and don't want to set a custom manager agent, you can specify the language model for the manager:

```python Code
from crewai import LLM

manager_llm = LLM(model="gpt-4o")

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    process=Process.hierarchical,
    manager_llm=manager_llm
)
```

<Note>
  Either `manager_agent` or `manager_llm` must be set when using the hierarchical process.
</Note>

# Customize Agents

> A comprehensive guide to tailoring agents for specific roles, tasks, and advanced customizations within the CrewAI framework.

## Customizable Attributes

Crafting an efficient CrewAI team hinges on the ability to dynamically tailor your AI agents to meet the unique requirements of any project. This section covers the foundational attributes you can customize.

### Key Attributes for Customization

| Attribute                           | Description                                                                                                         |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Role**                            | Specifies the agent's job within the crew, such as 'Analyst' or 'Customer Service Rep'.                             |
| **Goal**                            | Defines the agent’s objectives, aligned with its role and the crew’s overarching mission.                           |
| **Backstory**                       | Provides depth to the agent's persona, enhancing motivations and engagements within the crew.                       |
| **Tools** *(Optional)*              | Represents the capabilities or methods the agent uses for tasks, from simple functions to complex integrations.     |
| **Cache** *(Optional)*              | Determines if the agent should use a cache for tool usage.                                                          |
| **Max RPM**                         | Sets the maximum requests per minute (`max_rpm`). Can be set to `None` for unlimited requests to external services. |
| **Verbose** *(Optional)*            | Enables detailed logging for debugging and optimization, providing insights into execution processes.               |
| **Allow Delegation** *(Optional)*   | Controls task delegation to other agents, default is `False`.                                                       |
| **Max Iter** *(Optional)*           | Limits the maximum number of iterations (`max_iter`) for a task to prevent infinite loops, with a default of 25.    |
| **Max Execution Time** *(Optional)* | Sets the maximum time allowed for an agent to complete a task.                                                      |
| **System Template** *(Optional)*    | Defines the system format for the agent.                                                                            |
| **Prompt Template** *(Optional)*    | Defines the prompt format for the agent.                                                                            |
| **Response Template** *(Optional)*  | Defines the response format for the agent.                                                                          |
| **Use System Prompt** *(Optional)*  | Controls whether the agent will use a system prompt during task execution.                                          |
| **Respect Context Window**          | Enables a sliding context window by default, maintaining context size.                                              |
| **Max Retry Limit**                 | Sets the maximum number of retries (`max_retry_limit`) for an agent in case of errors.                              |

## Advanced Customization Options

Beyond the basic attributes, CrewAI allows for deeper customization to enhance an agent's behavior and capabilities significantly.

### Language Model Customization

Agents can be customized with specific language models (`llm`) and function-calling language models (`function_calling_llm`), offering advanced control over their processing and decision-making abilities.
It's important to note that setting the `function_calling_llm` allows for overriding the default crew function-calling language model, providing a greater degree of customization.

## Performance and Debugging Settings

Adjusting an agent's performance and monitoring its operations are crucial for efficient task execution.

### Verbose Mode and RPM Limit

* **Verbose Mode**: Enables detailed logging of an agent's actions, useful for debugging and optimization. Specifically, it provides insights into agent execution processes, aiding in the optimization of performance.
* **RPM Limit**: Sets the maximum number of requests per minute (`max_rpm`). This attribute is optional and can be set to `None` for no limit, allowing for unlimited queries to external services if needed.

### Maximum Iterations for Task Execution

The `max_iter` attribute allows users to define the maximum number of iterations an agent can perform for a single task, preventing infinite loops or excessively long executions.
The default value is set to 25, providing a balance between thoroughness and efficiency. Once the agent approaches this number, it will try its best to give a good answer.

## Customizing Agents and Tools

Agents are customized by defining their attributes and tools during initialization. Tools are critical for an agent's functionality, enabling them to perform specialized tasks.
The `tools` attribute should be an array of tools the agent can utilize, and it's initialized as an empty list by default. Tools can be added or modified post-agent initialization to adapt to new requirements.

```shell
pip install 'crewai[tools]'
```

### Example: Assigning Tools to an Agent

```python Code
import os
from crewai import Agent
from crewai_tools import SerperDevTool

# Set API keys for tool initialization
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key"

# Initialize a search tool
search_tool = SerperDevTool()

# Initialize the agent with advanced options
agent = Agent(
  role='Research Analyst',
  goal='Provide up-to-date market analysis',
  backstory='An expert analyst with a keen eye for market trends.',
  tools=[search_tool],
  memory=True, # Enable memory
  verbose=True,
  max_rpm=None, # No limit on requests per minute
  max_iter=25, # Default value for maximum iterations
)
```

## Delegation and Autonomy

Controlling an agent's ability to delegate tasks or ask questions is vital for tailoring its autonomy and collaborative dynamics within the CrewAI framework. By default,
the `allow_delegation` attribute is now set to `False`, disabling agents to seek assistance or delegate tasks as needed. This default behavior can be changed to promote collaborative problem-solving and
efficiency within the CrewAI ecosystem. If needed, delegation can be enabled to suit specific operational requirements.

### Example: Disabling Delegation for an Agent

```python Code
agent = Agent(
  role='Content Writer',
  goal='Write engaging content on market trends',
  backstory='A seasoned writer with expertise in market analysis.',
  allow_delegation=True # Enabling delegation
)
```

## Conclusion

Customizing agents in CrewAI by setting their roles, goals, backstories, and tools, alongside advanced options like language model customization, memory, performance settings, and delegation preferences,
equips a nuanced and capable AI team ready for complex challenges.


