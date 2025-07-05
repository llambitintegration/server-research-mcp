"""Light-weight stub of the `crewai` package so the test suite can run
without pulling in the real (heavy) CrewAI dependencies.

Only the minimal public surface used by our code and tests is implemented.
This is **not** feature-complete – just enough for import/patching.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable, List, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Core data classes mimicking CrewAI public API
# ---------------------------------------------------------------------------
@dataclass
class LLM:  # noqa: D101
    model: str
    api_key: Optional[str] = None

    def invoke(self, _prompt: str, **_kwargs):  # noqa: D401
        # Deterministic mock response
        return "LLM-stub response"

@dataclass
class Agent:  # noqa: D101
    config: dict | None = None
    tools: list[Any] = field(default_factory=list)
    verbose: bool = False
    llm: Optional[LLM] = None
    max_iter: int = 3
    respect_context_window: bool = True

    role: str = "agent"

    def execute(self, *args, **kwargs):  # noqa: D401
        return "agent-stub"


class Process:  # noqa: D101
    sequential = "sequential"
    parallel = "parallel"
    hierarchical = "hierarchical"


@dataclass
class Task:  # noqa: D101
    config: dict | None = None
    output_file: str | None = None
    guardrail: Optional[Callable[[str], tuple[bool, Any]]] = None
    max_retries: int = 1
    description: str = "task"

    def execute(self, *args, **kwargs):  # noqa: D401
        return "task-stub"


class Crew:  # noqa: D101
    def __init__(self, agents: List[Agent], tasks: List[Task], **kwargs):
        self.agents = agents
        self.tasks = tasks
        self.memory = kwargs.get("memory", False)

    def kickoff(self, *args, **kwargs):  # noqa: D401
        return {"result": "crew-stub"}


# ---------------------------------------------------------------------------
# Decorators used by CrewAI's declarative style – here they are no-ops.
# ---------------------------------------------------------------------------

def _identity_decorator(func: Callable):  # noqa: D401
    return func


# Build sub-modules with the expected structure so that `import crewai.project` etc. works.
project_module = ModuleType("crewai.project")
project_module.CrewBase = _identity_decorator  # type: ignore[attr-defined]
project_module.agent = lambda f: f  # type: ignore
project_module.crew = lambda f: f  # type: ignore
project_module.task = lambda f: f  # type: ignore
sys.modules["crewai.project"] = project_module

# crewai.crew submodule – expose Crew class
crew_module = ModuleType("crewai.crew")
crew_module.Crew = Crew  # type: ignore[attr-defined]
sys.modules["crewai.crew"] = crew_module

# crewai.agents.agent_builder.base_agent submodule – expose BaseAgent
base_agent_module = ModuleType("crewai.agents.agent_builder.base_agent")

class BaseAgent:  # noqa: D101
    pass

base_agent_module.BaseAgent = BaseAgent  # type: ignore[attr-defined]
sys.modules["crewai.agents.agent_builder.base_agent"] = base_agent_module

# crewai.tools submodule – expose BaseTool
tools_module = ModuleType("crewai.tools")
class BaseTool:  # noqa: D101
    name: str = "base_tool"
    description: str = "stub tool"

    def _run(self, **kwargs):  # noqa: D401
        return "tool-stub"

tools_module.BaseTool = BaseTool  # type: ignore[attr-defined]
sys.modules["crewai.tools"] = tools_module

# Useful export names at package level
__all__ = [
    "Agent",
    "Crew",
    "Process",
    "Task",
    "LLM",
]

globals().update({
    "Agent": Agent,
    "Crew": Crew,
    "Process": Process,
    "Task": Task,
    "LLM": LLM,
})

# ----------------- Memory submodule stubs ------------------------------------
memory_module = ModuleType("crewai.memory")
storage_module = ModuleType("crewai.memory.storage")
rag_storage_module = ModuleType("crewai.memory.storage.rag_storage")

class RAGStorage:  # noqa: D101
    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        return None

    def load(self, *args, **kwargs):
        return []

    def search(self, *args, **kwargs):
        return []

rag_storage_module.RAGStorage = RAGStorage  # type: ignore[attr-defined]
storage_module.rag_storage = rag_storage_module  # type: ignore[attr-defined]
memory_module.storage = storage_module  # type: ignore[attr-defined]

# Register modules
sys.modules["crewai.memory"] = memory_module
sys.modules["crewai.memory.storage"] = storage_module
sys.modules["crewai.memory.storage.rag_storage"] = rag_storage_module