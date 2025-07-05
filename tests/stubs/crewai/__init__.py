"""Improved CrewAI stub (tests/stubs)."""
from __future__ import annotations
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional
from dataclasses import dataclass, field

# ------------------------------------------------------------------
# Memory layer ------------------------------------------------------
memory=ModuleType("crewai.memory")
class _BaseMem:
    def __init__(self):
        self._store=[]
    def save(self,obj):
        self._store.append(obj)
    def search(self,*_,**__):
        return self._store
    def get_context(self):
        return "stub-context"
ShortTermMemory=_BaseMem
LongTermMemory=_BaseMem
memory.short_term=ShortTermMemory()  # type: ignore[attr-defined]
memory.long_term=LongTermMemory()  # type: ignore[attr-defined]

storage=ModuleType("crewai.memory.storage")
rag_storage=ModuleType("crewai.memory.storage.rag_storage")
class RAGStorage(_BaseMem):
    pass
rag_storage.RAGStorage=RAGStorage  # type: ignore
storage.rag_storage=rag_storage  # type: ignore
memory.storage=storage  # type: ignore
sys.modules.update({
    "crewai.memory":memory,
    "crewai.memory.storage":storage,
    "crewai.memory.storage.rag_storage":rag_storage,
})

# ------------------------------------------------------------------
# Core abstractions -------------------------------------------------
@dataclass
class LLM:
    model:str="stub-llm"
    api_key:Optional[str]=None
    def invoke(self,*a,**k): return "stub-llm-response"

@dataclass
class Agent:
    role:str="agent"
    config:dict|None=None
    tools:list[Any]=field(default_factory=list)
    verbose:bool=False
    llm:Optional[LLM]=None
    def execute(self,*a,**k): return "agent-exec"

@dataclass
class Task:
    description:str="task"
    def execute(self,*a,**k): return "task-run"

class Process:
    sequential="sequential"
    parallel="parallel"
    hierarchical="hierarchical"

class Crew:
    def __init__(self,agents:List[Agent]|None=None,tasks:List[Task]|None=None,**kw):
        self.agents=agents or []
        self.tasks=tasks or []
        self.memory=kw.get("memory",False)
        self.agents_config={}
        self.tasks_config={}
    def kickoff(self,*a,**k): return {"result":"crew"}
    # checkpoint helpers for tests
    def save_state(self,*a,**k): pass

# Dummy decorators
_identity=lambda f:f
project=ModuleType("crewai.project")
setattr(project,"CrewBase",_identity)
setattr(project,"agent",_identity)
setattr(project,"crew",_identity)
setattr(project,"task",_identity)
sys.modules["crewai.project"]=project

crew_mod=ModuleType("crewai.crew"); crew_mod.Crew=Crew  # type: ignore
sys.modules["crewai.crew"]=crew_mod

agent_builder=ModuleType("crewai.agents.agent_builder.base_agent")
class BaseAgent:pass
agent_builder.BaseAgent=BaseAgent  # type: ignore
sys.modules["crewai.agents.agent_builder.base_agent"]=agent_builder

tools_mod=ModuleType("crewai.tools")
class BaseTool:
    name:str="base_tool"
    description:str="tool"
    def __init__(self,*_,**__): pass
    def _run(self,**k): return "tool-run"

tools_mod.BaseTool=BaseTool  # type: ignore
sys.modules["crewai.tools"]=tools_mod

# Export
__all__=["Agent","Crew","Process","Task","LLM"]
globals().update({"Agent":Agent,"Crew":Crew,"Process":Process,"Task":Task,"LLM":LLM,"memory":memory})