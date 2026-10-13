"""CoProMem: dataset-independent procedural memory."""

__version__ = "0.2.0"

from .pipeline import CoProMem, run
from .agent import AgentEnvironment, AgentPipeline, ChatProvider, Tool, ToolAgent, ToolRegistry
from .agent_run import run_agent
from .providers import OpenRouterProvider, ScriptedProvider
from .admission import (admission_receipt, admit_agent_report, multi_schema_admission_bundle,
                        schema_identity, verify_admission, verify_multi_schema_admission)
from .intents import build_intent_registry, build_intent_registry_from_openapi, verify_intent_registry
from .retrieval import external_retrieval_policy

__all__ = ["CoProMem", "run", "run_agent", "ChatProvider", "Tool", "ToolRegistry", "ToolAgent",
           "AgentEnvironment", "AgentPipeline", "OpenRouterProvider", "ScriptedProvider",
           "admission_receipt", "admit_agent_report", "verify_admission",
           "multi_schema_admission_bundle", "schema_identity", "verify_multi_schema_admission",
           "build_intent_registry", "build_intent_registry_from_openapi", "verify_intent_registry",
           "external_retrieval_policy"]
