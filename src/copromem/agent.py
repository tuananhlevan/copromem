"""Provider-neutral tool agent and task-boundary CoProMem runner."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .graph import digest
from .pipeline import CoProMem, _field


DEFAULT_SYSTEM_PROMPT = "Solve the user's task using the available tools. Report the outcome."


class ChatProvider(Protocol):
    """Return one assistant message in the chat-completions tool-call shape."""

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...


class ToolInputError(ValueError):
    """A proposed tool call failed local argument validation."""


def _validate(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> None:
    """Validate the common JSON Schema subset; custom validators cover other APIs."""
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "description"}
    unsupported = set(schema) - allowed
    if unsupported:
        raise ValueError(f"{path}: unsupported schema keywords {sorted(unsupported)}; provide a custom validator")
    kind = schema.get("type")
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str), "integer": lambda: type(value) is int,
             "number": lambda: type(value) in (int, float), "boolean": lambda: type(value) is bool,
             "null": lambda: value is None}
    if kind not in valid or not valid[kind]():
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside enum")
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}: properties must be a mapping")
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"{path}: missing {name}")
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties", False)
        if additional is False and extras:
            raise ValueError(f"{path}: unknown fields {sorted(extras)}")
        if isinstance(additional, Mapping):
            for name in extras:
                _validate(value[name], additional, f"{path}.{name}")
        elif additional is not True and additional is not False:
            raise ValueError(f"{path}: invalid additionalProperties schema")
        for name, item in value.items():
            if name in properties:
                _validate(item, properties[name], f"{path}.{name}")
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


@dataclass(frozen=True)
class Tool:
    """Map a model-visible function name to one public registry operation."""

    name: str
    operation: str
    description: str
    parameters: Mapping[str, Any]
    output_slots: tuple[str, ...]
    handler: Callable[..., Mapping[str, Any]]
    access_mode: str = "read"
    http_method: str | None = None
    validator: Callable[[Mapping[str, Any]], None] | None = None
    succeeded: Callable[[Mapping[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.name):
            raise ValueError("tool name must be a simple function identifier")
        parts = self.operation.split(".", 2)
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError("operation must be <source>.<domain>.<action>")
        if self.access_mode not in {"read", "write"}:
            raise ValueError("access_mode must be read or write")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object schema")
        if any(not isinstance(slot, str) or not slot for slot in self.output_slots):
            raise ValueError("output slots must be nonempty strings")

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": dict(self.parameters)}}

    def registry_row(self) -> dict[str, Any]:
        return {"operation": self.operation, "app": self.operation.split(".", 2)[1],
                "function_name": self.operation.split(".", 2)[2], "access_mode": self.access_mode,
                "required_parameters": list(self.parameters.get("required", [])),
                "output_slots": list(self.output_slots),
                **({"http_method": self.http_method} if self.http_method else {})}

    def invoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if self.validator:
                self.validator(arguments)
            else:
                _validate(arguments, self.parameters)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        result = self.handler(**arguments)
        if not isinstance(result, Mapping):
            raise ValueError("tool handler must return a mapping")
        if any(slot not in result for slot in self.output_slots):
            raise ValueError("tool response lacks a declared output slot")
        if self.succeeded is not None and self.succeeded(result) is not True:
            raise ValueError("tool response did not attest success")
        return dict(result)


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]):
        if not tools:
            raise ValueError("at least one tool is required")
        self.tools = list(tools)
        self.by_name = {tool.name: tool for tool in self.tools}
        if len(self.by_name) != len(self.tools) or len({tool.operation for tool in self.tools}) != len(self.tools):
            raise ValueError("tool names and operations must be unique")

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools]

    def registry_rows(self) -> list[dict[str, Any]]:
        return [tool.registry_row() for tool in self.tools]


@dataclass
class AgentEnvironment:
    """Create a fresh instance for each trial; the scorer reads that instance."""

    tools: ToolRegistry
    score: Callable[[str], bool]


class ToolAgent:
    def __init__(self, provider: ChatProvider, *, max_steps: int = 12, max_tool_calls: int = 20,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        if max_steps < 1 or max_tool_calls < 1:
            raise ValueError("agent limits must be positive")
        self.provider = provider
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.system_prompt = system_prompt

    def run(self, instruction: str, guidance: str, tools: ToolRegistry) -> dict[str, Any]:
        system = self.system_prompt + ("\n\nRetrieved procedure:\n" + guidance if guidance else "")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": instruction}]
        initial_prompt_sha256 = digest(messages)
        calls: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        model_calls: list[dict[str, Any]] = []
        answer = ""
        stop_reason = "step_limit"
        for _ in range(self.max_steps):
            message = self.provider.complete(messages, tools.specs())
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                raise ValueError("provider did not return an assistant message")
            if isinstance(message.get("_provider"), Mapping):
                model_calls.append(dict(message["_provider"]))
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("assistant tool_calls must be a list")
            assistant_message = {"role": "assistant", "content": message.get("content"),
                                 **({"tool_calls": tool_calls} if tool_calls else {})}
            if "reasoning_details" in message:
                assistant_message["reasoning_details"] = message["reasoning_details"]
            messages.append(assistant_message)
            if not tool_calls:
                answer = str(message.get("content") or "")
                stop_reason = "completed"
                break
            for call in tool_calls:
                if len(events) >= self.max_tool_calls:
                    stop_reason = "tool_limit"
                    break
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    raise ValueError("malformed provider tool call")
                call_id = call.get("id")
                name = call["function"].get("name")
                raw = call["function"].get("arguments")
                if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
                    raise ValueError("tool call lacks identity or name")
                event = {"tool_call_id": call_id, "name": name}
                try:
                    tool = tools.by_name[name]
                    try:
                        arguments = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError as exc:
                        raise ToolInputError("tool arguments must be valid JSON") from exc
                    if not isinstance(arguments, dict):
                        raise ToolInputError("tool arguments must be a JSON object")
                    result = tool.invoke(arguments)
                    content = json.dumps(result, ensure_ascii=False)
                    calls.append({"operation": tool.operation, "inputs": arguments,
                                  "outputs": {slot: result[slot] for slot in tool.output_slots},
                                  "schema_accepted": True, "response_success": True})
                    event["operation"] = tool.operation
                    event["response_success"] = True
                except ToolInputError as exc:
                    event["response_success"] = False
                    content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                except Exception as exc:
                    event["response_success"] = False
                    content = json.dumps({"error": "tool execution failed", "kind": type(exc).__name__})
                events.append(event)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            if stop_reason == "tool_limit":
                break
        return {"answer": answer, "completed": stop_reason == "completed", "stop_reason": stop_reason,
                "calls": calls, "events": events, "model_calls": model_calls, "messages": messages,
                "initial_prompt_sha256": initial_prompt_sha256, "guidance_sha256": digest(guidance)}


class AgentPipeline:
    """Run scored trials against one frozen pre-task bank, then learn once."""

    def __init__(self, memory: CoProMem, agent: ToolAgent,
                 environment_factory: Callable[[Mapping[str, Any], int], AgentEnvironment]):
        self.memory = memory
        self.agent = agent
        self.environment_factory = environment_factory

    @staticmethod
    def _mapped_row(values: Mapping[str, Any], fields: Mapping[str, str]) -> dict[str, Any]:
        """Place normalized evidence at the paths declared by a dataset adapter."""
        row: dict[str, Any] = {}
        for name, value in values.items():
            path = fields.get(name, name).split(".")
            if any(not part for part in path):
                raise ValueError(f"invalid field path for {name}")
            node = row
            for part in path[:-1]:
                existing = node.setdefault(part, {})
                if not isinstance(existing, dict):
                    raise ValueError("dataset field paths overlap")
                node = existing
            if path[-1] in node:
                raise ValueError("dataset field paths overlap")
            node[path[-1]] = value
        return row

    def run_task(self, task: Mapping[str, Any], *, trials: int = 2,
                 learn: bool = True, validation_schema_id: str | None = None,
                 validation_task_id: str | None = None) -> dict[str, Any]:
        if trials < 1:
            raise ValueError("trials must be positive")
        if validation_schema_id is not None:
            if learn or not validation_task_id:
                raise ValueError("candidate validation requires a held-out task and learn=false")
            prepared = self.memory.prepare_candidate_validation(
                task, schema_id=validation_schema_id, validation_task_id=validation_task_id)
        else:
            prepared = self.memory.prepare(task)
        pre_state_sha256 = digest(self.memory.state)
        episodes = []
        traces = []
        for trial in range(trials):
            environment = self.environment_factory(task, trial)
            if not isinstance(environment, AgentEnvironment):
                raise TypeError("environment factory must return AgentEnvironment")
            registry_rows = {row["operation"]: row for row in self.memory.registry["operations"]}
            if {row["operation"]: row for row in environment.tools.registry_rows()} != registry_rows:
                raise ValueError("environment tools do not match the frozen public registry")
            instruction = str(_field(task, self.memory.task_fields, "instruction"))
            trace = self.agent.run(instruction, prepared["guidance"], environment.tools)
            scored = environment.score(trace["answer"])
            if type(scored) is not bool:
                raise ValueError("environment scorer must return a boolean")
            calls = [self._mapped_row(call, self.memory.call_fields) for call in trace["calls"]]
            episodes.append(self._mapped_row({"task_id": prepared["task_id"],
                                             "success": scored, "calls": calls},
                                            self.memory.episode_fields))
            traces.append({"trial": trial, "success": scored, **trace})
        update = self.memory.learn(prepared["task_id"], episodes, expected_trials=trials) if learn else None
        return {"task_id": prepared["task_id"], "prepared": prepared,
                "pre_state_sha256": pre_state_sha256, "trials": traces, "update": update,
                "post_state_sha256": digest(self.memory.state)}
