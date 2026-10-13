"""Optional native AppWorld adapter for the generic CoProMem agent boundary.

The adapter creates one isolated ``AppWorld`` instance and exposes its native
safe Python execution boundary as a single tool. Benchmark-specific API
evidence and published evaluation protocols belong in an external connector;
this module ships no task data, scorer fixtures, or credentials.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent import AgentEnvironment, Tool, ToolRegistry


class AppWorldUnavailableError(RuntimeError):
    """Raised when the optional native AppWorld dependency is unavailable."""


class AppWorldSession:
    """One disposable AppWorld task session with official scoring."""

    def __init__(self, task_id: str, *, experiment_name: str = "copromem",
                 timeout_seconds: int = 30, native_factory: Callable[..., Any] | None = None):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ValueError("experiment_name must be a nonempty string")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.timeout_seconds = timeout_seconds
        self._factory = native_factory
        self._world: Any | None = None
        self._closed = False

    def _open(self) -> Any:
        if self._closed:
            raise RuntimeError("AppWorld session is closed")
        if self._world is not None:
            return self._world
        kwargs = {"task_id": self.task_id, "experiment_name": self.experiment_name,
                  "raise_on_failure": False, "timeout_seconds": self.timeout_seconds,
                  "raise_on_unsafe_syntax": True, "null_patch_unsafe_execution": True}
        if self._factory is None:
            try:
                from appworld import AppWorld
            except ImportError as exc:
                raise AppWorldUnavailableError(
                    "AppWorld is optional; install it and configure a licensed local task root") from exc
            self._world = AppWorld(**kwargs)
        else:
            self._world = self._factory(**kwargs)
        return self._world

    def execute_python(self, program: str) -> dict[str, Any]:
        """Execute one native program through AppWorld's own safety boundary."""
        if not isinstance(program, str) or not program.strip():
            raise ValueError("program must be a nonempty string")
        world = self._open()
        output = world.execute(program)
        completed = world.task_completed()
        if type(completed) is not bool:
            raise RuntimeError("AppWorld returned an invalid completion status")
        return {"output": "" if output is None else str(output), "completed": completed}

    def score(self) -> bool:
        """Call AppWorld's official scorer for the current native state."""
        success = getattr(self._open().evaluate(), "success", None)
        if type(success) is not bool:
            raise RuntimeError("AppWorld scorer returned an invalid success field")
        return success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._world is not None:
            closer = getattr(self._world, "close", None)
            if callable(closer):
                closer()


def appworld_tool(session: AppWorldSession) -> Tool:
    """Build the public native-program tool for a single AppWorld session."""
    return Tool(
        name="appworld_program", operation="tools.appworld.execute_python",
        description="Execute one safe Python program in the current AppWorld task.", access_mode="write",
        parameters={"type": "object", "properties": {"program": {"type": "string"}},
                    "required": ["program"], "additionalProperties": False},
        output_slots=("output", "completed"), handler=session.execute_python,
    )


def make_appworld_environment(*, task_id: str, experiment_name: str = "copromem",
                              timeout_seconds: int = 30,
                              native_factory: Callable[..., Any] | None = None) -> AgentEnvironment:
    """Create a disposable generic-agent environment backed by AppWorld."""
    session = AppWorldSession(task_id, experiment_name=experiment_name, timeout_seconds=timeout_seconds,
                              native_factory=native_factory)
    return AgentEnvironment(tools=ToolRegistry([appworld_tool(session)]),
                            score=lambda _answer: session.score(), close=session.close)
