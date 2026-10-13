from __future__ import annotations

import sys

import pytest

from copromem import AppWorldSession, AppWorldUnavailableError, make_appworld_environment


class _Score:
    def __init__(self, success: bool):
        self.success = success


class _World:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.programs: list[str] = []
        self.closed = False

    def execute(self, program: str):
        self.programs.append(program)
        return "native output"

    def task_completed(self) -> bool:
        return False

    def evaluate(self):
        return _Score(False)

    def close(self):
        self.closed = True


def test_native_session_uses_safe_runtime_defaults_and_official_score() -> None:
    captured = {}

    def factory(**kwargs):
        world = _World(**kwargs)
        captured["world"] = world
        return world

    session = AppWorldSession("task-1", experiment_name="smoke", timeout_seconds=7, native_factory=factory)
    assert session.execute_python("pass") == {"output": "native output", "completed": False}
    assert session.score() is False
    world = captured["world"]
    assert world.programs == ["pass"]
    assert world.kwargs["raise_on_unsafe_syntax"] is True
    assert world.kwargs["null_patch_unsafe_execution"] is True
    session.close()
    assert world.closed is True
    session.close()


def test_environment_exposes_only_native_program_boundary() -> None:
    world = _World()
    environment = make_appworld_environment(task_id="task-1", native_factory=lambda **_kwargs: world)
    assert environment.tools.registry_rows() == [{
        "operation": "tools.appworld.execute_python", "app": "appworld",
        "function_name": "execute_python", "access_mode": "write",
        "required_parameters": ["program"], "output_slots": ["output", "completed"],
    }]
    assert environment.tools.by_name["appworld_program"].invoke({"program": "pass"})["output"] == "native output"
    environment.close()
    assert world.closed


def test_missing_optional_dependency_fails_only_when_session_opens(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "appworld", None)
    with pytest.raises(AppWorldUnavailableError, match="optional"):
        AppWorldSession("task-1").execute_python("pass")
