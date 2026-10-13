from __future__ import annotations

import pytest
import sys

from copromem import (AgentEnvironment, AgentPipeline, AlfWorldUnavailableError, AlfWorldSession,
                      CoProMem, ScriptedProvider, Tool, ToolAgent, ToolRegistry,
                      make_alfworld_environment)


class _RunningEnvironment:
    def __init__(self, won: bool = False, reward: float = 0.0):
        self.won = won
        self.reward = reward
        self.commands: list[list[str]] = []
        self.closed = False

    def reset(self):
        return None

    def step(self, commands):
        self.commands.append(commands)
        return ["native observation"], [self.reward], [self.won], {"won": [self.won]}

    def close(self):
        self.closed = True


class _MainEnvironment:
    def __init__(self, running: _RunningEnvironment):
        self.running = running
        self.game_files: list[str] = []

    def init_env(self, *, batch_size: int):
        assert batch_size == 1
        return self.running


def test_native_session_is_lazy_and_closes_its_environment() -> None:
    running = _RunningEnvironment(won=True, reward=1.0)
    seen = {}

    def factory(config):
        seen.update(config)
        return _MainEnvironment(running)

    session = AlfWorldSession("/data/game.tw-pddl", "/data", max_steps=3,
                              native_environment_factory=factory)
    result = session.execute_command("look")
    assert result == {"observation": "native observation", "reward": 1.0, "terminated": True,
                      "truncated": False, "step": 1}
    assert running.commands == [["look"]]
    assert seen["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
    session.close()
    assert running.closed is True
    session.close()  # idempotent cleanup is essential for native workers.


def test_environment_exposes_only_the_public_command_operation() -> None:
    running = _RunningEnvironment()
    environment = make_alfworld_environment(
        gamefile="/data/game.tw-pddl", data_root="/data",
        native_environment_factory=lambda _config: _MainEnvironment(running),
    )
    row = environment.tools.registry_rows()
    assert row == [{"operation": "tools.alfworld.execute_command", "app": "alfworld",
                    "function_name": "execute_command", "access_mode": "write",
                    "required_parameters": ["command"],
                    "output_slots": ["observation", "reward", "terminated", "truncated", "step"]}]
    assert environment.tools.by_name["alfworld_command"].invoke({"command": "inventory"})["step"] == 1
    environment.close()
    assert running.closed


def test_missing_optional_dependency_fails_only_when_session_opens(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "alfworld", None)
    session = AlfWorldSession("/no/game.tw-pddl", "/no/data")
    with pytest.raises(AlfWorldUnavailableError, match="optional"):
        session.execute_command("look")


def test_pipeline_closes_a_native_environment_when_provider_fails() -> None:
    dataset = {"name": "native", "operations": [{
        "operation": "tools.alfworld.execute_command", "app": "alfworld", "function_name": "execute_command",
        "access_mode": "write", "required_parameters": ["command"],
        "output_slots": ["observation", "reward", "terminated", "truncated", "step"],
    }]}
    memory = CoProMem(dataset)
    closed: list[bool] = []
    tool = Tool(name="alfworld_command", operation="tools.alfworld.execute_command", description="Command",
                access_mode="write", parameters={"type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}, output_slots=("observation", "reward", "terminated", "truncated", "step"),
                handler=lambda command: {"observation": command, "reward": 0.0, "terminated": False,
                                         "truncated": False, "step": 1})
    pipeline = AgentPipeline(memory, ToolAgent(ScriptedProvider([])),
                             lambda _task, _trial: AgentEnvironment(ToolRegistry([tool]), lambda _answer: False,
                                                                       close=lambda: closed.append(True)))
    with pytest.raises(RuntimeError, match="ran out"):
        pipeline.run_task({"task_id": "task", "instruction": "look", "domain": "alfworld"}, learn=False)
    assert closed == [True]
