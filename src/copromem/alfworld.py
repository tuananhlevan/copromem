"""Optional native ALFWorld adapter for the generic CoProMem agent boundary.

The core package does not import ALFWorld.  This module loads it only while a
native session is opened, so ordinary CoProMem users need no benchmark
dependency.  Callers supply their own ALFWorld data root and game file.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .agent import AgentEnvironment, Tool, ToolRegistry


class AlfWorldUnavailableError(RuntimeError):
    """Raised when the optional native ALFWorld dependency is unavailable."""


def _native_config(data_root: str, *, max_steps: int) -> dict[str, Any]:
    """Return the minimal documented configuration required by AlfredTWEnv."""
    return {
        "dataset": {"data_path": data_root, "num_train_games": 1,
                    "num_eval_games": 1},
        "env": {"goal_desc_human_anns_prob": 0.0, "task_types": [1, 2, 3, 4, 5, 6],
                "domain_randomization": False, "expert_type": "handcoded"},
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


class AlfWorldSession:
    """One disposable, text-only native ALFWorld game session.

    ``data_root`` is the ALFWorld split directory that contains the selected
    game.  No data, task text, or provider credential is read by this adapter.
    ``native_environment_factory`` exists for tests and controlled runners;
    production callers normally leave it unset.
    """

    def __init__(self, gamefile: str | Path, data_root: str | Path, *, max_steps: int = 50,
                 native_environment_factory: Callable[[Mapping[str, Any]], Any] | None = None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.gamefile = str(Path(gamefile))
        self.data_root = str(Path(data_root))
        self.max_steps = max_steps
        self._factory = native_environment_factory
        self._environment: Any | None = None
        self._step = 0
        self._won = False
        self._closed = False

    @property
    def success(self) -> bool:
        return self._won

    def _open(self) -> Any:
        if self._closed:
            raise RuntimeError("ALFWorld session is closed")
        if self._environment is not None:
            return self._environment
        config = _native_config(self.data_root, max_steps=self.max_steps)
        if self._factory is None:
            try:
                from alfworld.agents.environment import AlfredTWEnv
            except ImportError as exc:
                raise AlfWorldUnavailableError(
                    "ALFWorld is optional; install its native dependencies and provide a licensed data root") from exc
            main_env = AlfredTWEnv(config, train_eval="train")
        else:
            main_env = self._factory(config)
        main_env.game_files = [self.gamefile]
        self._environment = main_env.init_env(batch_size=1)
        self._environment.reset()
        return self._environment

    @staticmethod
    def _item(value: Any) -> Any:
        return value[0] if isinstance(value, (list, tuple)) and value else value

    def execute_command(self, command: str) -> dict[str, Any]:
        """Execute one native command and return only stable public fields."""
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a nonempty string")
        if self._step >= self.max_steps:
            return {"observation": "Maximum steps reached.", "reward": 0.0,
                    "terminated": False, "truncated": True, "step": self._step}
        env = self._open()
        observation, native_reward, done, info = env.step([command.strip()])
        self._step += 1
        won_values = info.get("won", []) if isinstance(info, Mapping) else []
        self._won = bool(self._item(won_values))
        terminated = bool(self._item(done)) or self._won
        reward = self._item(native_reward)
        if not isinstance(reward, (int, float)):
            raise RuntimeError("ALFWorld returned a nonnumeric reward")
        return {"observation": str(self._item(observation) or ""), "reward": float(reward),
                "terminated": terminated, "truncated": self._step >= self.max_steps and not terminated,
                "step": self._step}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._environment is not None:
            closer = getattr(self._environment, "close", None)
            if callable(closer):
                closer()


def alfworld_tool(session: AlfWorldSession) -> Tool:
    """Build the one public command operation exposed to an ALFWorld agent."""
    return Tool(
        name="alfworld_command", operation="tools.alfworld.execute_command",
        description="Execute one text command in the current ALFWorld game.", access_mode="write",
        parameters={"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"], "additionalProperties": False},
        output_slots=("observation", "reward", "terminated", "truncated", "step"),
        handler=session.execute_command,
    )


def make_alfworld_environment(*, gamefile: str | Path, data_root: str | Path, max_steps: int = 50,
                              native_environment_factory: Callable[[Mapping[str, Any]], Any] | None = None
                              ) -> AgentEnvironment:
    """Create a disposable generic-agent environment backed by one ALFWorld game."""
    session = AlfWorldSession(gamefile, data_root, max_steps=max_steps,
                              native_environment_factory=native_environment_factory)
    return AgentEnvironment(tools=ToolRegistry([alfworld_tool(session)]),
                            score=lambda _answer: session.success, close=session.close)
