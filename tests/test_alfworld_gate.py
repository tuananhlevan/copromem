"""Offline contract tests; no native benchmark data or provider requests."""
from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from copromem import make_alfworld_environment


run_gate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "examples/alfworld_native.py"))["run_gate"]


class _NativeWorker:
    def __init__(self, *, won: bool, fail: bool = False):
        self.won = won
        self.fail = fail
        self.commands = []
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def step(self, commands):
        self.commands.append(commands)
        if self.fail:
            raise RuntimeError("native worker failure")
        return ["observation"], [float(self.won)], [self.won], {"won": [self.won]}

    def close(self):
        self.closed = True


class _NativeFactory:
    def __init__(self, worker):
        self.worker = worker
        self.game_files = []

    def init_env(self, *, batch_size):
        assert batch_size == 1
        return self.worker


def _factory(workers, outcomes, *, fail_trial=None, invalid_score=False):
    def make_environment(*, gamefile, data_root):
        index = len(workers)
        worker = _NativeWorker(won=outcomes[index], fail=index == fail_trial)
        workers.append(worker)
        env = make_alfworld_environment(
            gamefile=gamefile, data_root=data_root,
            native_environment_factory=lambda _config: _NativeFactory(worker))
        if invalid_score:
            env.score = lambda _answer: "not a boolean"
        return env
    return make_environment


def test_three_fresh_trials_retain_all_native_scores_and_close_workers():
    workers = []
    result = run_gate(gamefile="/data/game.tw-pddl", data_root="/data",
                      environment_factory=_factory(workers, [True, False, True]))
    assert result["passed"] is True
    assert result["trials_per_task"] == 3
    assert [row["trial"] for row in result["trials"]] == [1, 2, 3]
    assert [row["native_success"] for row in result["trials"]] == [True, False, True]
    assert result["native_successes"] == 2
    assert result["native_success_avg_at_3"] == pytest.approx(2 / 3)
    assert len({id(worker) for worker in workers}) == 3
    assert all(worker.resets == 1 and worker.commands == [["look"]] and worker.closed
               for worker in workers)
    assert all(row["completed"] and row["tool_calls"] == 1 and row["model_calls"] == 0
               for row in result["trials"])


def test_integration_pass_does_not_require_or_claim_task_success():
    workers = []
    result = run_gate(gamefile="/data/game.tw-pddl", data_root="/data",
                      environment_factory=_factory(workers, [False, False, False]))
    assert result["passed"] is True
    assert result["scope"] == "integration_smoke_not_benchmark_evaluation"
    assert result["native_successes"] == 0
    assert result["native_success_avg_at_3"] == 0
    assert len(result["trials"]) == 3


def test_native_error_fails_gate_and_closes_current_worker():
    workers = []
    with pytest.raises(RuntimeError, match="trial 2"):
        run_gate(gamefile="/data/game.tw-pddl", data_root="/data",
                 environment_factory=_factory(workers, [False, False, False], fail_trial=1))
    assert len(workers) == 2
    assert all(worker.closed for worker in workers)


def test_invalid_native_score_is_not_coerced_to_success():
    workers = []
    with pytest.raises(ValueError, match="boolean"):
        run_gate(gamefile="/data/game.tw-pddl", data_root="/data",
                 environment_factory=_factory(workers, [False], invalid_score=True))
    assert len(workers) == 1
    assert workers[0].closed
