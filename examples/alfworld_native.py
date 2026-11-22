"""Three-trial, zero-provider ALFWorld native integration gate.

Set ALFWORLD_DATA_ROOT and ALFWORLD_GAMEFILE to data you are licensed to use.
The scripted ``look`` call deliberately does not attempt to solve a benchmark
task; it only checks that fresh native workers and the tool boundary connect.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable

from copromem import AgentEnvironment, ScriptedProvider, ToolAgent, make_alfworld_environment


TRIALS_PER_TASK = 3


def run_gate(*, gamefile: str, data_root: str,
             environment_factory: Callable[..., AgentEnvironment] = make_alfworld_environment
             ) -> dict:
    """Check three fresh sessions, retaining every native score, including failure.

    ``passed`` describes the integration contract, not benchmark task success.
    The factory injection permits offline contract tests without ALFWorld data.
    """
    trials = []
    for trial in range(1, TRIALS_PER_TASK + 1):
        environment = environment_factory(gamefile=gamefile, data_root=data_root)
        try:
            provider = ScriptedProvider([
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"native-{trial}", "type": "function", "function": {
                        "name": "alfworld_command", "arguments": '{"command":"look"}'}}]},
                {"role": "assistant", "content": "Native smoke completed."},
            ])
            trace = ToolAgent(provider, max_steps=3, max_tool_calls=3).run(
                "Inspect the native environment.", "", environment.tools)
            if (not trace["completed"] or len(trace["calls"]) != 1 or
                    len(trace["events"]) != 1 or
                    trace["calls"][0]["operation"] != "tools.alfworld.execute_command"):
                raise RuntimeError(f"ALFWorld native gate failed on trial {trial}")
            native_success = environment.score(trace["answer"])
            if type(native_success) is not bool:
                raise ValueError("ALFWorld native scorer must return a boolean")
            trials.append({"trial": trial, "completed": trace["completed"],
                           "tool_calls": len(trace["calls"]),
                           "operation": trace["calls"][0]["operation"],
                           "native_success": native_success,
                           "model_calls": len(trace["model_calls"])})
        finally:
            if environment.close is not None:
                environment.close()
    successes = sum(row["native_success"] for row in trials)
    return {"gate": "alfworld_native_integration", "passed": True,
            "scope": "integration_smoke_not_benchmark_evaluation",
            "trials_per_task": TRIALS_PER_TASK, "trials": trials,
            "native_successes": successes,
            "native_success_avg_at_3": successes / TRIALS_PER_TASK}


def main() -> None:
    result = run_gate(data_root=os.environ["ALFWORLD_DATA_ROOT"],
                      gamefile=os.environ["ALFWORLD_GAMEFILE"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
