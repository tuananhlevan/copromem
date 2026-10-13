"""Zero-provider native ALFWorld smoke through CoProMem's tool interface.

Set ALFWORLD_DATA_ROOT and ALFWORLD_GAMEFILE to data you are licensed to use.
The scripted ``look`` call deliberately does not attempt to solve a benchmark
task; it only proves the local native worker and tool boundary are connected.
"""
from __future__ import annotations

import json
import os

from copromem import ScriptedProvider, ToolAgent, make_alfworld_environment


def main() -> None:
    data_root = os.environ["ALFWORLD_DATA_ROOT"]
    gamefile = os.environ["ALFWORLD_GAMEFILE"]
    environment = make_alfworld_environment(gamefile=gamefile, data_root=data_root)
    provider = ScriptedProvider([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "native-1", "type": "function",
          "function": {"name": "alfworld_command", "arguments": '{"command":"look"}'}}]},
        {"role": "assistant", "content": "Native smoke completed."},
    ])
    try:
        trace = ToolAgent(provider, max_steps=3, max_tool_calls=3).run(
            "Inspect the native environment.", "", environment.tools)
        print(json.dumps({"completed": trace["completed"], "tool_calls": len(trace["calls"]),
                          "operation": trace["calls"][0]["operation"],
                          "native_success": environment.score(trace["answer"]),
                          "model_calls": len(trace["model_calls"])}, sort_keys=True))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
