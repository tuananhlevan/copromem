"""Zero-provider native AppWorld smoke through CoProMem's tool interface.

Set APPWORLD_TASK_ID for a task in a separately installed, licensed AppWorld
environment. The scripted ``pass`` program only tests the integration.
"""
from __future__ import annotations

import json
import os

from copromem import ScriptedProvider, ToolAgent, make_appworld_environment


def main() -> None:
    environment = make_appworld_environment(task_id=os.environ["APPWORLD_TASK_ID"],
                                             experiment_name="copromem-native-smoke")
    provider = ScriptedProvider([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "native-1", "type": "function",
          "function": {"name": "appworld_program", "arguments": '{"program":"pass"}'}}]},
        {"role": "assistant", "content": "Native smoke completed."},
    ])
    try:
        trace = ToolAgent(provider, max_steps=3, max_tool_calls=3).run(
            "Inspect the native environment.", "", environment.tools)
        print(json.dumps({"completed": trace["completed"], "tool_calls": len(trace["calls"]),
                          "operation": trace["calls"][0]["operation"],
                          "official_success": environment.score(trace["answer"]),
                          "model_calls": len(trace["model_calls"])}, sort_keys=True))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
