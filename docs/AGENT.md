# Run an LLM agent with CoProMem

`copromem agent examples/toy/agent_run.json` exercises the complete loop locally without an API key. It loads a dataset config, fresh tool environment, scripted model replies, scorer, and CoProMem. The first query uses the empty bank; both scored trials finish before one learning update. Results include the model-visible initial prompt hash, retrieval provenance, model response IDs/usage when available, tool events, scores, and the final bank.

For OpenRouter, set `OPENROUTER_API_KEY` and replace the provider block:

```json
{
  "dataset_config": "dataset.json",
  "environment_factory": "agent_env:make_environment",
  "provider": {"type": "openrouter", "model": "provider/model-name", "temperature": 0, "max_tokens": 1024},
  "trials": 2,
  "max_steps": 12,
  "max_tool_calls": 20,
  "output": "output/agent_result.json"
}
```

Use a model that supports tool calling. The OpenRouter wrapper sends non-streaming requests to [chat completions](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) and follows its [client tool loop](https://openrouter.ai/docs/guides/features/tool-calling): assistant tool calls are executed locally and returned as `tool` messages, with tool definitions included again on subsequent requests. The API key is read from `OPENROUTER_API_KEY` by default; `api_key_env` can name a different environment variable.

The optional `url` provider setting can point to another HTTPS endpoint with the same chat-completions request and response shape. API-specific adapters can instead implement `ChatProvider` directly.

The environment factory is a Python `module:callable` next to the run config. It receives `(task, trial_index)` and returns `AgentEnvironment(tools, score)`. Create a fresh environment on each call, so trials do not share world state. See `examples/toy/agent_env.py`. Every `Tool` maps one model-visible function name to one public registry operation; its declared required inputs, output slots, access mode, and HTTP method must match `dataset.json`. The tool handler returns a mapping. Use `succeeded` when an API reports application failures inside an otherwise successful HTTP response. For advanced argument schemas, provide a custom `validator`.

The scorer is an environment-owned callable that receives the agent's final answer and returns a boolean. The wrapper keeps only schema-valid, response-successful tool calls in CoProMem training evidence. It calls `prepare` once before all trials and `learn` once after they are scored. Set `"learn": false` to evaluate a fixed bank; `"initial_bank"` may point to a JSON bank or an earlier result JSON containing `bank`.

For held-out validation of one candidate schema, set `"learn": false`, supply `"initial_bank"`, and add `"candidate_validation": {"schema_id": "...", "task_id": "..."}` to the agent run config. The dataset task file must contain exactly that task. This uses a distinct provenance mode and produces scored traces. Call `admit_agent_report("output/validation.json", schema_id="...", validation_task_id="...")` after checking the independent scorer result. The report supplies the durable journal hash, or pass `journal_path` for a separate journal. Repeat for each candidate on its own held-out task. Create the deployment bundle with `multi_schema_admission_bundle(bank, receipts)` and save it as `"external_admission"` in the normal run config. A newly learned schema needs its own validation receipt before it can enter that bundle.

The agent wrapper writes evidence to the field paths in `episode_fields` and `call_fields` when the dataset config uses custom paths.

Any other model API can implement `ChatProvider.complete(messages, tools)` and return a chat-completions style assistant message with `role`, `content`, and optional `tool_calls`. Pass that provider to `ToolAgent`, then use `AgentPipeline` with a `CoProMem` instance and environment factory. The file CLI includes OpenRouter and a scripted provider; the Python API accepts other providers without changing CoProMem.

The result file includes model messages and tool inputs/outputs for local inspection. Keep it in the ignored output directory and remove sensitive or hidden dataset fields in the environment adapter before sharing it.
