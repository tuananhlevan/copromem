# Dataset connector contract for CoProMem

The connector writes three JSON/JSONL files. Only the connector knows the native benchmark APIs.

`run.json`:

```json
{"dataset_config":"dataset.json","output":"output/result.json"}
```

`dataset.json` defines `name`, `operations`, `episodes`, `tasks`, and `app_descriptions`. Paths are relative to `run.json`. Each operation needs a stable `operation` in `<source>.<domain>.<action>` form, `app`, `function_name`, `access_mode` (`read` or `write`), `required_parameters`, and `output_slots`. Optional `http_method`, `dependency_edges`, and `operation_aliases` are public metadata. The registry hash is computed from these fields. Keep the same registry across training and query tasks.

If exported field names differ, set `episode_fields`, `call_fields`, or `task_fields` in `dataset.json`. Each maps a normalized name to a source field or dotted path, for example `"episode_fields":{"task_id":"metadata.id","success":"score.won"}`. `operation_aliases` maps dataset operation names to registry operation names. This keeps routine dataset variation in configuration.

`episodes.jsonl` has one historical scored trajectory per line:

```json
{"task_id":"train-1","success":true,"calls":[{"operation":"tools.music.search_artist","inputs":{"artist_name":"A"},"outputs":{"artist_id":"1"},"schema_accepted":true,"response_success":true}]}
```

Use at least two successful, same-task trajectories to induce a schema. An episode's `success` must come from the dataset scorer. `schema_accepted` and `response_success` are required booleans based on actual validation and the native tool response; missing or false attestations fail closed. Include only public, accepted calls. The resulting graph and bank retain hashes and public slot names. Do not put secrets or hidden benchmark state in these files. The connector must strip any such values before export.

`tasks.jsonl` has one later query per line:

```json
{"task_id":"test-1","domain":"music","instruction":"Search and follow the artist in music"}
```

No answer, scorer outcome, expert path, or future tool call belongs in query rows. `app_descriptions` maps public domain names to descriptions. Results contain the frozen registry, bank, validation markers, query records, guidance, and full retrieval provenance. Empty guidance is a valid abstention.

For AppWorld or ALFWorld, write an external connector that maps native operations and scoring to this contract. Dataset-specific installation and runtime settings stay with that connector.

For live agent execution, the connector is an `AgentEnvironment` factory rather than a pre-exported episode file. It supplies native tool handlers and a scorer; the wrapper generates scored episodes in this same format. See [AGENT.md](AGENT.md).

For the externally validated terminal-slice path, put `"public_openapi_dir": "openapi"` in `dataset.json`. CoProMem reads public OpenAPI JSON files from that directory, matches each `operationId` to a declared operation in the same app, and freezes the descriptions plus source-file hashes. A dataset without OpenAPI files may instead supply a `public_operation_intents` object created by `build_intent_registry(...)`. No model generates these descriptions.

Validate each candidate schema on a held-out task in the isolated agent mode described in [AGENT.md](AGENT.md). Combine its successful receipts with `multi_schema_admission_bundle(bank, receipts)` and save that JSON as `"external_admission"` in the run config. The bundle admits each schema independently and binds its immutable retrieval semantics. New schemas remain excluded until separately admitted. Increasing support counts or adding other schemas does not invalidate an existing admission; changing an admitted schema's terminal, typed slots, or required operation order does. The CLI accepts `"initial_bank": "path/to/bank.json"` and `"learn": false` for fixed-bank evaluation.
