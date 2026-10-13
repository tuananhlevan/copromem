# CoProMem

CoProMem is a Python reference implementation of procedural memory for tool-using agents. It induces reusable, evidence-backed operation schemas from complete scored task batches, then retrieves only reproducible guidance before later tasks execute.

The public release includes a safe admitted terminal-slice retrieval policy. Its core learning rule is unchanged: learning consumes scored trajectories only after a complete task batch; retrieval consumes public task context and the frozen callable registry only.

## What the package provides

- Dataset-independent semantic-spine learning and fixed/dynamic banks.
- Provider-neutral agent adapter with a deterministic scripted provider for tests.
- Public OpenAPI description registry and content-addressed provenance.
- Independent schema-admission receipts for externally validated memories.
- Safe terminal-slice retrieval that renders one terminal operation for an exact public intent or a task-relevant public app.
- Fail-closed guards for cross-app retrieval, unproven prerequisites, registry drift, ambiguous terminal slices, and altered provenance.

The repository intentionally excludes benchmark datasets, live credentials, execution traces, provider logs, run artifacts, and unpublished evaluation outputs.

## Install

Python 3.10+ is required. The core has no runtime dependency outside the standard library.

```bash
python -m pip install -e '.[test]'
python -m pytest
copromem examples/toy/run.json
copromem agent examples/toy/agent_run.json
python examples/safe_retrieval.py
```

The last command is a zero-provider smoke pilot: it creates an admitted in-memory terminal schema and verifies that safe guidance is retrieved. It does not contact a model, embedding service, or benchmark environment.

## Quick task-boundary API

```python
from copromem import CoProMem

memory = CoProMem(dataset_config)
prepared = memory.prepare(public_task)       # before every task trial
system_prompt = "Use this guidance if applicable:\n" + prepared["guidance"]
# Execute and score every trial for this task.
update = memory.learn(task_id, scored_batch) # only after the complete batch
```

Use `state=memory.state` for a fixed bank. Dynamic users must call `learn` only after a complete scored batch. `prepared["provenance"]` is content-addressed and must reproduce before its guidance is injected into a prompt.

## Safe terminal-slice retrieval

The public admission path is designed for a runner that has separate held-out evidence for a schema. It requires a frozen public intent registry plus an independently validated admission receipt for each schema. Retrieval can return an empty string deliberately.

1. Build public operation intents from OpenAPI descriptions or summaries.
2. Learn an ordinary semantic-spine schema from complete scored trials.
3. Validate one candidate schema on a held-out task and issue an admission receipt.
4. Bundle the receipts for the frozen retrieval policy.
5. Retrieve before dispatch; reproduce the provenance before prompt injection.

A rendered slice contains only the selected terminal operation and its public input/output names. It never injects a prerequisite merely because it occurred in a learned trajectory. Distinct compatible terminal slices produce abstention. Equivalent slices may be rendered as one common slice, with every contributing schema retained in provenance.

See [the safe retrieval guide](docs/SAFE_RETRIEVAL.md), [dataset contract](docs/DATASET_CONTRACT.md), [agent guide](docs/AGENT.md), and [reproducibility guide](docs/REPRODUCIBILITY.md).

## Layout

- `src/copromem/graph.py`, `semantic.py`, `spine.py`: evidence, projection, induction, validation, and commit.
- `src/copromem/task_query.py`, `retrieval.py`: public-query derivation, terminal-slice retrieval, and provenance replay.
- `src/copromem/admission.py`, `intents.py`: content-addressed public intent and held-out admission custody.
- `src/copromem/agent.py`, `agent_run.py`, `providers.py`: provider-neutral execution boundary.
- `examples/`: offline, scripted-agent, and zero-provider safe-retrieval smoke examples.
- `tests/`: unit, integration, admission, and smoke checks.

## Scope

This is research software. The included toy examples validate interfaces and invariants; they do not reproduce benchmark results. For an external benchmark, install that benchmark separately, provide only its public task/tool metadata to CoProMem, and keep provider credentials outside the repository. See [SECURITY.md](SECURITY.md).

Licensed under [Apache-2.0](LICENSE).
