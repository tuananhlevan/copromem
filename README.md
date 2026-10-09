# CoProMem

Code for the procedure memory method described in [the paper](PAPER.pdf). This repository carries forward the original dataset-independent runner and agent adapter, with the learning and retrieval rules aligned to the method section.

CoProMem builds a bank of reusable operation schemas from **scored task batches**. It keeps the greedy ordered operation core shared by successful trials, retains only slot-value equalities observed in every success, and admits a schema after one success in a one-trial task or two successes in a multi-trial task. Before a later task executes, it retrieves against the public task description and the frozen operation registry. Retrieval can abstain.

## Install and run

Python 3.10 or later is required. The core has no runtime dependencies.

```bash
python3 -m pip install -e '.[test]'
copromem examples/toy/run.json
copromem agent examples/toy/agent_run.json
python3 -m pytest
```

The offline example consumes scored historical episodes and writes `examples/toy/output/result.json`. The agent example runs fresh trials with a scripted provider. See [the dataset contract](docs/DATASET_CONTRACT.md) and [agent guide](docs/AGENT.md) for connecting a benchmark or model provider.

## Task boundary

```python
from copromem import CoProMem

memory = CoProMem(dataset_config)
prepared = memory.prepare(public_task)  # once, before any task trial
# Pass prepared["guidance"] to every fresh trial of this task.
update = memory.learn(task_id, scored_batch, expected_trials=len(scored_batch))
```

For an offline dataset, set `trials_per_task` in its configuration to enforce complete batches. The agent runner passes its configured trial count automatically. Save `memory.state` to reuse a fixed bank; omit `learn` during fixed-bank evaluation.

The package does not ship AppWorld or ALFWorld environments or benchmark data. The paper's reported benchmark numbers are not produced by the toy example. The optional public-API admission path inherited from the original codebase is an extension beyond the core method evaluated here.

## Layout

- `src/copromem/graph.py`, `semantic.py`, `spine.py`: evidence, domain projection, schema induction and validation.
- `src/copromem/task_query.py`, `retrieval.py`: public-query derivation, compatibility checks and reproducible retrieval.
- `src/copromem/pipeline.py`: offline and task-boundary interfaces.
- `src/copromem/agent.py`, `providers.py`, `agent_run.py`: provider-neutral execution and scoring.
- `tests/`: method and runner checks.
