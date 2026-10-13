# Reproducibility and benchmark integration

## Included checks

Run the local, zero-provider checks after installation:

```bash
python -m pytest
python examples/safe_retrieval.py
copromem examples/toy/run.json
copromem agent examples/toy/agent_run.json
```

The tests exercise ordinary learning, task-boundary retrieval, admission custody, OpenAPI intent loading, cross-application rejection, semantic-tie abstention, provenance replay, and a scripted agent run. No command above needs a network call or credential.

## External benchmark adapter contract

The package deliberately does not ship AppWorld, other benchmark assets, task payloads, model logs, or provider credentials. To integrate a benchmark:

1. Install the benchmark in its own environment and follow its license.
2. Freeze a callable registry and public app descriptions before dispatch.
3. Build public operation intents from public OpenAPI files where available.
4. Call `prepare` once before each task's first trial, and bind the returned guidance and provenance to the model-visible prompt.
5. Store full execution evidence and official scorer output outside this repository.
6. Call `learn` only when the complete configured trial batch is settled.
7. Save the bank and admission bundle with their hashes for a fixed or dynamic run.

A benchmark adapter must reject registry drift, incomplete task batches, altered retrieval provenance, and an attempt to use an unadmitted schema in the external public-intent path.

## Releasing results

Do not commit raw task instructions, hidden state, account data, credentials, provider requests/responses, journals, scored artifacts, or evaluation reports unless their licenses and privacy constraints explicitly permit it. Publish aggregate statistics and a separately reviewed reproducibility package instead.
