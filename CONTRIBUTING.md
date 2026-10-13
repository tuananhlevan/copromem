# Contributing

1. Create a branch from `main`.
2. Keep the core dependency-free unless a dependency is necessary for a clearly separated optional adapter.
3. Add a regression test for every behavioral change.
4. Run `python -m pytest`, the offline example, and the safe-retrieval smoke example.
5. Do not commit credentials, benchmark datasets, execution logs, task payloads, or generated evaluation artifacts.

Changes to retrieval must preserve reproducible provenance and may not add schema-ID, support-count, or candidate-order tie breaking.
