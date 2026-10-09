# Method implemented by the core

The [paper](../PAPER.pdf) defines two task-boundary operations. `prepare` reads only the public task, public operation metadata and the pre-task bank. `learn` reads the complete scored batch after every trial finishes. The model and environment are supplied by the runner.

1. A frozen registry defines canonical operations, access modes, public required inputs, outputs, aliases and optional declared dependencies. Its digest binds each schema.
2. Calls with valid arguments and successful responses become an ordered evidence graph. Discovery, authentication and supervisor operations are audited and removed from the domain graph. Matching digests between an earlier public output and later public required input record an observed slot equality. Order and declared dependencies never create such an equality.
3. The learner greedily scans the first successful graph and matches operation/access-mode occurrences in order across all other successes. Failed trials do not prune this core. Aligned slot equalities are intersected across all successes. The last retained write is the preferred terminal; otherwise the final domain operation is used. Consecutive repeated terminal calls may form a group.
4. Validation requires a nonempty domain core, a valid terminal and forward relations between public slots. One configured trial needs one success; a larger batch needs two. A rejected candidate leaves the bank unchanged. A content-identical schema is not duplicated.
5. Retrieval derives supported operations from the public instruction and registry. It projects each schema around its terminal using only attested slot relations. It checks the terminal and every projected required input against the current public query or an incoming attested relation. It ranks compatible schemas by distinct supported nonterminal operations. A missing or tied winner yields empty guidance.

Guidance lists public operations and slots and asks the agent to verify current-task values for external inputs. Concrete past values are never stored in a schema. A digest equality attests matching values, not causation.

The inherited optional public-API admission path is separate from this core procedure. See [the dataset contract](DATASET_CONTRACT.md) for its configuration.
