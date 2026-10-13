# CoProMem safe terminal-slice method guide

## Objective

A tool-using agent should reuse a successful procedure only when the new task supplies public evidence that makes the reuse safe. CoProMem separates this into an **offline learning boundary** and an **online retrieval boundary**.

- Learning observes fully executed and officially scored trajectories.
- Retrieval observes only the current public task description, public application descriptions, a frozen callable registry, and a previously committed bank.
- The online path never reads an outcome, hidden state, future action, scorer result, or provider response from the current task.

## Bank and update policy

A schema is induced from a complete same-task batch. The semantic spine preserves the ordered operation core shared by successful trials and response-attested public value-flow constraints. The bank stores the required operation sequence, terminal effect, typed constraints, positive/negative support, and content-addressed audit material.

Fixed evaluation loads a read-only bank. Dynamic evaluation calls `learn` only after all configured trials for one task have complete scored evidence. A failed validation leaves the pre-update bank untouched.

## Public intent registry

`build_intent_registry_from_openapi` extracts an operation description or summary from public OpenAPI JSON and binds each source file hash. The resulting registry is content-addressed. A runtime never treats a test fixture, a model completion, or an execution trace as an API description.

## Admitted multi-schema retrieval

A candidate schema becomes available to the public terminal-slice path only after a separate held-out validation provides an admission receipt. A receipt binds the source bank, schema identity, task identity, artifact, scorer evidence, journal, and retrieval-policy hash. `multi_schema_admission_bundle` sorts and binds these independently admitted entries.

At retrieval time safe terminal-slice:

1. derives a query from public task text and the frozen callable registry;
2. verifies the public intent registry and every admitted schema identity;
3. accepts a terminal only when its application is public and task-relevant, and either the public intent matches exactly or the admitted app supplies one unambiguous terminal slice;
4. groups compatible candidates by terminal-slice semantics rather than schema ID, support count, or serialization order;
5. renders one terminal-only instruction or abstains.

The terminal-only rendering retains the operation name and public input/output slots. It excludes earlier learned steps unless they are independently justified for the current public task. Thus a Spotify task cannot surface a Venmo operation, and a previously observed contact lookup cannot be injected into a task merely because a later text-message operation is relevant.

## Reproducibility contract

Every retrieval records the pre-state hash, query, registry hash, candidate inventory, selected terminal slice, rendered guidance hash, and `retrieval_sha256`. `reproduce_retrieval` recomputes the full record and fails on any difference. The agent runner places the guidance in the system prompt before the first provider request and binds the prompt hash in its trial evidence.

## Explicit abstention

Empty guidance is a valid output. It occurs when no admitted schema is compatible, when an app is only historical context, or when multiple distinct terminal slices are equally compatible. The method never resolves such a tie using schema ID, support count, candidate order, or success frequency.

## What changed in safe terminal-slice

The learning/common-core mechanism is preserved. The retrieval policy adds multi-schema admission and allows a unique, independently admitted terminal capability for a named task-relevant public app even when the task wording does not spell out that exact API operation. It keeps the cross-app guard, terminal-only rendering, provenance replay, and semantic-tie abstention.
