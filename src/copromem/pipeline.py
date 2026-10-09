"""Dataset configuration -> attested evidence -> CoProMem -> results."""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .graph import digest
from .semantic import build_semantic_graph
from .spine import plan, validate, commit
from .task_query import derive_task_query
from .retrieval import retrieve, reproduce_retrieval, candidate_validation_retrieve
from .intents import build_intent_registry_from_openapi, verify_intent_registry


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _dataset_config(config_path: Path, value: str) -> dict[str, Any]:
    dataset_path = _path(config_path, value)
    dataset = _load(dataset_path)
    if "public_openapi_dir" in dataset:
        if "public_operation_intents" in dataset:
            raise ValueError("choose public_openapi_dir or public_operation_intents")
        dataset["public_operation_intents"] = build_intent_registry_from_openapi(
            _path(dataset_path, dataset["public_openapi_dir"]), dataset["operations"])
    return dataset


def _registry(config: Mapping[str, Any]) -> dict[str, Any]:
    operations = config.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("dataset config needs a nonempty operations list")
    body = {"operations": operations, "dependency_edges": config.get("dependency_edges", []),
            "normalization": {"operation_aliases": config.get("operation_aliases", {})}}
    return {**body, "registry_sha256": digest(body)}


def _field(row: Mapping[str, Any], mapping: Mapping[str, str], name: str) -> Any:
    value: Any = row
    for part in mapping.get(name, name).split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"missing required field: {mapping.get(name, name)}")
        value = value[part]
    return value


def _episode_records(episode: Mapping[str, Any], registry: Mapping[str, Any],
                     episode_fields: Mapping[str, str], call_fields: Mapping[str, str]) -> list[dict[str, Any]]:
    index = {row["operation"]: row for row in registry["operations"]}
    records = []
    for position, call in enumerate(_field(episode, episode_fields, "calls")):
        operation = str(_field(call, call_fields, "operation"))
        operation = str(registry["normalization"]["operation_aliases"].get(operation, operation))
        if operation not in index:
            raise ValueError(f"unknown operation: {operation}")
        meta = index[operation]
        inputs = _field(call, call_fields, "inputs")
        outputs = _field(call, call_fields, "outputs")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise ValueError("call inputs and outputs must be mappings")
        records.append({"monotonic_index": position,
                        "schema_accepted": _field(call, call_fields, "schema_accepted") is True,
                        "response_success": _field(call, call_fields, "response_success") is True,
                        "operation_signature": {"operation": operation,
                            "application": meta.get("app"), "callable_name": meta.get("function_name"),
                            "public_required": list(meta.get("required_parameters", [])),
                            "output_slots": list(meta.get("output_slots", []))},
                        "invocation_value_hashes": {str(k): digest(v) for k, v in inputs.items()},
                        "response_output_value_hashes": {str(k): digest(v) for k, v in outputs.items()}})
    return records


class CoProMem:
    """General task-boundary API for dataset runners.

    A runner calls ``prepare`` before dispatch using only public task data,
    then ``learn`` after a complete same-task batch has been scored. For a
    fixed arm, simply omit post-training ``learn`` calls.
    """

    def __init__(self, dataset: Mapping[str, Any], state: Mapping[str, Any] | None = None,
                 admission: Mapping[str, Any] | None = None):
        self.registry = _registry(dataset)
        self.metadata = {"app_descriptions": dataset.get("app_descriptions", {})}
        if "public_operation_intents" in dataset:
            intents = dataset["public_operation_intents"]
            verify_intent_registry(intents, self.registry)
            self.metadata["public_operation_intents"] = intents
        self.admission = dict(admission) if admission is not None else None
        self.episode_fields = dataset.get("episode_fields", {})
        self.call_fields = dataset.get("call_fields", {})
        self.task_fields = dataset.get("task_fields", {})
        self.trials_per_task = dataset.get("trials_per_task")
        if self.trials_per_task is not None and (type(self.trials_per_task) is not int or self.trials_per_task < 1):
            raise ValueError("trials_per_task must be a positive integer")
        self.state = json.loads(json.dumps(state)) if state is not None else {"schemas": {}}
        schemas = self.state.get("schemas")
        if not isinstance(schemas, Mapping) or any(
                not isinstance(schema, Mapping) or schema.get("registry_sha256") != self.registry["registry_sha256"]
                for schema in schemas.values()):
            raise ValueError("loaded bank does not match the frozen public registry")

    def prepare(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """Retrieve from the current frozen pre-task bank."""
        query = derive_task_query(str(_field(task, self.task_fields, "instruction")),
                                  str(_field(task, self.task_fields, "domain")), self.metadata, self.registry)
        guidance, provenance = retrieve(self.state, query, self.registry, self.admission)
        if reproduce_retrieval(self.state, query, self.registry, provenance, self.admission) != guidance:
            raise ValueError("retrieval did not reproduce")
        return {"task_id": str(_field(task, self.task_fields, "task_id")), "query": query,
                "guidance": guidance, "provenance": provenance}

    def prepare_candidate_validation(self, task: Mapping[str, Any], *, schema_id: str,
                                     validation_task_id: str) -> dict[str, Any]:
        task_id = str(_field(task, self.task_fields, "task_id"))
        query = derive_task_query(str(_field(task, self.task_fields, "instruction")),
                                  str(_field(task, self.task_fields, "domain")), self.metadata, self.registry)
        guidance, provenance = candidate_validation_retrieve(
            self.state, query, self.registry, schema_id=schema_id,
            validation_task_id=validation_task_id, current_task_id=task_id)
        if reproduce_retrieval(self.state, query, self.registry, provenance) != guidance:
            raise ValueError("candidate retrieval did not reproduce")
        return {"task_id": task_id, "query": query, "guidance": guidance, "provenance": provenance}

    def learn(self, task_id: str, episodes: list[Mapping[str, Any]], *, expected_trials: int | None = None) -> dict[str, Any]:
        """Validate and atomically commit one complete scored task batch."""
        if not episodes or any(str(_field(row, self.episode_fields, "task_id")) != task_id for row in episodes):
            raise ValueError("learn requires a nonempty same-task batch")
        expected = expected_trials if expected_trials is not None else self.trials_per_task
        if expected is not None and (type(expected) is not int or expected < 1 or len(episodes) != expected):
            raise ValueError("learn requires the complete configured trial batch")
        graphs = []
        audits = []
        for episode in episodes:
            graph, audit = build_semantic_graph(
                _episode_records(episode, self.registry, self.episode_fields, self.call_fields), self.registry)
            success = _field(episode, self.episode_fields, "success")
            if type(success) is not bool:
                raise ValueError("episode success must be a scorer boolean")
            graphs.append((graph, success))
            audits.append(audit)
        proposed = plan([graph for graph, ok in graphs if ok],
                        [graph for graph, ok in graphs if not ok], self.state, audits)
        validation = validate(proposed, self.registry)
        post_state, marker = commit(self.state, proposed, validation)
        self.state = post_state
        return {"task_id": task_id, "validation": validation, "marker": marker}


def run(config_path: str | Path) -> dict[str, Any]:
    """Run a frozen training split and retrieve for later public tasks.

    ``episodes`` must contain scored historical trajectories. ``tasks`` is a
    separate query split. Neither scorer outcomes nor calls enter task queries.
    """
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    dataset = _dataset_config(config_path, config["dataset_config"])
    learn = config.get("learn", True)
    if type(learn) is not bool:
        raise ValueError("learn must be a boolean")
    episodes = _rows(_path(config_path, dataset["episodes"])) if learn else []
    tasks = _rows(_path(config_path, dataset["tasks"]))
    admission = _load(_path(config_path, config["external_admission"])) if "external_admission" in config else None
    loaded = _load(_path(config_path, config["initial_bank"])) if "initial_bank" in config else None
    state = loaded.get("bank", loaded) if loaded is not None else None
    engine = CoProMem(dataset, state=state, admission=admission)
    updates = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[str(_field(episode, engine.episode_fields, "task_id"))].append(episode)
    for task_id, batch in groups.items():
        updates.append(engine.learn(task_id, batch))
    results = [engine.prepare(task) for task in tasks]
    report = {"system": "copromem", "dataset": dataset.get("name", "unnamed"),
              "registry": engine.registry, "bank": engine.state, "updates": updates, "results": results}
    output = _path(config_path, config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
