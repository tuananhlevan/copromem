from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
import pytest

from copromem.pipeline import CoProMem, run
from copromem.retrieval import reproduce_retrieval, retrieve


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy"


def _fixture(tmp_path: Path) -> Path:
    for name in ("run.json", "dataset.json", "episodes.jsonl", "tasks.jsonl"):
        (tmp_path / name).write_bytes((EXAMPLE / name).read_bytes())
    return tmp_path / "run.json"


def test_end_to_end_learning_and_reproducible_query(tmp_path: Path) -> None:
    report = run(_fixture(tmp_path))
    assert report["system"] == "copromem"
    assert report["updates"][0]["marker"]["state"] == "committed"
    result = report["results"][0]
    assert "tools.music.search_artist" in result["guidance"]
    assert "tools.music.follow_artist" in result["guidance"]
    assert reproduce_retrieval(report["bank"], result["query"], report["registry"],
                               result["provenance"]) == result["guidance"]
    saved = (tmp_path / "output" / "result.json").read_text(encoding="utf-8")
    assert "id-A" not in saved and "id-B" not in saved
    assert set(report["bank"]) == {"schemas"}
    assert result["provenance"]["policy_id"] == "copromem-semantic-spine-retrieval"
    assert json.loads(saved) == report


def test_insufficient_support_rejects_without_mutating_bank(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    episodes = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in episodes]
    rows[1]["success"] = False
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = run(config)
    assert report["updates"][0]["marker"]["state"] == "rejected"
    assert report["bank"] == {"schemas": {}}
    assert report["results"][0]["guidance"] == ""


def test_query_split_never_learns_from_its_instruction(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "secret", "domain": "music",
                                 "instruction": "Do something unrelated"}) + "\n", encoding="utf-8")
    report = run(config)
    assert report["results"][0]["guidance"] == ""
    assert report["results"][0]["query"]["canonical_query_operations"] == []


def test_missing_call_attestation_fails_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    del episodes[0]["calls"][0]["response_success"]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")
    with pytest.raises(ValueError, match="missing required field"):
        run(config)


def test_semantic_tie_abstains_and_tampered_provenance_is_rejected(tmp_path: Path) -> None:
    report = run(_fixture(tmp_path))
    state = deepcopy(report["bank"])
    first = next(iter(state["schemas"].values()))
    state["schemas"]["duplicate"] = deepcopy(first)
    query = report["results"][0]["query"]
    guidance, provenance = retrieve(state, query, report["registry"])
    assert guidance == ""
    assert provenance["selection_decision"] == "semantic_tie_abstention"
    bad = deepcopy(provenance)
    bad["guidance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance mismatch"):
        reproduce_retrieval(state, query, report["registry"], bad)


def test_online_task_boundary_uses_pre_task_state(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    task = json.loads((tmp_path / "tasks.jsonl").read_text())
    memory = CoProMem(dataset)
    assert memory.prepare(task)["guidance"] == ""
    assert memory.learn("train-artist", episodes)["marker"]["state"] == "committed"
    assert memory.prepare(task)["guidance"]
    fixed = CoProMem(dataset, memory.state)
    assert fixed.prepare(task) == memory.prepare(task)


def test_offline_fixed_bank_config_reuses_trained_state(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    trained = run(path)
    config = json.loads(path.read_text())
    config.update({"initial_bank": "output/result.json", "learn": False,
                   "output": "output/fixed.json"})
    path.write_text(json.dumps(config))
    fixed = run(path)
    assert fixed["updates"] == []
    assert fixed["bank"] == trained["bank"]
    assert fixed["results"] == trained["results"]


def test_single_trial_success_is_admitted(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first = (tmp_path / "episodes.jsonl").read_text().splitlines()[0]
    (tmp_path / "episodes.jsonl").write_text(first + "\n")
    report = run(config)
    assert report["updates"][0]["marker"]["state"] == "committed"
    schema = next(iter(report["bank"]["schemas"].values()))
    assert schema["support"]["min_successes"] == 1
    assert report["results"][0]["guidance"]


def test_failed_trials_do_not_prune_successful_core(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    successes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    failed = deepcopy(successes[0])
    failed["success"] = False
    memory = CoProMem(dataset)
    update = memory.learn("train-artist", successes + [failed])
    assert update["marker"]["state"] == "committed"
    schema = next(iter(memory.state["schemas"].values()))
    assert schema["required_operations"] == ["tools.music.search_artist", "tools.music.follow_artist"]
    assert schema["support"]["failures"] == 1


def test_identical_structure_is_not_duplicated(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    memory = CoProMem(dataset)
    memory.learn("train-artist", episodes)
    original = deepcopy(memory.state)
    for row in episodes:
        row["calls"][0]["inputs"]["artist_name"] += "-new"
    memory.learn("train-artist", episodes)
    assert memory.state == original


def test_consecutive_terminal_repetitions_keep_distinct_occurrences(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    for row in episodes:
        row["calls"].append(deepcopy(row["calls"][-1]))
    memory = CoProMem(dataset)
    assert memory.learn("train-artist", episodes)["validation"]["passed"]
    schema = next(iter(memory.state["schemas"].values()))
    assert schema["terminal_occurrence_ids"] == ["tools.music.follow_artist#1", "tools.music.follow_artist#2"]
    assert schema["terminal_occurrence_id"] == "repetition-group:2-3"


def test_configured_trial_count_rejects_partial_batch(tmp_path: Path) -> None:
    _fixture(tmp_path)
    dataset = json.loads((tmp_path / "dataset.json").read_text())
    dataset["trials_per_task"] = 3
    episode = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    memory = CoProMem(dataset)
    with pytest.raises(ValueError, match="complete configured trial batch"):
        memory.learn("train-artist", [episode])
