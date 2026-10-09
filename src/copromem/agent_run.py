"""Configuration loader for an agent, dataset environment, and CoProMem."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .agent import DEFAULT_SYSTEM_PROMPT, AgentPipeline, ToolAgent
from .pipeline import CoProMem, _dataset_config, _load, _path, _rows
from .providers import OpenRouterProvider, ScriptedProvider


def _load_factory(spec: str, directory: Path) -> Any:
    if ":" not in spec:
        raise ValueError("environment_factory must be module:callable")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("environment_factory must be module:callable")
    sys.path.insert(0, str(directory))
    try:
        local_file = directory / (module_name.replace(".", "/") + ".py")
        if local_file.is_file():
            unique = hashlib.sha256(str(local_file).encode()).hexdigest()[:16]
            module_spec = importlib.util.spec_from_file_location(f"_copromem_env_{unique}", local_file)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load environment module: {local_file}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            value = module
        else:
            value = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
    finally:
        sys.path.pop(0)
    if not callable(value):
        raise TypeError("environment_factory must resolve to a callable")
    return value


def run_agent(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    dataset = _dataset_config(config_path, config["dataset_config"])
    tasks = _rows(_path(config_path, dataset["tasks"]))
    provider_config = config["provider"]
    if provider_config.get("type") == "openrouter":
        provider = OpenRouterProvider(
            str(provider_config["model"]),
            api_key_env=str(provider_config.get("api_key_env", "OPENROUTER_API_KEY")),
            url=str(provider_config.get("url", OpenRouterProvider.DEFAULT_URL)),
            timeout=float(provider_config.get("timeout", 60)),
            temperature=float(provider_config.get("temperature", 0)),
            max_tokens=int(provider_config.get("max_tokens", 1024)))
    elif provider_config.get("type") == "scripted":
        provider = ScriptedProvider(_load(_path(config_path, provider_config["responses"])))
    else:
        raise ValueError("provider type must be openrouter or scripted; inject other providers through Python")
    state = None
    if "initial_bank" in config:
        loaded = _load(_path(config_path, config["initial_bank"]))
        state = loaded.get("bank", loaded)
    admission = _load(_path(config_path, config["external_admission"])) if "external_admission" in config else None
    memory = CoProMem(dataset, state, admission)
    agent = ToolAgent(provider, max_steps=int(config.get("max_steps", 12)),
                      max_tool_calls=int(config.get("max_tool_calls", 20)),
                      system_prompt=str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)))
    factory = _load_factory(str(config["environment_factory"]), config_path.parent)
    pipeline = AgentPipeline(memory, agent, factory)
    if type(config.get("learn", True)) is not bool:
        raise ValueError("learn must be a boolean")
    validation = config.get("candidate_validation")
    if validation is not None:
        if not isinstance(validation, dict) or len(tasks) != 1:
            raise ValueError("candidate validation requires one task and schema/task identity")
    results = [pipeline.run_task(task, trials=int(config.get("trials", 2)),
                                 learn=config.get("learn", True),
                                 validation_schema_id=str(validation["schema_id"]) if validation else None,
                                 validation_task_id=str(validation["task_id"]) if validation else None)
               for task in tasks]
    report = {"system": "copromem", "dataset": dataset.get("name", "unnamed"),
              "registry": memory.registry, "results": results, "bank": memory.state}
    output = _path(config_path, config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
