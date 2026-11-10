"""Validate inputs and freeze a reproducible task and generation schedule."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from string import Formatter


PROMPT_VERSION = "reasoning-v2"
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "plain.txt"
SOURCE_POLICY = "preserve-source-rows-v1"
ANSWER_FORCING_METHODS = ("reasoning_prefix",)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json_text(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict):
    required = {
        "model", "model_revision", "seed", "task_count", "samples_per_task",
        "temperature", "top_p", "top_k", "max_output_tokens", "max_model_len",
        "gpu_memory_utilization", "request_timeout_seconds", "request_attempts", "servers",
    }
    if (not isinstance(config, dict) or not required <= config.keys() or
            config.keys() - required - {"prompt_file", "generation_mode", "selected_problem_ids",
                                        "answer_forcing"}):
        raise ValueError(f"config requires {sorted(required)} and optional prompt_file")
    if config.get("generation_mode", "plain") != "plain":
        raise ValueError("generation_mode must be plain")
    if "answer_forcing" in config and (config["answer_forcing"] not in ANSWER_FORCING_METHODS):
        raise ValueError("answer_forcing requires plain generation and a supported method")
    if "prompt_file" in config and (not isinstance(config["prompt_file"], str) or not config["prompt_file"].strip()):
        raise ValueError("prompt_file must be a nonempty path")
    for name in ("model", "model_revision"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("task_count", "samples_per_task", "max_output_tokens", "max_model_len", "request_attempts"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if "selected_problem_ids" in config:
        ids = config["selected_problem_ids"]
        if (not isinstance(ids, list) or len(ids) != config["task_count"] or
                any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids)):
            raise ValueError("selected_problem_ids must contain task_count distinct string IDs")
    if not config["max_output_tokens"] < config["max_model_len"]:
        raise ValueError("require max_output_tokens < max_model_len")
    for name in ("temperature", "top_p", "gpu_memory_utilization", "request_timeout_seconds"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if config["top_p"] > 1 or config["gpu_memory_utilization"] >= 1:
        raise ValueError("top_p must be <= 1 and gpu_memory_utilization must be < 1")
    if type(config["top_k"]) is not int or (config["top_k"] != -1 and config["top_k"] <= 0):
        raise ValueError("top_k must be -1 or a positive integer")
    servers = config["servers"]
    if not isinstance(servers, list) or not servers:
        raise ValueError("at least one server is required")
    for server in servers:
        if not isinstance(server, dict) or set(server) != {"gpu", "port", "concurrency", "max_num_seqs"}:
            raise ValueError("server needs gpu, port, concurrency and max_num_seqs")
        if any(type(server[name]) is not int or server[name] <= 0
               for name in ("port", "concurrency", "max_num_seqs")):
            raise ValueError("server limits and port must be positive integers")
        if type(server["gpu"]) is not int or server["gpu"] < 0 or server["port"] > 65535:
            raise ValueError("invalid GPU or port")
    for name in ("gpu", "port"):
        if len({server[name] for server in servers}) != len(servers):
            raise ValueError(f"repeated server {name}")


def resolve_revision(model: str, revision: str) -> str:
    if re.fullmatch(r"[a-f0-9]{40}", revision):
        return revision
    from huggingface_hub import model_info
    resolved = model_info(model, revision=revision).sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[a-f0-9]{40}", resolved):
        raise ValueError("could not resolve an immutable model revision")
    return resolved


def runtime_servers(config, *, concurrency=None, max_num_seqs=None, gpus=None):
    overrides = {key: value for key, value in {"concurrency": concurrency, "max_num_seqs": max_num_seqs}.items()
                 if value is not None}
    if any(type(value) is not int or value <= 0 for value in overrides.values()):
        raise ValueError("runtime server limits must be positive integers")
    servers = config["servers"]
    if gpus is not None:
        available = {server["gpu"]: server for server in servers}
        if not gpus or len(set(gpus)) != len(gpus) or any(gpu not in available for gpu in gpus):
            raise ValueError("selected GPUs must be distinct GPUs from the frozen plan")
        servers = [available[gpu] for gpu in gpus]
    return [server | overrides for server in servers]


def read_source(source: Path):
    raw = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("source must be a nonempty JSON array or JSONL file")
    tasks = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source row {index} must be an object")
        problem, answer = row.get("problem"), row.get("answer")
        if (not isinstance(problem, str) or not problem.strip() or type(answer) not in (str, int, float)
                or (isinstance(answer, float) and not math.isfinite(answer))):
            raise ValueError(f"source row {index} needs nonempty problem and scalar answer")
        supplied_id = row.get("id")
        if supplied_id is not None and (type(supplied_id) not in (str, int) or not str(supplied_id).strip()):
            raise ValueError(f"source row {index} has an invalid id")
        tasks.append({"problem_id": f"row-{index}", "problem": problem,
                      "reference_answer": str(answer),
                      "source_id": str(supplied_id) if supplied_id is not None else None})
    return tasks, 0


def validate_prompt(template: str):
    try:
        fields = [(field, spec, conversion) for _, field, spec, conversion in Formatter().parse(template)
                  if field is not None]
    except ValueError as error:
        raise ValueError(f"invalid prompt template: {error}") from error
    expected = {"problem"}
    if ({field for field, _, _ in fields} != expected or
            any(spec or conversion for _, spec, conversion in fields)):
        raise ValueError(f"prompt must use exactly {sorted(expected)}, with literal braces escaped as {{{{ }}}}")


def make_messages(task: dict, prompt_template: str | None = None) -> list[dict]:
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_PATH.read_bytes().decode("utf-8")
    instruction = prompt_template.format(problem=task["problem"])
    return [{"role": "user", "content": instruction}]


def run_identity(config, source_hash, prompt_template, source_policy=None):
    identity = {"config": config, "source_sha256": source_hash,
                "prompt_version": PROMPT_VERSION, "prompt_template": prompt_template}
    if source_policy is not None:
        identity["source_policy"] = source_policy
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare(source: Path, output: Path, config: dict, *, prompt_path: Path | None = None) -> dict:
    validate_config(config)
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    prompt_path = Path(prompt_path or config.get("prompt_file", DEFAULT_PROMPT_PATH))
    prompt_bytes = prompt_path.read_bytes()
    prompt_template = prompt_bytes.decode("utf-8")
    validate_prompt(prompt_template)
    tasks, duplicates = read_source(source)
    if len(tasks) < config["task_count"]:
        raise ValueError(f"only {len(tasks)} source rows, need {config['task_count']}")
    frozen = {key: value for key, value in config.items() if key != "prompt_file"}
    frozen["model_revision"] = resolve_revision(config["model"], config["model_revision"])
    source_hash = digest(source)
    run_id = run_identity(frozen, source_hash, prompt_template, SOURCE_POLICY)
    rng = random.Random(config["seed"])
    selected = rng.sample(tasks, config["task_count"])
    if "selected_problem_ids" in config:
        task_map = {task["problem_id"]: task for task in tasks}
        if any(task_id not in task_map for task_id in config["selected_problem_ids"]):
            raise ValueError("selected_problem_ids contains an unknown source row")
        selected = [task_map[task_id] for task_id in config["selected_problem_ids"]]
    slots = []
    for task in selected:
        for index in range(config["samples_per_task"]):
            identity = f"{config['seed']}/{task['problem_id']}/{index}"
            seed_hash = hashlib.sha256(identity.encode()).hexdigest()
            slot_hash = hashlib.sha256(f"{run_id}/{seed_hash}".encode()).hexdigest()
            slots.append({"trace_id": slot_hash, "problem_id": task["problem_id"],
                          "sample_index": index, "generation_seed": int(seed_hash[:15], 16)})
    rng.shuffle(slots)
    manifest = {"schema_version": 2, "run_id": run_id, "config": frozen,
                "source": {"path": str(source.resolve()), "sha256": source_hash,
                           "duplicates_removed": duplicates, "rows_available": len(tasks),
                           "policy": SOURCE_POLICY},
                "prompt_version": PROMPT_VERSION, "prompt_template": prompt_template,
                "prompt_source": {"path": str(prompt_path.resolve()),
                                  "sha256": hashlib.sha256(prompt_bytes).hexdigest()},
                "expected_tasks": len(selected),
                "expected_traces": len(slots), "served_model": f"qwen4b-{frozen['model_revision'][:12]}"}
    if "answer_forcing" in frozen:
        manifest["served_model"] += f"-{frozen['answer_forcing']}"
    return write_plan(output, manifest, selected, slots)


def prepare_answer_comparison(base_run: Path, output: Path, method: str, *, gpus=None, task_count=10) -> dict:
    base, tasks, slots = load_plan(base_run)
    if (base["config"].get("generation_mode", "plain") != "plain" or
            "answer_forcing" in base["config"]):
        raise ValueError("comparison requires an ordinary plain base run")
    if type(task_count) is not int or not 1 <= task_count <= len(tasks):
        raise ValueError(f"comparison task count must be between 1 and {len(tasks)}")
    selected = tasks[:task_count]
    ids = [task["problem_id"] for task in selected]
    config = base["config"] | {"task_count": len(ids), "selected_problem_ids": ids, "answer_forcing": method}
    if gpus is not None:
        config["servers"] = runtime_servers(config, gpus=gpus)
    validate_config(config)
    selected_ids = set(ids)
    run_id = run_identity(config, base["source"]["sha256"], base["prompt_template"],
                          base["source"].get("policy"))
    selected_slots = [
        slot | {"trace_id": hashlib.sha256(f"{run_id}/{slot['trace_id']}".encode()).hexdigest()}
        for slot in slots if slot["problem_id"] in selected_ids
    ]
    if len(selected_slots) != len(ids) * config["samples_per_task"]:
        raise ValueError("base schedule does not contain the expected attempts for the comparison")
    manifest = base | {
        "config": config, "run_id": run_id, "comparison_base_run_id": base["run_id"],
        "expected_tasks": len(selected), "expected_traces": len(selected_slots),
        "served_model": f"{base['served_model']}-{method}",
    }
    return write_plan(Path(output), manifest, selected, selected_slots)


def write_plan(output, manifest, selected, slots):
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("tasks.jsonl", selected), ("schedule.jsonl", slots)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json_text(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    with (output / "prompt.txt").open("x", encoding="utf-8", newline="") as stream:
        stream.write(manifest["prompt_template"])
        stream.flush()
        os.fsync(stream.fileno())
    manifest["files"] = {name: digest(output / name) for name in ("tasks.jsonl", "schedule.jsonl", "prompt.txt")}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_plan(path: Path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("unsupported manifest or prompt version")
    validate_config(manifest["config"])
    validate_prompt(manifest["prompt_template"])
    if run_identity(manifest["config"], manifest["source"]["sha256"], manifest["prompt_template"],
                    manifest["source"].get("policy")) != manifest.get("run_id"):
        raise ValueError("manifest identity mismatch")
    for name in ("tasks.jsonl", "schedule.jsonl"):
        if digest(path / name) != manifest["files"].get(name):
            raise ValueError(f"checksum mismatch: {name}")
    if "prompt.txt" in manifest["files"]:
        if (digest(path / "prompt.txt") != manifest["files"]["prompt.txt"] or
                (path / "prompt.txt").read_bytes().decode("utf-8") != manifest["prompt_template"]):
            raise ValueError("checksum mismatch: prompt.txt")
    tasks = [json.loads(line) for line in (path / "tasks.jsonl").read_text().splitlines()]
    slots = [json.loads(line) for line in (path / "schedule.jsonl").read_text().splitlines()]
    if (len(tasks) != manifest["expected_tasks"] or len(slots) != manifest["expected_traces"] or
            len({task["problem_id"] for task in tasks}) != len(tasks) or
            len({slot["trace_id"] for slot in slots}) != len(slots)):
        raise ValueError("invalid task or schedule cardinality")
    return manifest, tasks, slots
