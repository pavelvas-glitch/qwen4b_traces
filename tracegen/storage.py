"""Single-writer durable JSONL and validation against the frozen schedule."""
from __future__ import annotations

from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import uuid

from .plan import json_text, load_plan, make_messages, write_json


class RunLock:
    def __init__(self, root: Path, name=".collect.lock"):
        self.path = Path(root) / name
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f"another process is already using {self.path.name}") from None
        return self

    def __exit__(self, *args):
        self.stream.close()


def compact_record(record):
    compact = {key: value for key, value in record.items()
               if key not in ("input", "output", "error", "timing", "check")}
    compact["output"] = {"finish_reason": (record.get("output") or {}).get("finish_reason")}
    inputs = record.get("input") or {}
    compact["input"] = {"messages": inputs.get("messages"),
                        "max_output_tokens": inputs.get("max_output_tokens")}
    return compact


def validate_record(record):
    required = {"schema_version", "run_id", "session_id", "trace_id", "problem_id", "problem",
                "reference_answer", "sample_index", "generation_seed",
                "model", "server_id", "input", "output", "usage", "check", "status", "error",
                "timing", "request_attempts"}
    if required - record.keys():
        raise ValueError(f"trace is missing required fields: {sorted(required - record.keys())}")
    inputs, usage, output = record["input"], record["usage"], record["output"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("messages"), list):
        raise ValueError("invalid trace input")
    if not isinstance(inputs.get("rendered_prompt"), str) or not isinstance(inputs.get("token_ids"), list):
        raise ValueError("invalid trace input prompt or token_ids")
    if not isinstance(usage, dict) or not isinstance(usage.get("local"), dict):
        raise ValueError("invalid trace usage")
    if usage.get("server") is not None and not isinstance(usage["server"], dict):
        raise ValueError("invalid server usage")
    local = usage["local"]
    local_keys = {"prompt_tokens", "completion_tokens_returned", "reasoning_tokens_generated",
                  "reasoning_tokens", "reasoning_complete"}
    if local_keys - local.keys() or local["prompt_tokens"] != len(inputs["token_ids"]):
        raise ValueError("invalid local usage")
    if record["status"] == "ok":
        if (not isinstance(output, dict) or not isinstance(output.get("text"), str) or
                output.get("finish_reason") not in ("stop", "length") or record["error"] is not None):
            raise ValueError("successful trace needs a complete output object")
    elif not isinstance(record["error"], dict):
        raise ValueError("failed trace needs an error object")


def iter_records(path: Path, *, recover_tail=False, compact=False):
    if not path.exists():
        return
    with path.open("rb") as stream:
        offset = 0
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                if not recover_tail:
                    raise ValueError(f"incomplete final line in {path}")
                archive = path.with_name(f"traces.partial-{uuid.uuid4().hex}.bin")
                with archive.open("xb") as saved:
                    saved.write(line)
                    saved.flush()
                    os.fsync(saved.fileno())
                with path.open("r+b") as repaired:
                    repaired.truncate(offset)
                    repaired.flush()
                    os.fsync(repaired.fileno())
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"invalid JSON at {path}:{number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"expected object at {path}:{number}")
            if compact:
                validate_record(record)
                record = compact_record(record)
            yield record
            offset += len(line)


def read_records(path: Path, *, recover_tail=False, compact=False) -> list[dict]:
    return list(iter_records(path, recover_tail=recover_tail, compact=compact))


def append_record(stream, record: dict):
    stream.write(json_text(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def check_record(manifest, task_map, expected, row):
    trace_id = row.get("trace_id")
    if trace_id not in expected:
        raise ValueError(f"unknown trace_id: {trace_id}")
    slot = expected[trace_id]
    for field, value in slot.items():
        if row.get(field) != value:
            raise ValueError(f"trace {trace_id}: changed {field}")
    task = task_map[slot["problem_id"]]
    if any(row.get(field) != task[field] for field in ("problem", "reference_answer")):
        raise ValueError(f"trace {trace_id}: changed task")
    if (row.get("run_id") != manifest["run_id"] or row.get("schema_version") != 2 or
            row.get("status") not in ("ok", "generation_error", "response_error")):
        raise ValueError(f"trace {trace_id}: invalid run or status")
    config = manifest["config"]
    if row.get("model") != {"id": config["model"], "revision": config["model_revision"]}:
        raise ValueError(f"trace {trace_id}: changed model")
    if (row["input"].get("max_output_tokens") != config["max_output_tokens"] or
            row["input"].get("messages") != make_messages(task, manifest["prompt_template"])):
        raise ValueError(f"trace {trace_id}: changed input")


def check_records(manifest, tasks, slots, records):
    expected = {slot["trace_id"]: slot for slot in slots}
    task_map = {task["problem_id"]: task for task in tasks}
    seen = set()
    for row in records:
        check_record(manifest, task_map, expected, row)
        trace_id = row["trace_id"]
        if trace_id in seen:
            raise ValueError(f"duplicate trace_id: {trace_id}")
        seen.add(trace_id)
    return seen


def result_path(root: Path, trace_id: str) -> Path:
    if not isinstance(trace_id, str) or not re.fullmatch(r"[a-f0-9]{64}", trace_id):
        raise ValueError("invalid trace_id for result JSON")
    return root / "results" / trace_id[:2] / f"{trace_id}.json"


def save_result_json(root: Path, record: dict):
    target = result_path(root, record["trace_id"])
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != record:
            raise ValueError(f"result JSON differs from journal: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, record)


def restore_results(root, manifest, tasks, slots):
    records = read_records(root / "traces.jsonl", recover_tail=True, compact=True)
    seen = check_records(manifest, tasks, slots, records)
    # Backfill legacy journals and compare the two representations one record at a time.
    for row in iter_records(root / "traces.jsonl"):
        save_result_json(root, row)
    expected = {slot["trace_id"]: slot for slot in slots}
    task_map = {task["problem_id"]: task for task in tasks}
    with (root / "traces.jsonl").open("a", encoding="utf-8") as stream:
        for target in (root / "results").glob("*/*.json"):
            if target != result_path(root, target.stem):
                raise ValueError(f"incorrect result JSON path: {target}")
            if target.stem in seen:
                continue
            row = json.loads(target.read_text(encoding="utf-8"))
            validate_record(row)
            check_record(manifest, task_map, expected, row)
            if row["trace_id"] != target.stem:
                raise ValueError(f"result JSON has a different trace_id: {target}")
            append_record(stream, row)
            records.append(compact_record(row))
            seen.add(row["trace_id"])
    return records


def summarize(manifest, tasks, slots, records):
    seen = check_records(manifest, tasks, slots, records)
    counts = Counter(row["status"] for row in records)
    completed = [row["usage"]["local"]["reasoning_tokens"] for row in records
                 if row["usage"]["local"]["reasoning_tokens"] is not None]
    measurements = {
        "truncated": sum((row.get("output") or {}).get("finish_reason") == "length" for row in records),
        "measured_complete_reasoning": len(completed),
        "unknown_or_incomplete_reasoning": len(records) - len(completed),
        "mean_reasoning_tokens": sum(completed) / len(completed) if completed else None,
    }
    missing = [slot["trace_id"] for slot in slots if slot["trace_id"] not in seen]
    return {"run_id": manifest["run_id"], "planned": len(slots), "recorded": len(records),
            "complete": not missing, "ok": counts["ok"],
            "errors": len(records) - counts["ok"], "missing_count": len(missing),
            "missing_first_20": missing[:20], **measurements}


def validate(path: Path):
    path = Path(path)
    with RunLock(path):
        manifest, tasks, slots = load_plan(path)
        records = read_records(path / "traces.jsonl", compact=True)
        summary = summarize(manifest, tasks, slots, records)
        if (path / "results").exists():
            for row in iter_records(path / "traces.jsonl"):
                target = result_path(path, row["trace_id"])
                if not target.exists() or json.loads(target.read_text(encoding="utf-8")) != row:
                    raise ValueError(f"missing or different result JSON: {target}")
            if sum(1 for _ in (path / "results").glob("*/*.json")) != len(records):
                raise ValueError("result JSON count differs from journal, resume collection to reconcile")
        write_json(path / "summary.json", summary)
        return summary
