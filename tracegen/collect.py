"""Bounded asynchronous collection from independent local vLLM replicas."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
import uuid

import httpx

from .answer_forcing import ANSWER_PREFIX, thinking_end_id, transition_metrics
from .plan import load_plan, make_messages, runtime_servers, write_json
from .progress import Progress
from .storage import (RunLock, append_record, compact_record, restore_results,
                      save_result_json, summarize)

ANSWER_FORCING_METHOD = "reasoning_prefix"


def check_answer_forcing_method(config):
    method = config.get("answer_forcing")
    if method not in (None, ANSWER_FORCING_METHOD):
        raise ValueError(f"this branch supports {ANSWER_FORCING_METHOD}, but the plan uses {method}")
    return method


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = {}
    for name in ("httpx", "transformers", "huggingface-hub", "vllm", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def make_input(tokenizer, task, config, prompt_template=None):
    messages = make_messages(task, prompt_template)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True)
    tokens = list(tokens)
    if not tokens or len(tokens) + config["max_output_tokens"] > config["max_model_len"]:
        raise ValueError(f"task {task['problem_id']}: prompt and output limit exceed context")
    if config.get("answer_forcing") and thinking_end_id(tokenizer) in tokens:
        raise ValueError(f"task {task['problem_id']}: prompt already contains </think>")
    return {"messages": messages, "rendered_prompt": tokenizer.decode(tokens, skip_special_tokens=False),
            "token_ids": tokens, "max_output_tokens": config["max_output_tokens"]}


def local_usage(tokenizer, output, inputs):
    local = {"prompt_tokens": len(inputs["token_ids"]), "completion_tokens_returned": None,
             "reasoning_tokens_generated": None, "reasoning_tokens": None,
             "reasoning_complete": None}
    tokens = output.get("token_ids")
    if tokens is None:
        return local
    local["completion_tokens_returned"] = len(tokens)
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    if len(opening) != 1 or len(closing) != 1:
        return local
    start = 0
    if opening[0] in tokens:
        start = tokens.index(opening[0]) + 1
    elif not inputs["rendered_prompt"].rstrip().endswith("<think>") and closing[0] not in tokens:
        return local
    closed = closing[0] in tokens[start:]
    end = tokens.index(closing[0], start) if closed else len(tokens)
    length = end - start
    local.update(reasoning_tokens_generated=length, reasoning_complete=closed)
    if closed:
        local.update(reasoning_tokens=length)
    return local


async def send(client, url, body, attempts, delay):
    for attempt in range(attempts):
        try:
            response = await client.post(url + "/v1/completions", json=body)
        except httpx.RequestError:
            if attempt + 1 == attempts:
                raise
        else:
            transient = response.status_code == 429 or response.status_code >= 500
            if not transient or attempt + 1 == attempts:
                return response, attempt + 1
        await asyncio.sleep(delay * 2 ** attempt)
    raise RuntimeError("unreachable retry state")


def parse_completion(raw, served_model):
    if not isinstance(raw, dict):
        raise ValueError("server response must be an object")
    usage = raw.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("usage must be an object or null")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice.get("text"), str) or choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("completion needs text and finish_reason stop or length")
    if raw.get("model") != served_model:
        raise ValueError("response model differs from manifest")
    tokens = choice.get("token_ids")
    if tokens is not None and (not isinstance(tokens, list) or
                               any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError("invalid completion token_ids")
    return {"text": choice["text"], "token_ids": tokens,
            "finish_reason": choice["finish_reason"], "stop_reason": choice.get("stop_reason"),
            "response_id": raw.get("id"), "response_model": raw.get("model")}, usage


async def continue_answer(client, url, body, config, tokenizer, record, retry_delay):
    reasoning = record["output"]
    tokens = reasoning["token_ids"]
    if tokens is None:
        raise ValueError("reasoning replay requires completion token_ids")
    end_id = body["stop_token_ids"][0]
    # Some responses omit the stop token from the returned sequence.
    if end_id not in tokens and reasoning["stop_reason"] == end_id:
        tokens = tokens + [end_id]
        reasoning["token_ids"] = tokens
        reasoning["text"] = tokenizer.decode(tokens, skip_special_tokens=False)
    record["answer_forcing"] = transition_metrics(tokenizer, reasoning, ANSWER_FORCING_METHOD)
    if end_id not in tokens:
        return
    if tokens.index(end_id) != len(tokens) - 1:
        raise ValueError("reasoning request generated tokens past </think>")

    prefix_tokens = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    remaining = body["max_tokens"] - len(tokens) - len(prefix_tokens)
    if remaining <= 0:
        record["answer_forcing"]["transition"] = "no_room_for_answer"
        return
    continuation = dict(body)
    del continuation["stop_token_ids"]
    continuation.update(prompt=body["prompt"] + tokens + prefix_tokens, max_tokens=remaining,
                        request_id=f"{body['request_id']}-answer")
    record["answer_forcing"]["transition"] = "answer_request_failed"
    response, attempts = await send(client, url, continuation, config["request_attempts"], retry_delay)
    record["request_attempts"] += attempts
    response.raise_for_status()
    answer, answer_usage = parse_completion(response.json(), body["model"])
    if answer["token_ids"] is None:
        raise ValueError("answer replay requires completion token_ids")
    combined = tokens + prefix_tokens + answer["token_ids"]
    record["output"] = answer | {"token_ids": combined,
                                 "text": tokenizer.decode(combined, skip_special_tokens=False)}
    record["usage"]["answer_server"] = answer_usage
    record["answer_forcing"] = transition_metrics(tokenizer, record["output"], ANSWER_FORCING_METHOD) | {
        "injected_tokens": len(prefix_tokens),
        "reasoning_response_id": reasoning["response_id"],
        "reasoning_finish_reason": reasoning["finish_reason"],
    }


async def generate(client, tokenizer, manifest, task, slot, server, inputs, session_id, retry_delay,
                   *, abort_on_server_failure=False):
    config = manifest["config"]
    url = f"http://127.0.0.1:{server['port']}"
    body = {"model": manifest["served_model"], "prompt": inputs["token_ids"],
            "max_tokens": config["max_output_tokens"], "temperature": config["temperature"],
            "top_p": config["top_p"], "top_k": config["top_k"], "seed": slot["generation_seed"],
            "n": 1, "stream": False, "skip_special_tokens": False, "return_token_ids": True,
            "request_id": slot["trace_id"]}
    method = check_answer_forcing_method(config)
    if method:
        body["stop_token_ids"] = [thinking_end_id(tokenizer)]
    record = {"schema_version": 2, "run_id": manifest["run_id"], "session_id": session_id,
              **task, **slot, "model": {"id": config["model"], "revision": config["model_revision"]},
              "server_id": f"gpu{server['gpu']}:{server['port']}", "input": inputs,
              "output": None, "usage": {"server": None, "local": {
                  "prompt_tokens": len(inputs["token_ids"]), "completion_tokens_returned": None,
                  "reasoning_tokens_generated": None, "reasoning_tokens": None,
                  "reasoning_complete": None}},
              "check": None, "status": "generation_error", "error": None,
              "timing": {"requested_at": now()}, "request_attempts": None}
    started = time.monotonic()
    response = None
    try:
        response, record["request_attempts"] = await send(
            client, url, body, config["request_attempts"], retry_delay)
        response.raise_for_status()
        record["status"] = "response_error"
        raw = response.json()
        record["output"] = {"raw_response": raw}
        if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
            record["usage"]["server"] = raw["usage"]
        record["output"], _ = parse_completion(raw, manifest["served_model"])
        if method:
            await continue_answer(client, url, body, config, tokenizer, record, retry_delay)
        record["usage"]["local"] = local_usage(tokenizer, record["output"], inputs)
        record["status"] = "ok"
    except (httpx.HTTPError, ValueError) as error:
        if abort_on_server_failure and (isinstance(error, httpx.RequestError) or
                isinstance(error, httpx.HTTPStatusError) and error.response.status_code >= 500):
            raise RuntimeError(f"{record['server_id']} failed after retries: {error}") from error
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, httpx.RequestError):
            record["request_attempts"] = (record["request_attempts"] or 0) + config["request_attempts"]
        if response is not None and record["output"] is None:
            record["output"] = {"raw_text": response.text}
    record["timing"].update(completed_at=now(), elapsed_seconds=time.monotonic() - started)
    return record


async def check_servers(client, manifest, servers):
    for server in servers:
        url = f"http://127.0.0.1:{server['port']}"
        deadline = time.monotonic() + manifest["config"]["request_timeout_seconds"]
        while True:
            try:
                response = await client.get(url + "/health", timeout=5)
                if response.status_code < 500:
                    break
            except httpx.RequestError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"server {url} did not become healthy")
            await asyncio.sleep(1)
        response.raise_for_status()
        response = await client.get(url + "/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        if manifest["served_model"] not in {model.get("id") for model in models}:
            raise ValueError(f"server {url} does not serve {manifest['served_model']}")


async def collect(path: Path, *, tokenizer=None, transport=None, task_limit=None, retry_delay=1.0,
                  concurrency=None, gpus=None, abort_on_server_failure=False):
    path = Path(path)
    if task_limit is not None and (type(task_limit) is not int or task_limit <= 0):
        raise ValueError("task_limit must be a positive integer")
    with RunLock(path):
        manifest, tasks, slots = load_plan(path)
        check_answer_forcing_method(manifest["config"])
        servers = runtime_servers(manifest["config"], concurrency=concurrency, gpus=gpus)
        records = restore_results(path, manifest, tasks, slots)
        seen = {row["trace_id"] for row in records}
        selected = tasks[:task_limit] if task_limit is not None else tasks
        task_map = {task["problem_id"]: task for task in selected}
        pending = [slot for slot in slots if slot["trace_id"] not in seen and slot["problem_id"] in task_map]
        session_id = uuid.uuid4().hex
        progress = Progress(path, manifest, slots, records, set(task_map), session_id)
        progress.publish(log=True)
        try:
            if not pending:
                summary = summarize(manifest, tasks, slots, records)
                write_json(path / "summary.json", summary)
                progress.publish(state="finished", log=True)
                return summary
            config = manifest["config"]
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
            groups = defaultdict(list)
            for slot in pending:
                groups[slot["problem_id"]].append(slot)
            # Validate every selected prompt before issuing the first generation request.
            for group in groups.values():
                slot = group[0]
                make_input(tokenizer, task_map[slot["problem_id"]], config,
                           manifest["prompt_template"])
            queue = deque(groups.values())
            total_concurrency = sum(server["concurrency"] for server in servers)
            limits = httpx.Limits(max_connections=total_concurrency + 8, max_keepalive_connections=total_concurrency)
            with (path / "sessions.jsonl").open("a", encoding="utf-8") as sessions:
                append_record(sessions, {"session_id": session_id, "started_at": now(),
                                         "packages": environment(), "servers": servers,
                                         "task_limit": task_limit, "pending": len(pending),
                                         "abort_on_server_failure": abort_on_server_failure})
            async with httpx.AsyncClient(timeout=config["request_timeout_seconds"], limits=limits,
                                         transport=transport) as client:
                async with asyncio.TaskGroup() as monitored:
                    heartbeat = monitored.create_task(progress.heartbeat())
                    try:
                        progress.publish(state="waiting_for_servers", log=True)
                        await check_servers(client, manifest, servers)
                        progress.publish(state="running", log=True)
                        with (path / "traces.jsonl").open("a", encoding="utf-8") as stream:
                            async def sample_worker(server, server_queue):
                                server_id = f"gpu{server['gpu']}:{server['port']}"
                                while True:
                                    if not server_queue:
                                        try:
                                            group = queue.popleft()
                                        except IndexError:
                                            return
                                        first = group[0]
                                        task = task_map[first["problem_id"]]
                                        inputs = make_input(tokenizer, task, config,
                                                            manifest["prompt_template"])
                                        for slot in group:
                                            server_queue.append((task, slot, inputs))
                                    task, slot, inputs = server_queue.popleft()
                                    progress.active[server_id] += 1
                                    progress.publish()
                                    try:
                                        row = await generate(client, tokenizer, manifest, task, slot, server,
                                                             inputs, session_id, retry_delay,
                                                             abort_on_server_failure=abort_on_server_failure)
                                    finally:
                                        progress.active[server_id] -= 1
                                    save_result_json(path, row)
                                    append_record(stream, row)
                                    records.append(compact_record(row))
                                    progress.saved(row)

                            async with asyncio.TaskGroup() as workers:
                                for server in servers:
                                    server_queue = deque()
                                    for _ in range(server["concurrency"]):
                                        workers.create_task(sample_worker(server, server_queue))
                    finally:
                        heartbeat.cancel()
            summary = summarize(manifest, tasks, slots, records)
            write_json(path / "summary.json", summary)
            progress.publish(state="finished", log=True)
            return summary
        except asyncio.CancelledError:
            progress.publish(state="interrupted", log=True)
            raise
        except Exception:
            progress.publish(state="failed", log=True)
            raise
        finally:
            progress.bar.close()
