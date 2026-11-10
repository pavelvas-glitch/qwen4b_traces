"""Contract regressions at the file and HTTP boundaries, without model weights."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

import httpx


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    """Deterministic tokenizer double for the external Transformers boundary."""
    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith("<think>"):
                result.append(100000)
                text = text[7:]
            elif text.startswith("</think>"):
                result.append(100001)
                text = text[8:]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, tokens, skip_special_tokens=False):
        special = {100000: "<think>", 100001: "</think>"}
        return "".join(special[token] if token in special else chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"] + "\n<assistant>\n<think>\n")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "config.json").read_text())
        self.config["prompt_file"] = str(ROOT / "tracegen/prompts/plain.txt")
        self.config.update(model_revision="a" * 40, task_count=2, request_attempts=2)
        self.config["servers"] = [
            {"gpu": 0, "port": 8000, "concurrency": 2, "max_num_seqs": 2},
            {"gpu": 1, "port": 8001, "concurrency": 2, "max_num_seqs": 2},
        ]
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps([
            {"id": "a", "problem": "What is 6 times 7?", "answer": "42"},
            {"id": "b", "problem": "What is 40 plus 2?", "answer": "42"},
            {"id": "c", "problem": "What is 84 divided by 2?", "answer": "42"},
        ]))

    def prepare(self, name="run"):
        from tracegen.plan import prepare
        path = self.root / name
        prepare(self.source, path, self.config)
        return path

    def test_schedule_has_five_stable_samples_per_task(self):
        first, second = self.prepare(), self.prepare("second")
        for name in ("tasks.jsonl", "schedule.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        slots = [json.loads(line) for line in (first / "schedule.jsonl").read_text().splitlines()]
        self.assertEqual(len(slots), 10)
        self.assertEqual(len({slot["trace_id"] for slot in slots}), 10)
        for problem_id in {slot["problem_id"] for slot in slots}:
            group = [slot for slot in slots if slot["problem_id"] == problem_id]
            self.assertEqual(len(group), 5)
            self.assertEqual({slot["sample_index"] for slot in group}, {0, 1, 2, 3, 4})
            self.assertEqual(len({slot["generation_seed"] for slot in group}), 5)

    def test_duplicate_source_rows_keep_separate_ids_and_original_answers(self):
        from tracegen.plan import load_plan
        self.source.write_text(json.dumps([
            {"id": "repeat", "problem": "x + y?", "answer": "1"},
            {"id": "repeat", "problem": "x + y?", "answer": "2"},
            {"id": "repeat", "problem": "Another question?", "answer": "3"},
            {"problem": "x + y?", "answer": "1"},
        ]))
        self.config["task_count"] = 4
        manifest, tasks, slots = load_plan(self.prepare())
        self.assertEqual({task["problem_id"] for task in tasks}, {"row-0", "row-1", "row-2", "row-3"})
        by_id = {task["problem_id"]: task for task in tasks}
        self.assertEqual(by_id["row-0"]["reference_answer"], "1")
        self.assertEqual(by_id["row-1"]["reference_answer"], "2")
        self.assertEqual(by_id["row-2"]["source_id"], "repeat")
        self.assertEqual(len(slots), 20)
        self.assertEqual(len({slot["trace_id"] for slot in slots}), 20)
        self.assertEqual(manifest["source"]["duplicates_removed"], 0)

    def test_empty_reference_is_preserved_for_raw_trace_collection(self):
        from tracegen.plan import read_source
        self.source.write_text(json.dumps([{"problem": "Question?", "answer": ""}]))
        tasks, removed = read_source(self.source)
        self.assertEqual(tasks[0]["reference_answer"], "")
        self.assertEqual(removed, 0)

    def test_not_enough_tasks_and_invalid_config_are_rejected(self):
        from tracegen.plan import prepare
        for update in ({"task_count": 4}, {"task_count": True}, {"max_output_tokens": 32768},
                       {"temperature": float("nan")}, {"samples_per_task": 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                prepare(self.source, self.root / "bad", self.config | update)

    def test_plan_mutation_is_detected(self):
        from tracegen.plan import load_plan
        path = self.prepare()
        with (path / "schedule.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_plan(path)

    def test_changed_model_parameters_change_run_and_trace_ids(self):
        first = self.prepare()
        self.config["temperature"] = 0.8
        second = self.prepare("second")
        first_ids = {json.loads(line)["trace_id"] for line in (first / "schedule.jsonl").read_text().splitlines()}
        second_ids = {json.loads(line)["trace_id"] for line in (second / "schedule.jsonl").read_text().splitlines()}
        self.assertFalse(first_ids & second_ids)

    def test_editing_frozen_manifest_is_rejected(self):
        from tracegen.plan import load_plan
        path = self.prepare()
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["config"]["temperature"] = 0.9
        (path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity"):
            load_plan(path)

    def test_prompt_file_is_loaded_and_frozen_for_resume(self):
        from tracegen.plan import load_plan, make_messages
        prompt = self.root / "prompt.txt"
        prompt.write_text("CUSTOM {problem}\nSolve carefully.\\boxed{{}}")
        self.config["prompt_file"] = str(prompt)
        path = self.prepare()
        first, tasks, slots = load_plan(path)
        self.assertEqual((path / "prompt.txt").read_text(), prompt.read_text())
        prompt.write_text("UPDATED {problem}\nSolve carefully.")
        frozen, _, _ = load_plan(path)
        messages = make_messages(tasks[0], frozen["prompt_template"])
        self.assertTrue(messages[0]["content"].startswith("CUSTOM "))
        second, _, _ = load_plan(self.prepare("second"))
        self.assertNotEqual(first["run_id"], second["run_id"])
        (path / "prompt.txt").write_text("edited frozen snapshot")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_plan(path)

    def test_invalid_prompt_placeholders_fail_before_run_creation(self):
        from tracegen.plan import prepare
        prompt = self.root / "prompt.txt"
        self.config["prompt_file"] = str(prompt)
        for template in ("{answer}", "{problem} {answer}", "{problem"):
            with self.subTest(template=template):
                prompt.write_text(template)
                with self.assertRaisesRegex(ValueError, "prompt"):
                    prepare(self.source, self.root / "bad", self.config)
                self.assertFalse((self.root / "bad").exists())

    def test_legacy_manifest_without_external_prompt_still_loads(self):
        from tracegen.plan import load_plan, run_identity, write_json
        path = self.prepare()
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["files"].pop("prompt.txt", None)
        manifest.pop("prompt_source", None)
        manifest["source"].pop("policy", None)
        (path / "prompt.txt").unlink(missing_ok=True)
        manifest["run_id"] = run_identity(manifest["config"], manifest["source"]["sha256"],
                                           manifest["prompt_template"])
        write_json(path / "manifest.json", manifest)
        self.assertEqual(load_plan(path)[0]["run_id"], manifest["run_id"])

    def test_missing_token_ids_and_unclosed_reasoning_measurements(self):
        from tracegen.collect import local_usage
        inputs = {"token_ids": [1], "rendered_prompt": "<think>\n"}
        missing = local_usage(Tokenizer(), {"token_ids": None}, inputs)
        unclosed = local_usage(Tokenizer(), {"token_ids": [65, 66]}, inputs)
        self.assertIsNone(missing["completion_tokens_returned"])
        self.assertEqual(unclosed["reasoning_tokens_generated"], 2)
        self.assertFalse(unclosed["reasoning_complete"])

    def test_server_command_pins_model_revision_and_gpu_batch_limit(self):
        from tracegen.plan import load_plan
        from tracegen.serve import server_command
        manifest, _, _ = load_plan(self.prepare())
        command = server_command(manifest, self.config["servers"][0])
        self.assertEqual(command[command.index("--revision") + 1], "a" * 40)
        self.assertEqual(command[command.index("--tokenizer-revision") + 1], "a" * 40)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")

    def test_replicas_have_separate_compiler_cache_directories(self):
        from tracegen.serve import server_environment
        run = self.prepare()
        first = server_environment(run, self.config["servers"][0])
        second = server_environment(run, self.config["servers"][1])
        self.assertEqual(first["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(second["CUDA_VISIBLE_DEVICES"], "1")
        for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            self.assertNotEqual(first[name], second[name])
            self.assertTrue(Path(first[name]).is_dir())
            self.assertTrue(Path(second[name]).is_relative_to(run.resolve()))

    def test_runtime_server_limits_preserve_frozen_configuration(self):
        from tracegen.plan import runtime_servers
        original = copy.deepcopy(self.config)
        servers = runtime_servers(self.config, concurrency=32, max_num_seqs=32)
        self.assertEqual(self.config, original)
        self.assertTrue(all(server["concurrency"] == 32 and server["max_num_seqs"] == 32 for server in servers))
        self.assertEqual([server["port"] for server in servers], [8000, 8001])
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime_servers(self.config, concurrency=value)

    def test_prepared_contract_works_through_cli(self):
        import subprocess
        import sys
        config_path = self.root / "config.json"
        relative_prompt = self.root / "relative_prompt.txt"
        relative_prompt.write_text("CLI {problem}\nSolve carefully.")
        config_path.write_text(json.dumps(self.config | {"prompt_file": relative_prompt.name}))
        run = self.root / "cli"
        command = [sys.executable, "-m", "tracegen", "prepare", "--source", str(self.source),
                   "--config", str(config_path), "--out", str(run)]
        prepared = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(json.loads(prepared.stdout)["traces"], 10)
        self.assertEqual((run / "prompt.txt").read_text(), relative_prompt.read_text())
        override = self.root / "override.txt"
        override.write_text("OVERRIDE {problem}\nSolve carefully.")
        override_command = command[:-1] + [str(self.root / "override_run"), "--prompt", str(override)]
        overridden = subprocess.run(override_command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(overridden.returncode, 0, overridden.stderr)
        self.assertEqual((self.root / "override_run/prompt.txt").read_text(), override.read_text())
        checked = subprocess.run([sys.executable, "-m", "tracegen", "validate", "--run", str(run)],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 1)
        self.assertEqual(json.loads(checked.stdout)["missing_count"], 10)

    def test_collect_preserves_raw_output_usage_and_resumes_without_duplicates(self):
        from tracegen.collect import collect
        from tracegen.storage import validate
        prompt = self.root / "custom.txt"
        prompt.write_text("CUSTOM {problem}\nSolve carefully. Return \\boxed{{...}}.")
        self.config["prompt_file"] = str(prompt)
        path = self.prepare()
        prompt.unlink()
        calls = []
        active, peaks = {}, {}
        async def server(request):
            port = request.url.port
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True})
            body = json.loads(request.content)
            calls.append((port, body))
            active[port] = active.get(port, 0) + 1
            peaks[port] = max(peaks.get(port, 0), active[port])
            await asyncio.sleep(0)
            active[port] -= 1
            text = "calculation</think>\\boxed{42}"
            return httpx.Response(200, json={
                "id": f"response-{body['seed']}", "model": body["model"],
                "choices": [{"text": text, "token_ids": Tokenizer().encode(text),
                             "finish_reason": "stop", "stop_reason": 151645}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 25,
                          "total_tokens": len(body["prompt"]) + 25,
                          "provider_extra": {"cache": 17}},
            })
        transport = httpx.MockTransport(server)
        frozen_manifest = (path / "manifest.json").read_bytes()
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=transport, task_limit=1, concurrency=3))
        self.assertEqual(len(calls), 5)
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=transport, concurrency=4))
        self.assertEqual(len(calls), 10)
        missing_json = next((path / "results").glob("*/*.json"))
        missing_json.unlink()
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=transport))
        self.assertEqual(len(calls), 10)
        self.assertTrue(missing_json.exists())
        rows = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
        self.assertEqual(len({row["trace_id"] for row in rows}), 10)
        self.assertTrue(validate(path)["complete"])
        for row in rows:
            self.assertTrue(row["input"]["messages"][0]["content"].startswith("CUSTOM "))
            self.assertEqual(row["output"]["text"], "calculation</think>\\boxed{42}")
            self.assertEqual(row["usage"]["server"]["provider_extra"], {"cache": 17})
            self.assertEqual(row["usage"]["local"]["reasoning_tokens"], 11)
            self.assertIsNone(row["check"])
            self.assertEqual(row["input"]["max_output_tokens"], 16384)
        self.assertTrue(all(peak <= 4 for peak in peaks.values()))
        self.assertGreaterEqual(max(peaks.values()), 3)
        self.assertEqual((path / "manifest.json").read_bytes(), frozen_manifest)
        sessions = [json.loads(line) for line in (path / "sessions.jsonl").read_text().splitlines()]
        self.assertTrue(all(server["concurrency"] == 3 for server in sessions[0]["servers"]))
        self.assertTrue(all(server["concurrency"] == 4 for server in sessions[1]["servers"]))
        groups = {}
        for row in rows:
            groups.setdefault(row["problem_id"], set()).add(row["server_id"])
        self.assertTrue(all(len(servers) == 1 for servers in groups.values()))
        broken = copy.deepcopy(rows[0])
        broken["output"] = None
        (path / "traces.jsonl").write_text(json.dumps(broken) + "\n")
        with self.assertRaisesRegex(ValueError, "output"):
            validate(path)
        broken = copy.deepcopy(rows[0])
        broken["model"]["revision"] = "b" * 40
        (path / "traces.jsonl").write_text(json.dumps(broken) + "\n")
        with self.assertRaisesRegex(ValueError, "model"):
            validate(path)

    def test_transient_retry_keeps_seed_and_terminal_errors_are_saved(self):
        from tracegen.collect import collect
        from tracegen.storage import validate
        path = self.prepare()
        calls = {}
        async def server(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
            body = json.loads(request.content)
            key = body["seed"]
            calls.setdefault(key, []).append(copy.deepcopy(body))
            if len(calls[key]) == 1:
                return httpx.Response(503, text="busy")
            return httpx.Response(400, text="invalid request")
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server), retry_delay=0))
        rows = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["status"] == "generation_error" for row in rows))
        self.assertTrue(all(values[0] == values[1] for values in calls.values()))
        summary = validate(path)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["errors"], 10)

    def test_interrupted_tail_is_archived_and_only_missing_slot_is_regenerated(self):
        from tracegen.storage import read_records, RunLock
        path = self.prepare()
        target = path / "traces.jsonl"
        target.write_bytes(b'{"trace_id":"done"}\n{"broken":')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_records(target)
        records = read_records(target, recover_tail=True)
        self.assertEqual(records, [{"trace_id": "done"}])
        self.assertEqual(target.read_bytes(), b'{"trace_id":"done"}\n')
        self.assertEqual(len(list(path.glob("traces.partial-*.bin"))), 1)
        with RunLock(path):
            with self.assertRaisesRegex(RuntimeError, "already"):
                with RunLock(path):
                    pass

    def test_response_error_keeps_server_body_and_unknown_usage_stays_null(self):
        from tracegen.collect import collect
        path = self.prepare()
        async def server(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
            return httpx.Response(200, json={"unexpected": "preserve me"})
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
        row = json.loads((path / "traces.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["status"], "response_error")
        self.assertEqual(row["output"]["raw_response"], {"unexpected": "preserve me"})
        self.assertIsNone(row["usage"]["server"])

    def test_memory_projection_omits_large_raw_payloads(self):
        from tracegen.storage import compact_record
        row = {"trace_id": "x", "output": {"text": "a" * 100000, "token_ids": [1] * 100000,
                                             "finish_reason": "length"},
               "input": {"token_ids": [2] * 100000}, "usage": {"server": None, "local": {}}}
        compact = compact_record(row)
        self.assertEqual(compact["output"], {"finish_reason": "length"})
        self.assertNotIn("token_ids", compact["input"])
        self.assertEqual(len(row["output"]["token_ids"]), 100000)

    def test_free_slot_starts_next_group_while_previous_group_has_a_slow_answer(self):
        from tracegen.collect import collect
        self.config["task_count"] = 2
        self.config["servers"] = [{"gpu": 0, "port": 8000, "concurrency": 5, "max_num_seqs": 5}]
        path = self.prepare()

        async def exercise():
            release, next_group = asyncio.Event(), asyncio.Event()
            calls = 0
            async def server(request):
                nonlocal calls
                if request.url.path == "/health":
                    return httpx.Response(200, json={})
                if request.url.path == "/v1/models":
                    return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
                calls += 1
                if calls == 1:
                    await release.wait()
                if calls > 5:
                    next_group.set()
                return httpx.Response(200, json={"model": "qwen4b-aaaaaaaaaaaa", "choices": [
                    {"text": "done", "finish_reason": "stop"}]})
            task = asyncio.create_task(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
            try:
                await asyncio.wait_for(next_group.wait(), timeout=0.5)
                advanced = True
                progress = json.loads((path / "progress.json").read_text())
                self.assertGreater(progress["saved_traces"], 0)
                self.assertLess(progress["saved_traces"], 10)
                self.assertEqual(progress["state"], "running")
                saved = list((path / "results").glob("*/*.json"))
                self.assertGreater(len(saved), 0)
                trace = json.loads(saved[0].read_text())
                self.assertEqual(trace["output"]["text"], "done")
                self.assertEqual(trace["problem"], next(task["problem"] for task in map(json.loads, (path / "tasks.jsonl").read_text().splitlines()) if task["problem_id"] == trace["problem_id"]))
            except TimeoutError:
                advanced = False
            finally:
                release.set()
                await task
            self.assertTrue(advanced, "free slots must not wait for the slowest answer in a group")
        asyncio.run(exercise())

    def test_atomic_json_is_recovered_if_journal_append_was_interrupted(self):
        from unittest.mock import patch
        from tracegen.collect import collect
        from tracegen.storage import append_record
        self.config["task_count"] = 1
        self.config["servers"] = [{"gpu": 0, "port": 8000, "concurrency": 1, "max_num_seqs": 1}]
        path = self.prepare()
        calls = []
        async def server(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
            calls.append(json.loads(request.content)["seed"])
            return httpx.Response(200, json={"model": "qwen4b-aaaaaaaaaaaa", "choices": [
                {"text": "saved output", "finish_reason": "stop"}]})
        def interrupted(stream, row):
            if str(stream.name).endswith("traces.jsonl"):
                raise OSError("simulated interrupted append")
            append_record(stream, row)
        with patch("tracegen.collect.append_record", side_effect=interrupted):
            with self.assertRaises(ExceptionGroup):
                asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
        self.assertEqual(len(list((path / "results").glob("*/*.json"))), 1)
        asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(set(calls)), 5)
        self.assertEqual(len((path / "traces.jsonl").read_text().splitlines()), 5)
        progress = json.loads((path / "progress.json").read_text())
        self.assertEqual(progress["saved_traces"], 5)
        self.assertEqual(progress["completed_tasks"], 1)
        self.assertEqual(progress["active_requests"], 0)
        self.assertEqual(progress["state"], "finished")
        self.assertIn("finished", (path / "progress.log").read_text())

    def test_cancelled_collection_publishes_interrupted_state(self):
        from tracegen.collect import collect
        self.config["task_count"] = 1
        self.config["servers"] = [{"gpu": 0, "port": 8000, "concurrency": 1, "max_num_seqs": 1}]
        path = self.prepare()
        async def exercise():
            entered = asyncio.Event()
            async def server(request):
                if request.url.path == "/health":
                    return httpx.Response(200, json={})
                if request.url.path == "/v1/models":
                    return httpx.Response(200, json={"data": [{"id": "qwen4b-aaaaaaaaaaaa"}]})
                entered.set()
                await asyncio.Event().wait()
            task = asyncio.create_task(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            progress = json.loads((path / "progress.json").read_text())
            self.assertEqual(progress["state"], "interrupted")
            self.assertEqual(progress["active_requests"], 0)
            self.assertEqual(progress["saved_traces"], 0)
            self.assertIn("interrupted", (path / "progress.log").read_text())
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
