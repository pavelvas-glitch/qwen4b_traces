"""Replay the exact reasoning tokens and preserve a single output trace."""
import asyncio
import json
import unittest

import httpx

import test_pipeline
from tracegen.collect import collect
from tracegen.plan import load_plan, prepare
from tracegen.storage import validate


class ReasoningPrefixTests(unittest.TestCase):
    setUp = test_pipeline.PipelineTests.setUp

    def test_unavailable_replica_leaves_slots_pending_on_selected_gpu(self):
        run = self.root / "unavailable"
        prepare(self.source, run, self.config)
        manifest, _, _ = load_plan(run)
        ports = []

        async def server(request):
            ports.append(request.url.port)
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": manifest["served_model"]}]})
            raise httpx.ConnectError("replica gone", request=request)

        with self.assertRaises(ExceptionGroup):
            asyncio.run(collect(run, tokenizer=test_pipeline.Tokenizer(),
                                transport=httpx.MockTransport(server), gpus=[1],
                                abort_on_server_failure=True, retry_delay=0))
        self.assertEqual(set(ports), {self.config["servers"][1]["port"]})
        self.assertEqual((run / "traces.jsonl").read_text(), "")
        progress = json.loads((run / "progress.json").read_text())
        self.assertEqual(progress["state"], "failed")
        self.assertEqual(progress["error_traces"], 0)

    def test_replay_preserves_tokens_and_does_not_force_unfinished_reasoning(self):
        config = dict(self.config)
        config.update(generation_mode="plain", samples_per_task=2, task_count=1,
                      prompt_file="tracegen/prompts/plain.txt", answer_forcing="reasoning_prefix",
                      max_output_tokens=64)
        run = self.root / "prefix"
        prepare(self.source, run, config)
        manifest, _, slots = load_plan(run)
        trace = next(slot["trace_id"] for slot in slots if slot["sample_index"] == 0)
        tokenizer = test_pipeline.Tokenizer()
        end_id = tokenizer.encode("</think>")[0]
        requests = {}

        async def server(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": manifest["served_model"]}]})
            body = json.loads(request.content)
            requests[body["request_id"]] = body
            if body["request_id"] == trace:
                text, finish, stop = "reason", "stop", end_id
            elif body["request_id"] == f"{trace}-answer":
                text, finish, stop = " \\boxed{42}", "stop", None
            else:
                text, finish, stop = "x" * 64, "length", None
            tokens = tokenizer.encode(text)
            return httpx.Response(200, json={"model": manifest["served_model"], "choices": [
                {"text": text, "token_ids": tokens, "finish_reason": finish, "stop_reason": stop}
            ], "usage": {"completion_tokens": len(tokens)}})

        result = asyncio.run(collect(run, tokenizer=tokenizer, transport=httpx.MockTransport(server)))
        self.assertEqual(result["ok"], 2)
        self.assertEqual(len(requests), 3)
        first, second = requests[trace], requests[f"{trace}-answer"]
        prefix = tokenizer.encode("reason</think>\n\nFinal Answer:")
        self.assertEqual(first["stop_token_ids"], [end_id])
        self.assertNotIn("stop_token_ids", second)
        self.assertEqual(second["prompt"], first["prompt"] + prefix)
        self.assertEqual(second["max_tokens"], 64 - len(prefix))
        self.assertEqual(second["seed"], first["seed"])
        rows = {row["trace_id"]: row for row in map(json.loads, (run / "traces.jsonl").read_text().splitlines())}
        self.assertEqual(rows[trace]["output"]["token_ids"], prefix + tokenizer.encode(" \\boxed{42}"))
        self.assertEqual(rows[trace]["request_attempts"], 2)
        self.assertTrue(rows[trace]["answer_forcing"]["has_answer"])
        unfinished = next(row for trace_id, row in rows.items() if trace_id != trace)
        self.assertEqual(unfinished["answer_forcing"]["transition"], "no_reasoning_end")
        self.assertTrue(validate(run)["complete"])
