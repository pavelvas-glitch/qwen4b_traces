"""Sampling keeps five independent solutions per source task."""
import unittest
import asyncio
import httpx
import json
from collections import Counter
import test_pipeline
from test_pipeline import Tokenizer
from tracegen.plan import prepare, load_plan, make_messages
from tracegen.collect import local_usage

class PlainTests(unittest.TestCase):
    setUp = test_pipeline.PipelineTests.setUp
    def test_plain_schedule_and_usage(self):
        config = dict(self.config)
        config.update(generation_mode='plain', samples_per_task=5,
                      prompt_file=str(self.root / 'plain.txt'))
        (self.root / 'plain.txt').write_text('{problem}\nPut the final answer in \\boxed{{...}}.')
        path = self.root / 'plain'
        prepare(self.source, path, config)
        manifest, tasks, slots = load_plan(path)
        self.assertEqual(Counter(s['problem_id'] for s in slots), {t['problem_id']: 5 for t in tasks})
        self.assertEqual(len({s['generation_seed'] for s in slots}), 10)
        message = make_messages(tasks[0], manifest['prompt_template'])[0]['content']
        self.assertIn(tasks[0]['problem'], message)
        tokens = Tokenizer().encode('reason</think>42')
        usage = local_usage(Tokenizer(), {'token_ids': tokens},
                            {'token_ids': [1], 'rendered_prompt': '<think>'})
        self.assertTrue(usage['reasoning_complete'])
        self.assertEqual(set(usage), {'prompt_tokens', 'completion_tokens_returned',
                                     'reasoning_tokens_generated', 'reasoning_tokens', 'reasoning_complete'})

        from tracegen.collect import collect
        from tracegen.storage import validate
        async def server(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={})
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': manifest['served_model']}]})
            body = json.loads(request.content)
            return httpx.Response(200, json={'model': manifest['served_model'], 'choices': [
                {'text': 'reason</think>42', 'token_ids': tokens, 'finish_reason': 'stop'}]})
        result = asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
        self.assertTrue(result['complete'])
        self.assertEqual(result['ok'], 10)
        self.assertEqual(result['recorded'], 10)
        self.assertTrue(validate(path)['complete'])
        again = asyncio.run(collect(path, tokenizer=Tokenizer(), transport=httpx.MockTransport(server)))
        self.assertEqual(again['recorded'], 10)
