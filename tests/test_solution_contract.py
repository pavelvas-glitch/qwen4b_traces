"""Solution schedules contain only task, attempt, seed and trace identity."""
import unittest
from pathlib import Path
from tracegen.plan import load_plan
import test_pipeline

class SolutionContractTests(unittest.TestCase):
    setUp = test_pipeline.PipelineTests.setUp
    prepare = test_pipeline.PipelineTests.prepare

    def test_schedule_has_only_solution_fields(self):
        manifest, tasks, slots = load_plan(self.prepare())
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(len(slots), len(tasks) * self.config['samples_per_task'])
        for slot in slots:
            self.assertEqual(set(slot), {'trace_id', 'problem_id', 'sample_index', 'generation_seed'})
