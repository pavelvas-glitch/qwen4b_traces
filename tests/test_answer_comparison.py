"""Prefix runs preserve the base tasks and seeds across GPU assignments."""
import json
from pathlib import Path
import tempfile
import unittest

from tracegen.plan import load_plan, prepare, prepare_answer_comparison


class AnswerComparisonPlanTests(unittest.TestCase):
    def test_prefix_runs_keep_the_same_first_ten_tasks_and_sample_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("config.plain.json").read_text())
            config.update(task_count=12, model_revision="a" * 40)
            config.pop("generation_mode", None)
            source = root / "source.json"
            source.write_text(json.dumps([
                {"problem": f"Compute {i} + 1.", "answer": i + 1} for i in range(12)
            ]))
            prepare(source, root / "base", config)
            base, base_tasks, base_slots = load_plan(root / "base")
            source.unlink()  # Preparing a comparison only requires the frozen plan.
            task_ids = {task["problem_id"] for task in base_tasks[:10]}
            expected = [(s["problem_id"], s["sample_index"], s["generation_seed"])
                        for s in base_slots if s["problem_id"] in task_ids]
            run_ids = set()
            for gpus in ([0, 1], [2, 3]):
                mode = "reasoning_prefix"
                run = root / f"{mode}-{gpus[0]}"
                prepare_answer_comparison(root / "base", run, mode, gpus=gpus)
                manifest, tasks, slots = load_plan(run)
                self.assertEqual(tasks, base_tasks[:10])
                self.assertEqual(len(slots), 50)
                self.assertEqual([(s["problem_id"], s["sample_index"], s["generation_seed"])
                                  for s in slots], expected)
                self.assertEqual(manifest["prompt_template"], base["prompt_template"])
                self.assertEqual(manifest["config"]["answer_forcing"], mode)
                self.assertEqual([server["gpu"] for server in manifest["config"]["servers"]], gpus)
                run_ids.add(manifest["run_id"])
            self.assertEqual(len(run_ids), 2)
            prepare_answer_comparison(root / "base", root / "full", "reasoning_prefix",
                                      task_count=len(base_tasks))
            _, tasks, slots = load_plan(root / "full")
            self.assertEqual(tasks, base_tasks)
            self.assertEqual([(s["problem_id"], s["sample_index"], s["generation_seed"]) for s in slots],
                             [(s["problem_id"], s["sample_index"], s["generation_seed"]) for s in base_slots])
