"""Commands for preparing, serving, collecting and validating raw traces."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="freeze tasks, attempts, seeds and model revision")
    prep.add_argument("--source", type=Path, required=True, help="local DeepScaleR JSON or JSONL")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--prompt", type=Path, help="override the prompt template file")
    prep.add_argument("--tasks", type=int, help="override task_count before freezing the run")
    comparison = commands.add_parser("prepare-comparison", help="freeze a subset of a plain run")
    comparison.add_argument("--base-run", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--method", choices=("reasoning_prefix",), required=True)
    comparison.add_argument("--gpus", type=int, nargs="+", help="select GPU indices from the base plan")
    comparison.add_argument("--tasks", type=int, default=10, help="number of tasks to keep (default: 10)")
    for name in ("serve", "collect", "validate"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--run", type=Path, required=True)
        if name in ("serve", "collect"):
            subparser.add_argument("--gpus", type=int, nargs="+", help="select GPU indices from the frozen plan")
        if name == "collect":
            subparser.add_argument("--task-limit", type=int, help="collect only the first N selected tasks")
            subparser.add_argument("--concurrency", type=int, help="override concurrent requests per server")
            subparser.add_argument("--abort-on-server-failure", action="store_true",
                                   help="stop on exhausted network or 5xx retries and leave slots pending")
        if name == "serve":
            subparser.add_argument("--max-num-seqs", type=int, help="override vLLM sequences per GPU")
            subparser.add_argument("--gpu-memory-utilization", type=float, help="override vLLM GPU memory fraction")
        if name == "validate":
            subparser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .plan import prepare
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if args.tasks is not None:
                config["task_count"] = args.tasks
            prompt_path = args.prompt
            if prompt_path is None and "prompt_file" in config:
                prompt_path = Path(config["prompt_file"])
                if not prompt_path.is_absolute():
                    prompt_path = args.config.parent / prompt_path
            result = prepare(args.source, args.out, config, prompt_path=prompt_path)
            print(json.dumps({"run_id": result["run_id"], "tasks": result["expected_tasks"],
                              "traces": result["expected_traces"]}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "prepare-comparison":
            from .plan import prepare_answer_comparison
            result = prepare_answer_comparison(args.base_run, args.out, args.method,
                                               gpus=args.gpus, task_count=args.tasks)
            print(json.dumps({"run_id": result["run_id"], "tasks": result["expected_tasks"],
                              "traces": result["expected_traces"]}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "serve":
            from .serve import serve
            serve(args.run, max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_memory_utilization, gpus=args.gpus)
            return 0
        if args.command == "collect":
            from .collect import collect
            result = asyncio.run(collect(args.run, task_limit=args.task_limit, concurrency=args.concurrency,
                                         gpus=args.gpus, abort_on_server_failure=args.abort_on_server_failure))
            allow_incomplete = args.task_limit is not None
        else:
            from .storage import validate
            result = validate(args.run)
            allow_incomplete = args.allow_incomplete
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result["errors"] > 0 or (not allow_incomplete and not result["complete"]))
    except KeyboardInterrupt:
        print("Stopped. Resume collection with the same --run.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
