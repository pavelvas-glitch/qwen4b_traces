"""Launch one frozen-model vLLM replica per configured GPU."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import suppress
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .collect import check_answer_forcing_method, environment
from .plan import load_plan, runtime_servers, validate_config, write_json
from .storage import RunLock


def server_command(manifest, server):
    config = manifest["config"]
    check_answer_forcing_method(config)
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model"], "--revision", config["model_revision"],
            "--tokenizer-revision", config["model_revision"],
            "--served-model-name", manifest["served_model"], "--host", "127.0.0.1",
            "--port", str(server["port"]), "--tensor-parallel-size", "1",
            "--dtype", "bfloat16", "--max-model-len", str(config["max_model_len"]),
            "--max-num-seqs", str(server["max_num_seqs"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--enable-prefix-caching", "--generation-config", "vllm", "--seed", str(config["seed"])]


def server_environment(path: Path, server):
    root = Path(path).resolve() / ".cache" / f"gpu{server['gpu']}"
    caches = {"VLLM_CACHE_ROOT": root / "vllm", "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
              "TRITON_CACHE_DIR": root / "triton"}
    for directory in caches.values():
        directory.mkdir(parents=True, exist_ok=True)
    return os.environ | {"CUDA_VISIBLE_DEVICES": str(server["gpu"]),
                         **{name: str(directory) for name, directory in caches.items()}}


def serve(path: Path, *, max_num_seqs=None, gpu_memory_utilization=None, gpus=None):
    with RunLock(Path(path), name=".serve.lock"):
        launch_replicas(Path(path), max_num_seqs=max_num_seqs,
                        gpu_memory_utilization=gpu_memory_utilization, gpus=gpus)


def launch_replicas(path: Path, *, max_num_seqs=None, gpu_memory_utilization=None, gpus=None):
    path = Path(path)
    manifest, _, _ = load_plan(path)
    config = dict(manifest["config"])
    config["servers"] = runtime_servers(config, max_num_seqs=max_num_seqs, gpus=gpus)
    if gpu_memory_utilization is not None:
        config["gpu_memory_utilization"] = gpu_memory_utilization
    validate_config(config)
    manifest = manifest | {"config": config}
    if sys.platform != "linux":
        raise RuntimeError("GPU servers require Linux with CUDA and the gpu dependencies")
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError('install GPU dependencies with pip install -e ".[gpu]"')
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logs = path / "server_logs" / stamp
    logs.mkdir(parents=True)
    hardware = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    write_json(logs / "runtime.json", {"packages": environment(), "hardware": hardware,
                                        "servers": config["servers"],
                                        "gpu_memory_utilization": config["gpu_memory_utilization"]})
    processes, streams = [], []
    previous_handlers = {}

    def stop(signum, frame):
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, stop)
        for server in manifest["config"]["servers"]:
            log = (logs / f"gpu{server['gpu']}.log").open("x")
            streams.append(log)
            command = server_command(manifest, server)
            env = server_environment(path, server)
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        print(f"Started {len(processes)} replicas. Startup logs: {logs}", flush=True)
        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM process exited with {process.returncode}. See {logs}")
            time.sleep(1)
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
