#!/usr/bin/env bash
# Install the Linux GPU environment inside the project directory.
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer requires Linux x86_64 with NVIDIA GPUs.' >&2
  exit 1
fi

QWEN_TRACES_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$QWEN_TRACES_ROOT"
export UV_CACHE_DIR="$QWEN_TRACES_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$QWEN_TRACES_ROOT/.tools/python"
mkdir -p .tools .cache

if [[ ! -x .tools/uv ]]; then
  curl -fLsS https://astral.sh/uv/install.sh -o .tools/uv-installer.sh
  UV_UNMANAGED_INSTALL="$QWEN_TRACES_ROOT/.tools" sh .tools/uv-installer.sh
fi

.tools/uv python install 3.12
if [[ -d .venv ]]; then
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv must use Python 3.12"'
else
  .tools/uv venv --python 3.12 --seed .venv
fi

.tools/uv pip install --python .venv/bin/python --torch-backend cu128 -e '.[gpu]'
.tools/uv pip check --python .venv/bin/python

.venv/bin/python - <<'PY'
import platform
import torch
import vllm
import vllm._C

assert torch.cuda.is_available(), "CUDA is unavailable in the project environment"
assert torch.cuda.device_count() >= 4, "Expected at least four visible GPUs"
print(f"Python {platform.python_version()}; torch {torch.__version__}; CUDA {torch.version.cuda}; vLLM {vllm.__version__}")
for index in range(4):
    values = torch.ones((16, 16), device=f"cuda:{index}")
    assert (values @ values).sum().item() == 4096
    torch.cuda.synchronize(index)
    print(f"GPU {index}: {torch.cuda.get_device_name(index)} — OK")
PY

.venv/bin/python -m unittest discover -s tests -q
printf '\nEnvironment ready. Activate it with: source .venv/bin/activate\n'
