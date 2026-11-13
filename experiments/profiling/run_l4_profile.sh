#!/bin/bash
# Profile Qwen3-30B-A3B on an NVIDIA L4 (Camber) for LLMServingSim.
# Same settings as the RTX 4050 profile: bf16 weights, default attention backend,
# factor-4 attention grid, no skew sweep.
# Output: L4_profile.zip next to this script (about 2 MB). Takes about 1-2 hours.
set -uo pipefail
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "NO GPU: pick an L4 machine"; exit 1; }
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1

# Install vLLM 0.19.0 into its own environment so it does not disturb the machine's packages
if ! python3 -c "import vllm, pandas" 2>/dev/null; then
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv 2>/dev/null || echo "Could not create a venv; installing into the current Python"
  fi
  [ -x .venv/bin/python ] && source .venv/bin/activate
  if ! python3 -c "import vllm, pandas" 2>/dev/null; then
    echo "Installing vLLM 0.19.0 (a few minutes)..."
    python3 -m pip install -q --upgrade pip
    python3 -m pip install -q vllm==0.19.0 pandas || { echo "INSTALL FAILED: check internet access / disk space"; exit 1; }
  fi
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" \
  || { echo "FAILED: PyTorch cannot use the GPU (driver too old for CUDA 12.8?). Send the output above."; exit 1; }

HW=L4
OUT=profiler/perf/$HW/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16
ARGS=(Qwen/Qwen3-30B-A3B-Instruct-2507 --hardware $HW --tp 1,2 --dtype bfloat16
      --max-num-batched-tokens 2048 --max-num-seqs 256 --attention-max-kv 16384
      --attention-chunk-factor 4 --attention-kv-factor 4 --measurement-iterations 3
      --skip-skew --gpu-memory-utilization 0.85)
# Resumes automatically: if the run is interrupted, running this script again
# continues where it stopped.
python3 -m profiler profile "${ARGS[@]}" 2>&1 | tee -a profile_l4.log
if [ ! -f $OUT/meta.yaml ]; then
  echo "First attempt did not finish; retrying from scratch with Triton attention" | tee -a profile_l4.log
  python3 -m profiler profile "${ARGS[@]}" --attention-backend TRITON_ATTN --force 2>&1 | tee -a profile_l4.log
fi
cp profile_l4.log profiler/perf/$HW/ 2>/dev/null
if [ -f $OUT/meta.yaml ]; then
  (cd profiler/perf && python3 -m zipfile -c ../../L4_profile.zip $HW) && echo "DONE: download L4_profile.zip"
else
  echo "FAILED: please send profile_l4.log"
fi
