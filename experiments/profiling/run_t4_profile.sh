#!/bin/bash
# Profile Qwen3-30B-A3B on a T4 (Kaggle or Colab) for LLMServingSim.
# Same settings as the RTX 4050 profile, except fp16 weights (the T4 has no bf16).
# Output: T4_profile.zip next to this script (about 2 MB).
set -uo pipefail
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "NO GPU: turn on the T4 accelerator"; exit 1; }
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
if ! python -c "import vllm, pandas" 2>/dev/null; then
  echo "Installing vLLM 0.19.0 (a few minutes)..."
  pip install -q vllm==0.19.0 pandas || { echo "INSTALL FAILED: is Internet turned on?"; exit 1; }
fi
HW=T4
OUT=profiler/perf/$HW/Qwen/Qwen3-30B-A3B-Instruct-2507/fp16
ARGS=(Qwen/Qwen3-30B-A3B-Instruct-2507 --hardware $HW --tp 1,2 --dtype float16
      --max-num-batched-tokens 2048 --max-num-seqs 256 --attention-max-kv 16384
      --attention-chunk-factor 4 --attention-kv-factor 4 --measurement-iterations 3
      --skip-skew --gpu-memory-utilization 0.85)
python -m profiler profile "${ARGS[@]}" 2>&1 | tee profile_t4.log
if [ ! -f $OUT/meta.yaml ]; then
  echo "First attempt did not finish; retrying from scratch with Triton attention" | tee -a profile_t4.log
  python -m profiler profile "${ARGS[@]}" --attention-backend TRITON_ATTN --force 2>&1 | tee -a profile_t4.log
fi
cp profile_t4.log profiler/perf/$HW/ 2>/dev/null
if [ -f $OUT/meta.yaml ]; then
  (cd profiler/perf && python -m zipfile -c ../../T4_profile.zip $HW) && echo "DONE: download T4_profile.zip"
else
  echo "FAILED: please send profile_t4.log"
fi
