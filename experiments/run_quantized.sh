#!/bin/bash
# "Quantize all experts" baseline: same setup as run_dsa.sh (reduced = 0.4x).
cd /app/LLMServingSim
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-residency-policy QUANTIZED --expert-reduced-bytes 0.4"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run quant75 --expert-memory-fraction 0.75
run quant50 --expert-memory-fraction 0.5
run quant25 --expert-memory-fraction 0.25
echo ALL_DONE
