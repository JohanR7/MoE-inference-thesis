#!/bin/bash
# FREQUENCY vs LRU residency: Qwen3-30B-A3B, TP=2/EP=2, 30 ShareGPT requests, PCIe 5.0 host link, EVEN placement.
cd /app/LLMServingSim
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-residency-policy FREQUENCY"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run freq75     --expert-memory-fraction 0.75
run freq50     --expert-memory-fraction 0.5
run freq25     --expert-memory-fraction 0.25
run freq50_i10 --expert-memory-fraction 0.5 --expert-placement-interval 10
echo ALL_DONE
