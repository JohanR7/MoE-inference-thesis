#!/bin/bash
# Expert placement experiment: Qwen3-30B-A3B, TP=2/EP=2, 30 ShareGPT requests, PCIe 5.0 host link.
cd /app/LLMServingSim
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30"
TRACE="--expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run pop        $TRACE --expert-placement-policy POPULARITY
run pop_free   $TRACE --expert-placement-policy POPULARITY --expert-placement-overhead 0
run balanced   --expert-routing-policy BALANCED
run even_pairs $TRACE --expert-placement-policy EVEN --moe-rank-cost pairs
run pop_pairs  $TRACE --expert-placement-policy POPULARITY --moe-rank-cost pairs
echo ALL_DONE
