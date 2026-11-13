#!/bin/bash
# Fair placement baselines: Qwen3-30B-A3B, TP=2/EP=2, 30 ShareGPT requests, PCIe 5.0 host link.
cd /app/LLMServingSim
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run pop_ceiling --expert-placement-policy POPULARITY --no-charge-expert-migration --expert-placement-overhead 0
run pop_i50     --expert-placement-policy POPULARITY --expert-placement-interval 50
run pop_i200    --expert-placement-policy POPULARITY --expert-placement-interval 200
run swap_k1     --expert-placement-policy SWAP --expert-placement-max-swaps 1
run swap_k1_i10 --expert-placement-policy SWAP --expert-placement-max-swaps 1 --expert-placement-interval 10
run swap_k4_i10 --expert-placement-policy SWAP --expert-placement-max-swaps 4 --expert-placement-interval 10
echo ALL_DONE
