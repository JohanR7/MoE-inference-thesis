#!/bin/bash
# HOBBIT-style baseline under the same conditions as the final DSA results.
cd /app/LLMServingSim
BASE="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-placement-interval 20 --expert-reduced-bytes 0.4 --expert-residency-policy HOBBIT"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $BASE "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
for m in 0.75 0.5 0.25; do run hobbit$m --expert-routing-trace outputs/zipf_qwen30.npz --expert-memory-fraction $m; done
for seed in 1 2; do for m in 0.5 0.25; do run seed${seed}_hobbit$m --expert-routing-trace outputs/zipf_seed$seed.npz --expert-memory-fraction $m; done; done
echo ALL_DONE
