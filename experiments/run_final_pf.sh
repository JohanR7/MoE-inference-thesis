#!/bin/bash
# Re-run DSA + prefetch after prefetch was fixed to load at the policy's miss tier.
cd /app/LLMServingSim
BASE="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-placement-interval 20 --expert-reduced-bytes 0.4 --expert-residency-policy DSA --dsa-quality-fill --expert-prefetch-policy ORACLE"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $BASE "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run fin_dsa0.5_pf    --expert-memory-fraction 0.5
run fin_dsa0.25_pf   --expert-memory-fraction 0.25
run fin_dsa0.25_pf75 --expert-memory-fraction 0.25 --expert-prefetch-accuracy 0.75
echo ALL_DONE
