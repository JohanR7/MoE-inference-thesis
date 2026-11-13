#!/bin/bash
# DSA scheduler vs baselines: Qwen3-30B-A3B, TP=2/EP=2, 30 ShareGPT requests, PCIe 5.0 host link.
# Earlier baselines (same setup): mem50 = LRU at 50% memory, freq50 / freq50_i10 = FREQUENCY.
cd /app/LLMServingSim
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz"
DSA="--expert-residency-policy DSA --expert-reduced-bytes 0.4"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run freq50_i10_recheck --expert-memory-fraction 0.5 --expert-residency-policy FREQUENCY --expert-placement-interval 10
run dsa50_i1     --expert-memory-fraction 0.5  $DSA
run dsa50_i10    --expert-memory-fraction 0.5  $DSA --expert-placement-interval 10
run dsafull50_i10 --expert-memory-fraction 0.5 $DSA --expert-placement-policy DSA --expert-placement-interval 10
run dsa75_i10    --expert-memory-fraction 0.75 $DSA --expert-placement-interval 10
run dsa25_i10    --expert-memory-fraction 0.25 $DSA --expert-placement-interval 10
echo ALL_DONE
