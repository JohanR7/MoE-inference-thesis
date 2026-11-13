#!/bin/bash
# HOBBIT v2 (per-token precision from gating weights, two caches) under the same
# conditions as the final DSA results. Traces *_g.npz = same experts plus gate weights.
cd /app/LLMServingSim
BASE="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-placement-interval 20 --expert-reduced-bytes 0.4"
HOB="--expert-residency-policy HOBBIT"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $BASE "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
for m in 0.75 0.5 0.25; do run hob2_$m $HOB --expert-routing-trace outputs/zipf_qwen30_g.npz --expert-memory-fraction $m; done
run dsa_g0.5 --expert-residency-policy DSA --dsa-quality-fill --expert-routing-trace outputs/zipf_qwen30_g.npz --expert-memory-fraction 0.5
for m in 0.5 0.25; do run hob2_split50_$m $HOB --hobbit-high-cache 0.5 --expert-routing-trace outputs/zipf_qwen30_g.npz --expert-memory-fraction $m; done
for seed in 1 2; do for m in 0.5 0.25; do run seed${seed}_hob2_$m $HOB --expert-routing-trace outputs/zipf_seed${seed}_g.npz --expert-memory-fraction $m; done; done
echo ALL_DONE
