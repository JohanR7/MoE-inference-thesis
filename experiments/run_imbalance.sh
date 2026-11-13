#!/bin/bash
# Unequal expert ownership on heterogeneous GPUs (GPU 1 = stand-in, experts 2x slower).
cd /app/LLMServingSim
HET="--cluster-config outputs/moe_ep2_hetero.json --dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-placement-interval 10 --moe-rank-cost pairs"
DSA="--expert-placement-policy DSA --expert-residency-policy DSA --expert-reduced-bytes 0.4 --dsa-quality-fill"
run() { name=$1; shift; start=$(date +%s); python3 -m serving "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
run het_dsafull100_q   $HET $DSA --expert-memory-fraction 1.0
run het_dsafull100_imb $HET $DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.5
run het_dsafull50_imb  $HET $DSA --expert-memory-fraction 0.5 --dsa-max-imbalance 0.5
echo ALL_DONE
