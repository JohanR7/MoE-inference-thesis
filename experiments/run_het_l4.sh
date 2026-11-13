#!/bin/bash
# Heterogeneous EP with a MEASURED second GPU: GPU 0 = RTX PRO 6000, GPU 1 = L4 (Camber)
# (profiler/perf/L4, bf16 profile). Final DSA settings throughout.
# Memory is modelled as 96 GB on both GPUs (only the L4's speed is measured).
cd /app/LLMServingSim
BASE="--dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-placement-interval 20 --expert-reduced-bytes 0.4"
L4="--cluster-config outputs/moe_ep2_hetero_l4.json $BASE --moe-rank-cost pairs"
DSA="--expert-residency-policy DSA --dsa-quality-fill"
run() { name=$1; shift; start=$(date +%s); python3 -m serving "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
# All experts resident
run l4_even100      $L4 --expert-placement-policy EVEN
run l4_swap100      $L4 --expert-placement-policy SWAP
run l4_dsaplace100  $L4 --expert-placement-policy DSA
run l4_fin100_imb05 $L4 $DSA --expert-placement-policy DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.5
run l4_fin100_imb09 $L4 $DSA --expert-placement-policy DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.9
# 50% expert memory
run l4_freq50       $L4 --expert-residency-policy FREQUENCY --expert-memory-fraction 0.5
run l4_dsares50     $L4 $DSA --expert-memory-fraction 0.5
run l4_fin50        $L4 $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5
run l4_fin50_imb09  $L4 $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5 --dsa-max-imbalance 0.9
echo ALL_DONE
