#!/bin/bash
# Heterogeneous EP with a MEASURED second GPU: GPU 0 = RTX PRO 6000, GPU 1 = T4 (Kaggle)
# (profiler/perf/T4, fp16 profile used via the bf16/fp16 fallback). Final DSA settings throughout.
# Memory is modelled as 96 GB on both GPUs (only the 4050's speed is measured).
cd /app/LLMServingSim
BASE="--dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-placement-interval 20 --expert-reduced-bytes 0.4"
T4="--cluster-config outputs/moe_ep2_hetero_t4.json $BASE --moe-rank-cost pairs"
DSA="--expert-residency-policy DSA --dsa-quality-fill"
run() { name=$1; shift; start=$(date +%s); python3 -m serving "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
# All experts resident
run t4_even100      $T4 --expert-placement-policy EVEN
run t4_swap100      $T4 --expert-placement-policy SWAP
run t4_dsaplace100  $T4 --expert-placement-policy DSA
run t4_fin100_imb05 $T4 $DSA --expert-placement-policy DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.5
run t4_fin100_imb09 $T4 $DSA --expert-placement-policy DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.9
# 50% expert memory
run t4_freq50       $T4 --expert-residency-policy FREQUENCY --expert-memory-fraction 0.5
run t4_dsares50     $T4 $DSA --expert-memory-fraction 0.5
run t4_fin50        $T4 $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5
run t4_fin50_imb09  $T4 $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5 --dsa-max-imbalance 0.9
echo ALL_DONE
