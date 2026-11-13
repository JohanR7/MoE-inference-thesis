#!/bin/bash
# Steps 2-4: quality fill, faster placement, heterogeneous GPUs (GPU 1 = stand-in 2x slower experts).
cd /app/LLMServingSim
BASE="--dtype bfloat16 --block-size 16 --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --num-reqs 30 --expert-routing-policy TRACE --expert-routing-trace outputs/zipf_qwen30.npz --expert-placement-interval 10"
HOMO="--cluster-config outputs/moe_ep2_pcie5.json $BASE"
HET="--cluster-config outputs/moe_ep2_hetero.json $BASE --moe-rank-cost pairs"
DSA="--expert-residency-policy DSA --expert-reduced-bytes 0.4"
run() { name=$1; shift; start=$(date +%s); python3 -m serving "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }
# Step 2: quality fill (identical GPUs)
run dsaq75  $HOMO $DSA --dsa-quality-fill --expert-memory-fraction 0.75
run dsaq50  $HOMO $DSA --dsa-quality-fill --expert-memory-fraction 0.5
run dsaq25  $HOMO $DSA --dsa-quality-fill --expert-memory-fraction 0.25
# Step 3: faster placement (compare with exp_dsafull50_i10: 33.7 s, 10.3 s decision time)
run dsafull50_fast $HOMO $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5
# Step 4: heterogeneous GPUs, all experts resident
run het_even100     $HET --expert-placement-policy EVEN
run het_swap100     $HET --expert-placement-policy SWAP
run het_dsaplace100 $HET --expert-placement-policy DSA
# Step 4: heterogeneous GPUs, 50% expert memory
run het_freq50    $HET --expert-memory-fraction 0.5 --expert-residency-policy FREQUENCY
run het_dsa50     $HET --expert-memory-fraction 0.5 $DSA --dsa-quality-fill
run het_dsafull50 $HET --expert-memory-fraction 0.5 $DSA --dsa-quality-fill --expert-placement-policy DSA
echo ALL_DONE
