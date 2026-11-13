#!/bin/bash
# Final DSA configuration: residency DSA + quality fill, interval 20, reduced size 0.4x.
# Baselines that decide periodically (FREQUENCY) use the same interval 20.
cd /app/LLMServingSim
DATA=workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl
BASE="--dtype bfloat16 --block-size 16 --dataset $DATA --num-reqs 30 --expert-routing-policy TRACE --expert-placement-interval 20 --expert-reduced-bytes 0.4"
HOMO="--cluster-config outputs/moe_ep2_pcie5.json $BASE"
HET="--cluster-config outputs/moe_ep2_hetero.json $BASE --moe-rank-cost pairs"
DSA="--expert-residency-policy DSA --dsa-quality-fill"
FRQ="--expert-residency-policy FREQUENCY"
QNT="--expert-residency-policy QUANTIZED"
T0="--expert-routing-trace outputs/zipf_qwen30.npz"
run() { name=$1; shift; start=$(date +%s); python3 -m serving "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }

# A. Main results with final settings (seed 0 trace)
for m in 0.75 0.5 0.25; do
  run fin_dsa$m  $HOMO $T0 $DSA --expert-memory-fraction $m
  run fin_freq$m $HOMO $T0 $FRQ --expert-memory-fraction $m
done
# B. Two more routing traces (same skew 1.2 settings, different random seed)
for seed in 1 2; do
  python3 -m workloads.generators routing-synthetic --model Qwen/Qwen3-30B-A3B-Instruct-2507 --dataset $DATA --num-reqs 30 --skew 1.2 --phase-reqs 10 --stickiness 0.3 --seed $seed --output outputs/zipf_seed$seed.npz > /dev/null 2>&1; echo "trace seed $seed exit=$?"
  T="--expert-routing-trace outputs/zipf_seed$seed.npz"
  for m in 0.5 0.25; do
    run seed${seed}_dsa$m   $HOMO $T $DSA --expert-memory-fraction $m
    run seed${seed}_freq$m  $HOMO $T $FRQ --expert-memory-fraction $m
    run seed${seed}_quant$m $HOMO $T $QNT --expert-memory-fraction $m
  done
done
# C. Gaps
run fin_place100   $HOMO $T0 --expert-placement-policy DSA --expert-residency-policy LRU --expert-memory-fraction 1.0
run fin_dsa0.5_pf  $HOMO $T0 $DSA --expert-memory-fraction 0.5  --expert-prefetch-policy ORACLE
run fin_dsa0.25_pf $HOMO $T0 $DSA --expert-memory-fraction 0.25 --expert-prefetch-policy ORACLE
run fin_dsa0.25_pf75 $HOMO $T0 $DSA --expert-memory-fraction 0.25 --expert-prefetch-policy ORACLE --expert-prefetch-accuracy 0.75
# D. Heterogeneous with final settings
run het_fin100_imb $HET $T0 $DSA --expert-placement-policy DSA --expert-memory-fraction 1.0 --dsa-max-imbalance 0.5
run het_fin50      $HET $T0 $DSA --expert-placement-policy DSA --expert-memory-fraction 0.5
run het_fin_freq50 $HET $T0 $FRQ --expert-memory-fraction 0.5
echo ALL_DONE
