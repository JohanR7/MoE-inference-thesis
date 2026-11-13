#!/bin/bash
# Sensitivity study + skew sweep. Base: Qwen3-30B-A3B, TP=2/EP=2, 30 ShareGPT requests, PCIe 5.0 host link,
# DSA + quality fill, 50% expert memory, interval 10, reduced size 0.4x, reduced speed 1.0, Zipf 1.2.
cd /app/LLMServingSim
DATA=workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl
COMMON="--cluster-config outputs/moe_ep2_pcie5.json --dtype bfloat16 --block-size 16 --dataset $DATA --num-reqs 30 --expert-routing-policy TRACE"
Z12="--expert-routing-trace outputs/zipf_qwen30.npz"
DSA="--expert-residency-policy DSA --dsa-quality-fill"
QNT="--expert-residency-policy QUANTIZED"
run() { name=$1; shift; start=$(date +%s); python3 -m serving $COMMON "$@" --output outputs/exp_$name.csv --iteration-output outputs/iter_$name.csv > outputs/exp_$name.log 2>&1; echo "$name exit=$? wall=$(( $(date +%s) - start ))s"; }

# Interval
for i in 5 20 50; do run sens_int$i $Z12 $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-placement-interval $i; done
# Bands
run sens_bands_strict $Z12 $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-placement-interval 10 --dsa-bands 0.8,0.5,0.2
run sens_bands_loose  $Z12 $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-placement-interval 10 --dsa-bands 0.5,0.2,0.05
# Reduced size
for b in 0.25 0.5 0.6; do run sens_size$b $Z12 $DSA --expert-reduced-bytes $b --expert-memory-fraction 0.5 --expert-placement-interval 10; done
run sens_quant_size0.6 $Z12 $QNT --expert-reduced-bytes 0.6 --expert-memory-fraction 0.5
# Reduced-expert speed
for s in 0.6 1.2; do
  run sens_speed$s       $Z12 $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-placement-interval 10 --expert-reduced-latency $s
  run sens_quant_speed$s $Z12 $QNT --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-reduced-latency $s
done
# Skew sweep (skew 1.2 already exists: freq50_i10, quant50, dsaq50, quant25, dsaq25)
for k in 0 0.5 1.0 1.5; do
  python3 -m workloads.generators routing-synthetic --model Qwen/Qwen3-30B-A3B-Instruct-2507 --dataset $DATA --num-reqs 30 --skew $k --phase-reqs 10 --stickiness 0.3 --output outputs/zipf_s$k.npz > /dev/null 2>&1; echo "trace skew $k exit=$?"
  T="--expert-routing-trace outputs/zipf_s$k.npz"
  run skew${k}_freq50 $T --expert-residency-policy FREQUENCY --expert-memory-fraction 0.5 --expert-placement-interval 10
  run skew${k}_quant50 $T $QNT --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5
  run skew${k}_dsa50  $T $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.5 --expert-placement-interval 10
done
for k in 0 0.5; do
  T="--expert-routing-trace outputs/zipf_s$k.npz"
  run skew${k}_quant25 $T $QNT --expert-reduced-bytes 0.4 --expert-memory-fraction 0.25
  run skew${k}_dsa25  $T $DSA --expert-reduced-bytes 0.4 --expert-memory-fraction 0.25 --expert-placement-interval 10
done
echo ALL_DONE
