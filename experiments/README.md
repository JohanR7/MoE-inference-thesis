# DSA-Aware MoE scheduling: experiment scripts

Scripts and configs behind the thesis results. They were written to run from
`outputs/` (which is git-ignored), so copy them there first. Run everything inside
the simulator container from the repo root (`/app/LLMServingSim`).

## 1. Set up from a fresh clone

The Chakra converter lives in a nested submodule, so its change (the `PREFETCH`
lines used by expert prefetching) is kept as a patch. Apply it **before**
building, because `scripts/compile.sh` installs Chakra into the container:

```bash
git clone --recurse-submodules <this repo's URL> LLMServingSim
cd LLMServingSim
git -C astra-sim/extern/graph_frontend/chakra apply ../../../../experiments/patches/chakra-llm_converter.patch
./scripts/docker-sim.sh          # simulator container (mounts the repo at /app/LLMServingSim)
./scripts/compile.sh             # inside the container: builds ASTRA-Sim, installs patched Chakra
cp experiments/*.sh experiments/*.json experiments/*.py outputs/
```

If Chakra was installed before the patch was applied, reinstall it inside the
container: `cd astra-sim/extern/graph_frontend/chakra && pip3 install . && cd -`.

## 2. Generate the synthetic routing traces

Traces are not committed (about 20 MB each). Gating weights are stored as
well, so the same file serves the HOBBIT runs (`*_g.npz` names in the scripts).

```bash
gen() { python3 -m workloads.generators routing-synthetic \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl \
  --num-reqs 30 --phase-reqs 10 --stickiness 0.3 "$@"; }
gen --skew 1.2 --output outputs/zipf_qwen30.npz
cp outputs/zipf_qwen30.npz outputs/zipf_qwen30_g.npz
gen --skew 1.2 --seed 1 --output outputs/zipf_seed1.npz   # repeatability
gen --skew 1.2 --seed 2 --output outputs/zipf_seed2.npz
cp outputs/zipf_seed1.npz outputs/zipf_seed1_g.npz; cp outputs/zipf_seed2.npz outputs/zipf_seed2_g.npz
for s in 0 0.5 1.0 1.5; do gen --skew $s --output outputs/zipf_s$s.npz; done   # skew sweep
```

## 3. Run the experiments

| Script | What it produces |
|---|---|
| `run_final.sh`, `run_final_pf.sh` | Main results (final configuration, memory budgets, prefetching) |
| `run_baselines.sh`, `run_frequency.sh`, `run_quantized.sh`, `run_hobbit2.sh` | Baseline policies (`run_hobbit2.sh` = HOBBIT with two caches) |
| `run_memory.sh`, `run_prefetch.sh`, `run_dsa.sh`, `run_steps234.sh` | Exploratory and ablation runs |
| `run_sensitivity.sh` | Sensitivity study and skew sweep |
| `run_imbalance.sh` | 2x-slower stand-in device (`moe_ep2_hetero.json`) |
| `run_het4050.sh`, `run_het_l4.sh`, `run_het_t4.sh` | Measured heterogeneous pairs (RTX PRO 6000 + RTX 4050 / L4 / T4) |

Each run writes `outputs/exp_<name>.{csv,log}` and `outputs/iter_<name>.csv`.
`test_*.py` are unit checks of the policies; run them from the repo root (`python3 experiments/test_dsa.py`, ...).

## 4. Profiles

New latency profiles are under `profiler/perf/`: `RTX4050L`, `L4` (MoE table
and meta only), `T4` (fp16, not used: vLLM's MoE kernel is pathological on
sm75), `STANDIN-2xSLOW` (RTX PRO 6000 MoE x2), and `CPU-i5-13500HX` (MoE only,
reference PyTorch on CPU). The scripts in `profiling/` reproduce them:
`run_t4_profile.sh` (Kaggle T4), `run_l4_profile.sh` (L4 notebook) and
`profile_cpu_moe.py` (any CPU).
