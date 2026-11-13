---
sidebar_position: 1
title: CLI flags
---

# `python -m serving` CLI flags

Complete reference for every command-line flag accepted by
`python -m serving`. For the conceptual side of each flag (what it
*does* internally), see **[Simulator](/docs/simulator/architecture)**.

:::tip 14 of these can be set per instance
Flags marked **(per-instance)** below can also be written into an
individual `instances[i]` object in the cluster config, which wins over
the CLI value for that instance only. That is how one run serves
heterogeneous instances. The other 15 flags are cluster-wide. See
**[Cluster config → Runtime overrides](./cluster-config#runtime-overrides-optional)**.
:::

## Cluster topology

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--cluster-config` | path | `configs/cluster/single_node_single_instance.json` | Path to a cluster-config JSON. See **[Cluster config](./cluster-config)** |
| `--network-backend` | choice | `analytical` | Network simulation backend. `analytical` (fast) or `ns3` (detailed, WIP) |

## Batching and scheduling

These flags are deployment defaults. A cluster config can override the
matching runtime knobs per `instances[i]`; see
**[Cluster config](./cluster-config#runtime-overrides-optional)**.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--max-num-seqs` **(per-instance)** | int | `128` | Max sequences in a batch. `0` = unlimited |
| `--max-num-batched-tokens` **(per-instance)** | int | `2048` | Max tokens per iteration across all requests (token budget). Clamped to the model config's `max_position_embeddings`, so `0` ("unlimited") resolves to the context length, not infinity |
| `--long-prefill-token-threshold` **(per-instance)** | int | `0` | Per-request token cap per step for chunked prefill. `0` = disabled |
| `--enable-chunked-prefill` **(per-instance)** | bool | `True` | Split long prefill across iterations. Use `--no-enable-chunked-prefill` to disable |
| `--npu-memory-utilization` **(per-instance,** as `npu_mem.mem_util`**)** | float | `0.9` | Fraction of NPU memory usable for weights plus KV cache. Corresponds to vLLM's `--gpu-memory-utilization`; KV capacity is `npu_mem.mem_size * this - model weight`. Override per instance with `npu_mem.mem_util` |
| `--reserve-full-isl` / `--no-reserve-full-isl` **(per-instance)** | flag | on | Admit a request only if its whole sequence fits, not merely its first chunk. Mirrors vLLM's `scheduler_reserve_full_isl`; without it chunked prefill over-admits and thrashes the KV cache |
| `--block-size` **(per-instance)** | int | `16` | KV cache block size in tokens |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `TRACE` / `CUSTOM` | `BALANCED` | MoE expert token routing. `TRACE` replays `--expert-routing-trace` and disables block copy |
| `--expert-routing-trace` | path | none | Routing trace (`.npz`) for `TRACE`, captured from the same workload as `--dataset` |
| `--expert-placement-policy` | `EVEN` / `POPULARITY` / `SWAP` / `DSA` / `CUSTOM` | `EVEN` | Expert to EP rank placement, recomputed from earlier iterations' routing. A moved expert is loaded from host memory on its new rank |
| `--expert-placement-interval` | int | `1` | Re-place every N iterations; load is tracked every iteration |
| `--expert-placement-max-swaps` | int | `1` | `SWAP`: most hot/cold expert swaps per layer per re-placement |
| `--charge-expert-migration` | bool | `True` | Charge a weight load per moved expert. `--no-charge-expert-migration` makes moves free, to bound what placement can gain |
| `--expert-memory-fraction` | float | `1.0` | Share of each EP rank's expert weights (full precision) that fits on the NPU. Below 1, the rest stay in host memory, a miss is fetched at `cpu_mem.mem_bw`, and the freed memory goes to KV cache. Needs per-token routing (not `BALANCED`) |
| `--expert-residency-policy` | `LRU` / `FREQUENCY` / `QUANTIZED` / `HOBBIT` / `DSA` / `CUSTOM` | `LRU` | Which experts stay on the NPU: a reactive LRU cache at full precision; `FREQUENCY`, which evicts the lowest observed load and refills each rank with its highest-load experts every re-placement; or user-defined tiers `FULL` / `REDUCED` / `HOST` |
| `--expert-reduced-bytes` | float | `0.5` | Size of a `REDUCED` expert relative to full |
| `--expert-reduced-latency` | float | `1.0` | Compute time of a `REDUCED` expert relative to the profiled one (1.0 until profiled) |
| `--hobbit-t1` | float | `0.6` | HOBBIT baseline: unimportance threshold above which a token's expert runs at low precision |
| `--hobbit-high-cache` | float | `0.75` | HOBBIT baseline: share of the expert memory budget for the high-precision cache; the rest holds low-precision copies |
| `--dsa-alpha` | float | `0.3` | DSA: EWMA smoothing factor |
| `--dsa-hedge-eta` | float | `1.0` | DSA: Hedge learning rate |
| `--dsa-horizon` | int | `20` | DSA: iterations a new placement is expected to last |
| `--dsa-switch-threshold-ns` | float | `100000` | DSA: minimum net gain to migrate (ns) |
| `--dsa-bands` | `F,R,E` | `0.7,0.3,0.1` | DSA: sensitivity thresholds for FULL, REDUCED, evict |
| `--dsa-max-imbalance` | float | `0.0` | DSA: a GPU may own up to (1 + x) times its even share of experts (within its memory at REDUCED size); needs DSA residency |
| `--dsa-quality-fill` | bool | `False` | DSA: spend spare expert memory on FULL precision, not only on residency |
| `--expert-prefetch-policy` | `NONE` / `ORACLE` / `CUSTOM` | `NONE` | Load experts of the next layer ahead of time. `ORACLE` uses the next layer's real routing (perfect-predictor bound, `TRACE` only) |
| `--expert-prefetch-accuracy` | float | `1.0` | `ORACLE`: probability each correct expert is kept; the rest are swapped for wrong ones |
| `--expert-placement-overhead` | `measured` / ns | `measured` | Time charged to an iteration whose placement was recomputed |
| `--moe-rank-cost` | `tokens` / `pairs` | `tokens` | Token count an EP rank's MoE latency is looked up with. `pairs` = local (token, expert) pairs / top-k; not yet validated against vLLM |
| `--enable-block-copy` **(per-instance)** | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--dtype` **(per-instance)** | `float16` / `bfloat16` / `float32` / `fp8` / `int8` | model's `torch_dtype`, fallback `bfloat16` | Model weight dtype |
| `--kv-cache-dtype` **(per-instance)** | `auto` / `fp8` | `auto` (inherits dtype) | KV cache dtype. `fp8` halves KV memory and selects a `*-kvfp8` profile variant |

## Prefix caching and offloading

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` **(per-instance)** | `True` | Prefix caching over a per-tier block pool with chained block hashes. Use `--no-enable-prefix-caching` to disable |
| `--enable-prefix-sharing` | off | Second-tier prefix pool shared across instances within a node |
| `--prefix-storage` | `None` | Where the second-tier pool lives. `None` / `CPU` / `CXL` |
| `--enable-local-offloading` **(per-instance)** | off | Weight offloading to NPU (counts weight reads in profiling) |
| `--enable-attn-offloading` **(per-instance)** | off | Attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` **(per-instance)** | off | Overlap GPU compute with PIM attention. Requires `--enable-attn-offloading` |

## Dataset and output

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | path | `None` | JSONL workload file. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
| `--num-reqs` | int | `0` | Entries to load from the dataset (`0` = all). For agentic, each entry is a session |
| `--output` | path | `None` | Per-request CSV output path. Stdout only if `None`. The literal `{run_id}` is replaced with the active run id |
| `--iteration-output` | path | `None` | Per-iteration CSV: one row per NPU per batch with compute and non-compute time, batch shape and expert weight traffic. `{run_id}` is replaced as in `--output` |

## Run isolation

Each invocation writes ASTRA-Sim intermediates under a run-specific input
root so parallel simulations do not overwrite each other's generated
configs, traces, or Chakra workloads. Generated text traces are removed
after Chakra conversion by default, and the run-specific input root is
removed after a successful simulation by default.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | string | auto-generated | Path-safe id for this simulation run. Used in `astra-sim/inputs/runs/<run-id>` and the `{run_id}` output placeholder |
| `--inputs-root` | path | `astra-sim/inputs/runs/<run-id>` | Override the generated ASTRA-Sim input root, for example to place intermediates on local SSD or tmpfs |
| `--cleanup-inputs` / `--no-cleanup-inputs` | bool | `true` | Remove generated trace files after Chakra conversion and remove the generated run directory after a successful simulation. Use `--no-cleanup-inputs` to preserve traces, Chakra workloads, and input configs for debugging |

## Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-interval` | float | `1.0` | Seconds between throughput / memory log lines |
| `--log-level` | choice | `WARNING` | `WARNING` (default) / `INFO` / `DEBUG` |

## Quick reference: which flag for which feature

| Feature | Flag(s) |
| --- | --- |
| Multi-instance (parallelism via cluster config) | (cluster config `num_instances`) |
| Tensor parallel | (cluster config `tp_size`) |
| MoE expert parallel | (cluster config `ep_size`) |
| DP+EP MoE | (cluster config `dp_group`) |
| Prefix caching | `--enable-prefix-caching` (default on), `--enable-prefix-sharing`, `--prefix-storage` |
| Chunked prefill | `--enable-chunked-prefill` (default on), `--long-prefill-token-threshold` |
| PIM attention offload | `--enable-attn-offloading` (cluster config sets `pim_config`) |
| FP8 KV cache | `--kv-cache-dtype fp8` |
| ns3 backend | `--network-backend ns3` |
| Heterogeneous instances in one run | (cluster config per-instance overrides; see the tip above) |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.
