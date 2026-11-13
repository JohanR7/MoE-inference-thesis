"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import csv               # new: writes the --iteration-output CSV
import yaml              # new: reads the generated network config (congestion backend check)
import subprocess
import argparse
import json
import shutil
from time import time
from collections import defaultdict

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.logger import *
from serving.core.run_paths import build_run_paths, resolve_run_id
from serving.core.expert_placement import ExpertPlacement, expert_weight_bytes   # new: expert state + expert size
from serving.core.gate_function import load_routing_trace                          # new: opens a routing trace once
from serving.core.dsa_scheduler import DSAScheduler                                # the team's DSA scheduler
from serving.core.hobbit_policy import HobbitPolicy                                # HOBBIT-style baseline
from serving.core.trace_generator import resolve_variant, _load_perf_db, _load_rank_perf_db, _lookup_moe   # DSA: per-GPU latency
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    MoE AG/RS comm size is anchored separately to ``max_total_len`` (no
    ``× group_size``) in the iteration loop — that calibrates the
    bandwidth model against the same ``link_bw`` AllReduce already uses.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _runtime_limit(value):
    return float('inf') if value == 0 else value


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _cleanup_inputs_root(run_paths, logger):
    """Remove generated ASTRA-Sim inputs after a completed simulation."""
    runs_root = os.path.abspath(os.path.join("inputs", "runs"))
    inputs_root = os.path.abspath(run_paths.inputs_root)
    if inputs_root in (os.path.abspath("inputs"), runs_root):
        raise RuntimeError(f"Refusing to remove broad inputs root: {inputs_root}")
    if not inputs_root.startswith(runs_root + os.sep):
        logger.warning(
            "Skipping ASTRA-Sim inputs cleanup because inputs_root is outside %s: %s",
            runs_root, inputs_root,
        )
        return
    shutil.rmtree(inputs_root, ignore_errors=True)
    logger.info("Removed ASTRA-Sim inputs root: %s", inputs_root)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtype(instance, cli_dtype, dtype_to_bits):
    dtype = instance.get("dtype", cli_dtype)
    if dtype is None:
        config = get_config(instance["model_name"])
        torch_dtype = config.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in dtype_to_bits:
            dtype = torch_dtype
        else:
            dtype = "bfloat16"
    if dtype not in dtype_to_bits:
        raise ValueError(f"Unsupported dtype '{dtype}' for instance {instance.get('instance_id')}")
    return dtype


def _resolve_mem_util(instance, cli_default):
    """Per-instance NPU memory utilization, from ``npu_mem.mem_util``.

    Lives inside ``npu_mem`` because its only job is to scale ``mem_size``, and
    it follows that block's ``mem_*`` naming. Falls back to the CLI default.
    """
    util = instance.get("npu_mem", {}).get("mem_util", cli_default)
    try:
        util = float(util)
    except (TypeError, ValueError):
        raise ValueError(
            f"npu_mem.mem_util for instance {instance.get('instance_id')} must be a "
            f"number in (0, 1]; got {util!r}"
        ) from None
    if not 0 < util <= 1:
        raise ValueError(
            f"npu_mem.mem_util for instance {instance.get('instance_id')} must be in "
            f"(0, 1]; got {util}. It is a fraction of npu_mem.mem_size, so 0.9 rather "
            f"than 90"
        )
    return util


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    runtime_configs = []
    for instance_id, instance in enumerate(instances):
        dtype = _resolve_instance_dtype(instance, args.dtype, dtype_to_bits)
        kv_cache_dtype = instance.get("kv_cache_dtype", args.kv_cache_dtype)
        if kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(f"Unsupported kv_cache_dtype '{kv_cache_dtype}' for instance {instance_id}")

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")
        if enable_sub_batch_interleaving and instance.get("pp_size", 1) > 1:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving with pp_size "
                f"{instance['pp_size']}: an interleaved trace leaves both sub-batches "
                f"mid-block at every group edge, so a pipeline stage has no single "
                f"hidden state to pass on")

        runtime_configs.append({
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": instance.get("block_size", args.block_size),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "npu_memory_utilization": _resolve_mem_util(
                instance, args.npu_memory_utilization),
            "reserve_full_isl": instance.get("reserve_full_isl", args.reserve_full_isl),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
        })
    return runtime_configs


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_single_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32', 'fp8', 'int8'], default=None,
                        help='model weight data type (vLLM-style). When omitted, defaults to the model config\'s '
                        '``torch_dtype`` (falling back to bfloat16). Overrides only take effect if the profiler '
                        'produced matching data under perf/<hw>/<model>/<variant>/tp<N>/')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'TRACE', 'CUSTOM'],   # TRACE is new
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'TRACE (replay recorded router decisions from --expert-routing-trace; '
                        'disables block copy), CUSTOM (user-defined)')
    # ---- New expert flags. Their defaults reproduce the original simulator exactly:
    # routing trace (A2), placement (A1, baselines), migration cost (A6), memory pressure
    # and precision tiers (A4), prefetch (C13), decision time (C15), rank cost.
    parser.add_argument('--expert-routing-trace', type=str, default=None,
                        help='routing trace (.npz) replayed by --expert-routing-policy TRACE, '
                        'captured from the same workload file as --dataset')
    parser.add_argument('--expert-placement-policy', type=str,
                        choices=['EVEN', 'POPULARITY', 'SWAP', 'DSA', 'CUSTOM'], default='EVEN',
                        help='expert -> EP rank placement for MoE models, recomputed every '
                        '--expert-placement-interval iterations from routing observed in earlier '
                        'iterations: EVEN (default; contiguous even split, never changes), '
                        'POPULARITY (greedy packing of observed per-expert load, rebuilds the whole '
                        'map), SWAP (incremental: up to --expert-placement-max-swaps hot/cold swaps '
                        'per layer), CUSTOM (user-defined). A moved expert is loaded from host '
                        'memory (cpu_mem.mem_bw) on its new rank before its next expert compute')
    parser.add_argument('--expert-placement-interval', type=int, default=1,
                        help='re-place experts every N iterations (load is tracked every iteration)')
    parser.add_argument('--expert-placement-max-swaps', type=int, default=1,
                        help='SWAP placement: most expert swaps per layer per re-placement')
    parser.add_argument('--charge-expert-migration', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='charge a weight load for every expert placement moves (default). '
                        '--no-charge-expert-migration makes moves free, to bound what placement '
                        'can gain')
    parser.add_argument('--expert-memory-fraction', type=float, default=1.0,
                        help='share of each EP rank\'s expert weights (full precision) that fits on '
                        'the NPU (default 1.0: all resident). Below 1, the rest stay in host memory, '
                        'a token routed to one is a miss fetched at cpu_mem.mem_bw, and the freed '
                        'memory goes to KV cache. Needs per-token routing (not BALANCED)')
    parser.add_argument('--expert-residency-policy', type=str,
                        choices=['LRU', 'FREQUENCY', 'QUANTIZED', 'HOBBIT', 'DSA', 'CUSTOM'],
                        default='LRU',
                        help='which experts stay on the NPU: LRU (default; reactive cache at full '
                        'precision, evicting the least recently used), FREQUENCY (evict the lowest '
                        'observed load, and refill each rank with its highest-load experts every '
                        're-placement), QUANTIZED (every expert at REDUCED precision, LRU when even '
                        'those do not fit), DSA (the DSA scheduler) or CUSTOM (user-defined tiers FULL / REDUCED / HOST each '
                        're-placement, LRU evicting what does not fit)')
    parser.add_argument('--expert-reduced-bytes', type=float, default=0.5,
                        help='size of a REDUCED-precision expert relative to full (default 0.5, int8 '
                        'from 16-bit)')
    parser.add_argument('--expert-reduced-latency', type=float, default=1.0,
                        help='compute time of a REDUCED expert relative to the profiled one (default '
                        '1.0: no speedup assumed until profiled)')
    parser.add_argument('--expert-prefetch-policy', type=str, choices=['NONE', 'ORACLE', 'CUSTOM'],
                        default='NONE',
                        help='load experts of the next layer ahead of time, overlapping the current '
                        'layer\'s combine and the next layer\'s attention: NONE (default), ORACLE '
                        '(the next layer\'s real misses: perfect-predictor bound, TRACE routing only) '
                        'or CUSTOM (user-defined). Only matters with --expert-memory-fraction < 1')
    parser.add_argument('--expert-prefetch-accuracy', type=float, default=1.0,
                        help='ORACLE prefetch: probability each correct expert is kept; the rest are '
                        'swapped for wrong ones (default 1.0)')
    # DSA scheduler settings (serving/core/dsa_scheduler.py); used when a policy above is DSA
    parser.add_argument('--hobbit-t1', type=float, default=0.6,
                        help='HOBBIT baseline: unimportance threshold for loading a missed expert at '
                        'high precision (default 0.6, the paper\'s Mixtral value)')
    parser.add_argument('--hobbit-high-cache', type=float, default=0.75,
                        help='HOBBIT baseline: share of the expert memory budget given to the '
                        'high-precision cache; the rest holds low-precision copies (default 0.75)')
    parser.add_argument('--dsa-alpha', type=float, default=0.3,
                        help='DSA stage A: EWMA smoothing factor (default 0.3)')
    parser.add_argument('--dsa-hedge-eta', type=float, default=1.0,
                        help='DSA stage B: Hedge learning rate (default 1.0)')
    parser.add_argument('--dsa-horizon', type=int, default=20,
                        help='DSA stage D: iterations a new placement is expected to last (default 20)')
    parser.add_argument('--dsa-switch-threshold-ns', type=float, default=100_000,
                        help='DSA stage D: minimum net gain to migrate, in ns (default 0.1 ms)')
    parser.add_argument('--dsa-bands', type=str, default='0.7,0.3,0.1',
                        help='DSA stage E: sensitivity thresholds FULL,REDUCED,EVICT (default 0.7,0.3,0.1)')
    parser.add_argument('--dsa-max-imbalance', type=float, default=0.0,
                        help='DSA stage C: a GPU may own up to (1 + x) times its even share of experts '
                        'per layer, within what its memory holds at REDUCED size (default 0: equal). '
                        'Needs --expert-residency-policy DSA')
    parser.add_argument('--dsa-quality-fill', action=argparse.BooleanOptionalAction, default=False,
                        help='DSA stage E: spend spare expert memory on FULL precision too, not only on '
                        'keeping experts resident at REDUCED (default off)')
    parser.add_argument('--expert-placement-overhead', type=str, default='measured',
                        help='time charged to an iteration whose expert placement was recomputed: '
                        '"measured" (default; wall clock of the policy in this process) or a fixed '
                        'value in ns. EVEN never recomputes, so it is never charged')
    parser.add_argument('--moe-rank-cost', type=str, choices=['tokens', 'pairs'], default='tokens',
                        help='token count an EP rank\'s MoE latency is looked up with: tokens '
                        '(default; tokens that reached the rank) or pairs (local (token, expert) '
                        'pairs / top-k, so a token hitting one local expert costs less than one '
                        'hitting k). pairs is not yet validated against vLLM')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching to reuse KV cache blocks across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], default='None',
                        help='storage medium for the second-tier prefix cache pool: None (NPU only), CPU, or CXL')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--reserve-full-isl', action=argparse.BooleanOptionalAction, default=True,
                        help='admit a request only if its whole sequence fits in the KV cache, '
                        'not merely its first chunk. Mirrors vLLM\'s scheduler_reserve_full_isl '
                        '(True there too); without it chunked prefill over-admits and thrashes '
                        'the KV cache. Override per instance with "reserve_full_isl"')
    parser.add_argument('--npu-memory-utilization', type=float, default=0.9,
                        help='fraction of NPU memory an instance may use for weights plus '
                        "KV cache. Corresponds to vLLM's --gpu-memory-utilization, renamed "
                        'because every other memory surface here is NPU-terminology; '
                        'override per instance with "npu_mem": {"mem_util": ...}. KV capacity is '
                        '(npu_mem * this - model weight); the activation peak and CUDA '
                        'context that vLLM also subtracts are not modelled, so the '
                        'resulting capacity is an upper bound on vLLM\'s at the same value')
    parser.add_argument('--block-size', type=int, default=16,
                        help='KV cache block size in tokens (number of tokens per block)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--cleanup-inputs', action=argparse.BooleanOptionalAction, default=True,
                        help='remove generated ASTRA-Sim inputs under astra-sim/inputs/runs/<run-id> '
                        'after a successful simulation (default: enabled). Use --no-cleanup-inputs '
                        'to preserve generated trace files, Chakra workloads, and input configs for debugging')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    # New (B8): per-batch timing and expert-traffic CSV
    parser.add_argument('--iteration-output', type=str, default=None,
                        help='write one CSV row per NPU per batch: time span, GPU compute, non-compute '
                        'time (communication, memory loads, idle), batch shape and expert weight '
                        'traffic. Off by default; {run_id} is replaced like in --output')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8'], default='auto',
                        help='KV cache data type: auto (inherit --dtype) or fp8. Selects the profile '
                        'variant folder -- fp8 resolves to <dtype>-kvfp8, e.g. bf16-kvfp8 -- and '
                        'halves KV cache memory. Override per instance with "kv_cache_dtype"')
    parser.add_argument('--network-backend', type=str,
                        choices=['analytical', 'analytical-congestion', 'ns3'], default='analytical',   # new choice (C14)
                        help='network simulation backend: analytical (fast, default; transfers sharing '
                        'a link do not slow each other), analytical-congestion (links are shared: '
                        'concurrent transfers split bandwidth; 1-D topologies only, i.e. a single '
                        'instance without DP groups) or ns3 (detailed, WIP)')

    args = parser.parse_args()
    
    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)
    args.iteration_output = _resolve_output_file(args.iteration_output, args.run_id)   # fill in {run_id}

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    print_input_config(args=args)
    flush.stdout.flush()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        inputs_root=run_paths.inputs_root)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=run_paths.network_config
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'analytical-congestion':   # new (C14): links shared between transfers
        network=run_paths.network_config               # same network file as 'analytical'
        with open(network) as f:
            network_dims = yaml.safe_load(f)["npus_count"]   # one entry per network dimension
        # ASTRA-Sim's congestion-aware model only handles 1-D networks; fail clearly
        # here instead of letting the C++ binary exit with a cryptic message
        if len(network_dims) != 1:
            raise ValueError(
                f"--network-backend analytical-congestion supports 1-D topologies only, but this "
                f"cluster generates {len(network_dims)} dimensions {network_dims} (multiple "
                f"instances or DP groups). Use --network-backend analytical.")
        # The congestion-aware build of ASTRA-Sim (already compiled in the container)
        binary=os.path.join(astra_sim, "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware")
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if any_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(model, cfg["fp"], cfg["kv_cache_dtype"])

        def _pool_block_size(inst_ids):
            sizes = {instance_runtime_configs[i]["block_size"] for i in inst_ids}
            if len(sizes) > 1:
                raise RuntimeError(
                    f"Shared prefix pool requires instances to share block_size; got {sizes}")
            return sizes.pop()

        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] <= 0:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
                inst_ids = node2inst_mapping[i]
                prefix_pools.append(build_prefix_pool(
                    pool_device, cpu_mem_size[i] * GB_TO_BYTE,
                    _pool_block_size(inst_ids), _pool_kv_bytes_per_token(inst_ids),
                    node_id=i))
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] <= 0:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            inst_ids = list(range(num_instances))
            prefix_pools.append(build_prefix_pool(
                pool_device, cluster["cxl_mem_size"] * GB_TO_BYTE,
                _pool_block_size(inst_ids), _pool_kv_bytes_per_token(inst_ids)))
            # This means every instance shares the same universal prefix pool (maybe fixed later)
            prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
            npu_memory_utilization=inst_cfg["npu_memory_utilization"],
            reserve_full_isl=inst_cfg["reserve_full_isl"],
            expert_memory_fraction=args.expert_memory_fraction,   # new: sizes expert weight vs KV cache
        ))

    # The derived KV capacity, not the utilization fraction, is what decides
    # memory pressure. It is per instance and only known once the schedulers
    # exist, so it gets its own section rather than a row in the input-config
    # block, which is printed before any of this is resolved.
    print_heading("KV Cache Initialization")
    print_markup("")
    # Pad only as far as the widest label, so the line stays inside the rule.
    pad = max(len(f"Instance [{i}]") for i in range(len(schedulers)))
    for inst_id, sched in enumerate(schedulers):
        pool = sched.memory.npu_pool
        label = f"Instance \\[{inst_id}]"
        print_markup(
            f"  \u2022 [cyan]{label:<{pad + 1}}[/cyan] : "
            f"{pool.num_blocks * pool.block_size} tokens / {pool.num_blocks} blocks "
            f"({pool.num_blocks * pool.bytes_per_block / GB_TO_BYTE:.2f} GiB/rank "
            f"at util {sched.memory.npu_memory_utilization:.2f})"
        )
    print_rule()

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(num_instances, schedulers, num_req, request_routing_policy)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        router.load_requests(dataset, enable_prefix_caching=any_prefix_caching, is_init=is_init)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    # Per-interval token counts -> tokens/s, so 1/log_interval. Floor division
    # collapsed to 0 for any --log-interval > 1, which zeroed every logged
    # throughput and made the summary line divide by zero.
    RATIO = FREQ/INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = router.get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time), inputs_root=run_paths.inputs_root)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root,
                   cleanup_trace=args.cleanup_inputs)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    astra_args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        astra_args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        astra_args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        astra_args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Pending batches per DP group (waiting for all members to schedule)
    dp_pending = {dg: {} for dg in dp_groups}  # dp_group -> {instance_id: (new_req, sys)}
    # Pre-generated workloads ready to submit on next "Waiting"
    dp_ready_workloads = {}  # instance_id -> workload_path

    # Expert placement persists across iterations: one object per instance,
    # shared by a DP group because EP spans the group.
    routing_trace = None                 # only TRACE routing uses one
    if expert_routing_policy == 'TRACE':
        if args.expert_routing_trace is None:     # TRACE without a file has nothing to replay
            raise ValueError("--expert-routing-policy TRACE needs --expert-routing-trace")
        # '../' because the simulator runs from astra-sim/, while paths are given from the repo root
        routing_trace = load_routing_trace(f'../{args.expert_routing_trace}')
    # None = measure the real decision time; otherwise a fixed number of ns
    placement_overhead_ns = (None if args.expert_placement_overhead == 'measured'
                             else int(args.expert_placement_overhead))
    expert_placements = {}  # instance_id -> ExpertPlacement (MoE instances only)
    shared_placements = {}  # instance or DP-group name -> ExpertPlacement (each object once)

    def _dsa_inputs(inst, model_cfg, num_experts):
        """Settings and measured inputs for DSAScheduler."""
        # Every instance sharing this placement (a DP group, or just this one), in EP-rank order
        members = sorted((m for m in instances
                          if (m.get("dp_group") or f'instance{m["instance_id"]}')
                          == (inst.get("dp_group") or f'instance{inst["instance_id"]}')),
                         key=lambda m: m["dp_rank"])
        experts_per_rank = num_experts // inst["ep_total"]
        top_k = model_cfg.get("num_experts_per_tok", 1)
        ref_tokens = 128                  # reference batch for the per-pair latency estimate
        latency = []
        for m in members:
            cfg = instance_runtime_configs[m["instance_id"]]
            variant = resolve_variant(cfg["dtype"], cfg["kv_cache_dtype"], model_cfg)
            for hw in m.get("ep_hardware") or [m["hardware"]] * m["local_ep"]:   # one per GPU
                db = _load_rank_perf_db(hw, m["model_name"], variant, model_cfg["model_type"])
                # Profiled MoE time for ref_tokens tokens over this rank's experts, per (token, expert) pair
                t_ns = _lookup_moe(db, ref_tokens, min(experts_per_rank, ref_tokens * top_k))
                latency.append(t_ns / (ref_tokens * top_k))
        node_cpu = raw_cluster_config["nodes"][inst["node_id"]]["cpu_mem"]   # this node's host memory
        return dict(rank_latency_ns=latency,
                    host_bw=float(node_cpu["mem_bw"]),         # GB/s = bytes per ns
                    alpha=args.dsa_alpha, hedge_eta=args.dsa_hedge_eta, horizon=args.dsa_horizon,
                    switch_threshold_ns=args.dsa_switch_threshold_ns,
                    bands=tuple(float(b) for b in args.dsa_bands.split(',')),
                    quality_fill=args.dsa_quality_fill,
                    max_imbalance=args.dsa_max_imbalance)
    for inst in instances:
        model_cfg = get_config(inst["model_name"])
        num_experts = model_cfg.get("num_local_experts", model_cfg.get("num_experts"))
        if not num_experts:              # dense model: no experts to place
            continue
        # Reject flag combinations the new code does not support, before the run starts
        if args.expert_prefetch_policy != 'NONE':
            if args.expert_prefetch_policy == 'ORACLE' and expert_routing_policy != 'TRACE':
                raise ValueError("--expert-prefetch-policy ORACLE reads the next layer's routing "
                                 "and needs --expert-routing-policy TRACE")
            # The converter's PREFETCH support covers the colocated, single-stage path only
            if (inst["pp_size"] > 1 or inst["pd_type"] is not None
                    or instance_runtime_configs[inst["instance_id"]]["enable_sub_batch_interleaving"]):
                raise ValueError("Expert prefetch supports only pp_size 1, no prefill/decode "
                                 "disaggregation and no sub-batch interleaving")
        if ((args.expert_placement_policy != 'EVEN' or args.expert_residency_policy != 'LRU')
                and instance_runtime_configs[inst["instance_id"]]["enable_sub_batch_interleaving"]):
            raise ValueError(
                "An expert placement policy other than EVEN, or a residency policy other than "
                "LRU, is not supported with --enable-sub-batch-interleaving: the decision has to "
                "delay both sub-batches, which the interleaved trace cannot express")
        # DP group members share one object (EP spans the group); other instances get their own
        key = inst.get("dp_group") or f'instance{inst["instance_id"]}'
        if key not in shared_placements:
            # DSA needs extra inputs: per-GPU expert latency (from the profiles) and host bandwidth
            dsa = 'DSA' in (args.expert_placement_policy, args.expert_residency_policy)
            extra = _dsa_inputs(inst, model_cfg, num_experts) if dsa else {}
            cls = DSAScheduler if dsa else ExpertPlacement
            if args.expert_residency_policy == 'HOBBIT':   # HOBBIT-style baseline (not combinable with DSA)
                if dsa:
                    raise ValueError('HOBBIT residency cannot be combined with a DSA policy')
                cls, extra = HobbitPolicy, dict(t1=args.hobbit_t1, high_cache=args.hobbit_high_cache)
            shared_placements[key] = cls(
                model_cfg["num_hidden_layers"], num_experts, inst["ep_total"],   # layers, experts, ranks
                # one expert's size; fp is in bits, so // 8 gives bytes per element
                expert_weight_bytes(model_cfg, instance_runtime_configs[inst["instance_id"]]["fp"] // 8),
                policy=args.expert_placement_policy, overhead_ns=placement_overhead_ns,
                interval=args.expert_placement_interval,
                max_swaps=args.expert_placement_max_swaps,
                charge_migration=args.charge_expert_migration,
                memory_fraction=args.expert_memory_fraction,
                residency_policy=args.expert_residency_policy,
                reduced_bytes=args.expert_reduced_bytes,
                reduced_latency=args.expert_reduced_latency,
                prefetch_policy=args.expert_prefetch_policy,
                prefetch_accuracy=args.expert_prefetch_accuracy,
                node_id=inst["node_id"], instance_id=inst["instance_id"], **extra)
        expert_placements[inst["instance_id"]] = shared_placements[key]   # look-up by instance id

    def begin_expert_iteration(inst_id):
        """Let the instance's placement policies decide, just before its next trace is built."""
        placement_state = expert_placements.get(inst_id)
        if placement_state is not None:           # dense models have none
            placement_state.begin_iteration(current)   # current = simulated time now

    # Per-iteration CSV (--iteration-output). ASTRA-Sim reports, per NPU and
    # batch, the current time and that time minus all GPU compute so far, both
    # cumulative. A row runs from when the batch was scheduled to when this NPU
    # finished it; compute is the difference from the NPU's previous report,
    # and the rest is communication, memory loads and synchronization waits
    # (idle time between batches has no compute and falls outside the row).
    iteration_file = iteration_writer = None   # stay None when the flag is not given
    if args.iteration_output:
        # Relative paths are from the repo root; the simulator runs inside astra-sim/
        iteration_path = (args.iteration_output if os.path.isabs(args.iteration_output)
                          else f'../{args.iteration_output}')
        if os.path.dirname(iteration_path):     # create the output folder if needed
            os.makedirs(os.path.dirname(iteration_path), exist_ok=True)
        iteration_file = open(iteration_path, 'w', newline='')
        iteration_writer = csv.writer(iteration_file)
        iteration_writer.writerow([                # header row
            'instance id', 'npu', 'batch id', 'start_ns', 'end_ns', 'duration_ns', 'compute_ns',
            'non_compute_ns',
            'requests', 'prefill', 'decode', 'tokens',
            'expert_loaded_bytes', 'expert_moved_bytes', 'expert_prefetched_bytes',
            'expert_misses', 'placement_decision_ns'])
    batch_stats = {}       # (instance id, batch id) -> batch columns
    last_report = {}       # npu -> cumulative compute_ns at its previous report
    written_reports = set()  # (npu, iteration) already written: a report can be read more than once

    def expert_counters(inst_id):
        """Running expert-traffic totals; their change across one trace = that batch's traffic."""
        pl = expert_placements.get(inst_id)
        if pl is None:                            # dense model: no expert traffic
            return (0.0, 0.0, 0.0, 0)
        return (pl.loaded_bytes, pl.moved_bytes, pl.prefetched_bytes, pl.misses)

    def note_batch(inst_id, batch, before):
        """Record a batch's shape and the expert traffic its trace charged."""
        if iteration_writer is None:              # CSV not requested: keep nothing
            return
        after = expert_counters(inst_id)
        pl = expert_placements.get(inst_id)
        # Kept until ASTRA-Sim reports this batch finished, then written as a row
        batch_stats[(inst_id, batch.batch_id)] = [
            # when the batch was scheduled, and its shape
            batch.batch_time, len(batch.requests), batch.num_prefill, batch.num_decode, batch.total_len,
            *[int(a - b) for a, b in zip(after, before)],   # traffic this batch's trace charged
            pl.decision_ns if pl is not None else 0]        # decision time charged to it

    # ----------------------------------- Start simulation loop ------------------------------------
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()

    # Starting simulation, one while loop processes one iteration
    while True:
        
        out = controller.read_wait(p)
        out_dict = controller.parse_output(out[-2])
        
        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']

        # Route newly arrived requests to instances based on current load
        if dataset is not None:
            router.route_arrived_requests(current)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # New (B8): one CSV row each time an NPU reports a finished batch
        if iteration_writer is not None and out_dict is not None and (sys, id) not in written_reports:
            written_reports.add((sys, id))
            compute_total = out_dict['cycle'] - out_dict['exposed']   # this NPU's compute so far (cumulative)
            prev_compute = last_report.get(sys, 0)                    # ... as of its previous report
            stats = batch_stats.get((instance_id, id - 1))  # ASTRA-Sim iteration id = batch id + 1
            if stats is not None:                 # skip reports with no recorded batch (e.g. start-up)
                start, rest = stats[0], stats[1:]
                duration = out_dict['cycle'] - start       # scheduled -> finished on this NPU
                compute = compute_total - prev_compute     # GPU compute since the previous report
                iteration_writer.writerow([instance_id, sys, id - 1, start, out_dict['cycle'],
                                           duration, compute, duration - compute, *rest])
            last_report[sys] = compute_total      # baseline for this NPU's next report

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Notify router of completed requests for dependency chain release
        if instances[instance_id]["pd_type"] != "prefill":
            for req in finished_reqs:
                router.notify_request_completed(req.id, current)

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(finished_reqs) > 0:
            router.transfer_prefill_request(finished_reqs)

        # schedule requests
        new_req = schedulers[instance_id].schedule(current, sys, id)
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Check if a pre-generated workload is ready for this instance (from DP sync)
        if new_req is None and instance_id in dp_ready_workloads:
            controller.write_flush(p, dp_ready_workloads.pop(instance_id))
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        elif new_req is None and instance_id in inst_dp_group and sys == inst2npu_mapping[instance_id] and len(schedulers[instance_id].inflight) == 0:
            dg = inst_dp_group[instance_id]
            if dp_pending[dg]:
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                dummy.fired.append(sys)
                dp_pending[dg][instance_id] = (dummy, inst2node_mapping[instance_id])

                if len(dp_pending[dg]) == len(dp_groups[dg]):
                    # All DP members accounted for — pad every batch to the
                    # group's max (vLLM CUDA-graph DP padding) and generate.
                    config = get_config(instances[instance_id]["model_name"])
                    max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                    for b, _ in dp_pending[dg].values():
                        _pad_batch_to_max(b, max_total_len)
                    # MoE AG/RS comm size is anchored to ``max_total_len``
                    # (not ``max × group_size``). The trace generator divides
                    # this by ep_total internally for the per-rank AG chunk
                    # and uses the same value for the RS pre-scatter buffer.
                    # Empirically this matches real NCCL AG/RS bandwidth on
                    # PCIe 5.0 at the same ``link_bw`` that already calibrates
                    # AllReduce — i.e. ASTRA-Sim's Ring half-duplex model
                    # ends up correct for AR but 2× over real AG/RS, and the
                    # "× group_size" we used previously stacked the two errors.
                    sum_total_len = max_total_len

                    # Shared workload folder for all DP members
                    first_inst_id = dp_groups[dg][0]
                    first_batch = dp_pending[dg][first_inst_id][0]
                    dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                    # One decision per DP wave (the group shares one placement object)
                    begin_expert_iteration(first_inst_id)
                    for inst_id in dp_groups[dg]:
                        batch, nid = dp_pending[dg][inst_id]
                        inst = instances[inst_id]
                        inst_cfg = instance_runtime_configs[inst_id]
                        counters_before = expert_counters(inst_id)   # traffic totals before this trace
                        generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                       inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                       nid, inst_id,
                                       inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                       placement[inst_id], block_mode_on[inst_id],
                                       expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                       inst_cfg["enable_attn_offloading"],
                                       power_model, pim_models[nid],
                                       inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                       dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                       tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                       dp_sum_total_len=sum_total_len,
                                       enable_block_copy=inst_cfg["enable_block_copy"],
                                       inputs_root=run_paths.inputs_root,
                                       expert_placement=expert_placements.get(inst_id),
                                       routing_trace=routing_trace,
                                       # this member's first global EP rank
                                       ep_rank_offset=inst["dp_rank"] * inst["local_ep"],
                                       moe_rank_cost=args.moe_rank_cost,
                                       ep_hardware=inst.get("ep_hardware"))
                        note_batch(inst_id, batch, counters_before)   # record it for --iteration-output
                        generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                       inst_id, inst2npu_mapping[inst_id],
                                       inst_cfg["enable_local_offloading"],
                                       workload_name=dp_workload_name,
                                       inputs_root=run_paths.inputs_root,
                                       cleanup_trace=args.cleanup_inputs)
                        if inst_id != instance_id:
                            dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                    workload_name=dp_workload_name,
                                                                    inputs_root=run_paths.inputs_root)

                    dp_pending[dg].clear()
                    workload = get_workload(dummy, instances[instance_id]["hardware"], instance_id,
                                            workload_name=dp_workload_name,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
                    responded = True
                else:
                    controller.write_flush(p, "pass")
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            if sys == inst2npu_mapping[instance_id]:  # first NPU of the instance
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id] = (new_req, node_id)

                    if len(dp_pending[dg]) == len(dp_groups[dg]):
                        # All DP members have scheduled — pad every batch to
                        # the group's max (vLLM CUDA-graph DP padding) so
                        # smaller batches gain dummy decodes that all layers
                        # still compute over.
                        config = get_config(instance["model_name"])
                        max_total_len = max(b.total_len for b, _ in dp_pending[dg].values())
                        for b, _ in dp_pending[dg].values():
                            _pad_batch_to_max(b, max_total_len)
                        # See twin block above: anchor MoE comm to max_total_len
                        # (no group-size multiplier).
                        sum_total_len = max_total_len

                        # Shared workload folder for all DP members
                        first_inst_id = dp_groups[dg][0]
                        first_batch = dp_pending[dg][first_inst_id][0]
                        dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                        # One decision per DP wave (the group shares one placement object)
                        begin_expert_iteration(first_inst_id)
                        for inst_id in dp_groups[dg]:
                            batch, nid = dp_pending[dg][inst_id]
                            inst = instances[inst_id]
                            inst_cfg = instance_runtime_configs[inst_id]
                            counters_before = expert_counters(inst_id)   # traffic totals before this trace
                            generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                           inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                           nid, inst_id,
                                           inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                           placement[inst_id], block_mode_on[inst_id],
                                           expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                           inst_cfg["enable_attn_offloading"],
                                           power_model, pim_models[nid],
                                           inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                           dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                           tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                           dp_sum_total_len=sum_total_len,
                                           enable_block_copy=inst_cfg["enable_block_copy"],
                                           inputs_root=run_paths.inputs_root,
                                           expert_placement=expert_placements.get(inst_id),
                                           routing_trace=routing_trace,
                                           # this member's first global EP rank
                                           ep_rank_offset=inst["dp_rank"] * inst["local_ep"],
                                           moe_rank_cost=args.moe_rank_cost,
                                           ep_hardware=inst.get("ep_hardware"))
                            note_batch(inst_id, batch, counters_before)   # record it for --iteration-output
                            generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                           inst_id, inst2npu_mapping[inst_id],
                                           inst_cfg["enable_local_offloading"],
                                           workload_name=dp_workload_name,
                                           inputs_root=run_paths.inputs_root,
                                           cleanup_trace=args.cleanup_inputs)
                            if inst_id != instance_id:
                                dp_ready_workloads[inst_id] = get_workload(batch, inst["hardware"], inst_id,
                                                                        workload_name=dp_workload_name,
                                                                        inputs_root=run_paths.inputs_root)

                        dp_pending[dg].clear()
                        workload = get_workload(new_req, instance["hardware"], instance_id,
                                                workload_name=dp_workload_name,
                                                inputs_root=run_paths.inputs_root)
                        controller.write_flush(p, workload)
                    else:
                        # Waiting for other DP members — send pass
                        controller.write_flush(p, "pass")
                        responded = True
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    begin_expert_iteration(instance_id)              # policies decide before the trace
                    counters_before = expert_counters(instance_id)   # traffic totals before this trace
                    generate_trace(new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                   tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                   enable_block_copy=inst_cfg["enable_block_copy"],
                                   inputs_root=run_paths.inputs_root,
                                   expert_placement=expert_placements.get(instance_id),
                                   routing_trace=routing_trace,
                                   moe_rank_cost=args.moe_rank_cost,
                                   ep_hardware=instance.get("ep_hardware"))
                    note_batch(instance_id, new_req, counters_before)   # record it for --iteration-output
                    generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                   instance_id, inst2npu_mapping[instance_id],
                                   inst_cfg["enable_local_offloading"],
                                   inputs_root=run_paths.inputs_root,
                                   cleanup_trace=args.cleanup_inputs)
                    workload = get_workload(new_req, instance["hardware"], instance_id,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
            elif new_req is not None:
                # Non-first NPU: pick up existing batch workload
                workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                        inputs_root=run_paths.inputs_root)
                controller.write_flush(p, workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * RATIO:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                # len(running), not the size of the in-flight batch: the persistent
                # running set is the exact analogue of vLLM's num_running_reqs, which
                # is what bench compares this column against. The batch is only the
                # subset that fit in this step's token budget.
                running_reqs = len(schedulers[inst_id].running)
                waiting_reqs = len([req for req in schedulers[inst_id].waiting if req.arrival <= current])

                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                    f"Each NPU Memory Usage {npu_used_mb:.2f} MB "
                    f"({npu_util:.3f} % Used)"
                )
                if schedulers[inst_id].enable_prefix_caching:
                    line += schedulers[inst_id].memory.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    inst_usage = []
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = prefix_pools[node_id].used_bytes()
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Used "
                    )
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        line += prefix_pools[node_id].stats.format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

            ######### Per CXL Metrics #########
            if any_prefix_caching and prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.used_bytes()
                        cxl_util = cxl_pool.usage()
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = schedulers[inst_id].memory.storage_pool
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.used_bytes()
                        cxl_util = second_tier.usage()
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty() and not router.has_pending_requests() and not router.has_deferred_sessions():
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                )
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        controller.write_flush(p, "pass")
                    flush.stdout.flush()
                    continue

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["num_npus"] == 1 else 2):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                # check memory leak before exit
                schedulers[inst_idx].memory.is_free()

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                controller.write_flush(p, "exit")
                break
            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None and not responded:
            # If all instances are idle but deferred sessions have pending
            # requests with future arrival times (tool calls still running),
            # advance current time so the next iteration can pick them up.
            if router.has_deferred_sessions() or router.has_pending_requests():
                next_arrival = router.get_next_pending_arrival()
                if next_arrival is not None and next_arrival > current:
                    current = next_arrival
            controller.write_flush(p, "pass")
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)
    if iteration_file is not None:
        iteration_file.close()               # flush the per-iteration CSV to disk

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.stats.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    # total_prompt is the vLLM prompt-throughput gauge: it counts every token
    # pushed through prefill, including prefix-cache hits and anything recomputed
    # after a preemption. Report the dataset input from the requests themselves
    # rather than by subtracting the recompute counter -- a request preempted
    # again mid-recompute is charged its full remaining work each time it is
    # re-admitted, so the two are not each other's complement.
    total_recompute = sum(s.recompute_tokens for s in schedulers)
    total_preempt = sum(s.num_preemptions for s in schedulers)
    total_input = sum(req.original_input for s in schedulers for req in s.done)
    print_markup(f"Total input tokens:                                                 {total_input}")
    if total_preempt:
        print_markup(f"Preemptions:                                                        {total_preempt}")
    if total_recompute:
        print_markup(f"Recomputed prompt tokens (preemption):                               {total_recompute}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {log_interval:g} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    # New summary block, printed only when some expert state could change during the run
    if any(not pl.is_static for pl in shared_placements.values()):
        print_rule("[sim.tagline]Expert Placement Results[/]")
        for key, pl in shared_placements.items():   # one block per instance or DP group
            print_markup(f"{key}: experts moved {pl.total_moves} ({pl.moved_bytes / GB_TO_BYTE:.2f} GB "
                         f"charged), decision time (ms) {pl.total_decision_ns / 1e6:.3f}")
            if pl.tracks_residency:              # memory pressure / precision tiers were active
                # share of (token, expert) pairs served at REDUCED precision (accuracy proxy)
                reduced = pl.reduced_pairs / pl.total_pairs * 100 if pl.total_pairs else 0.0
                print_markup(f"{key}: expert misses {pl.misses}, fetched {pl.fetched_bytes / GB_TO_BYTE:.2f} GB, "
                             f"evictions {pl.evictions}, pairs at reduced precision (%) {reduced:.2f}")
            if pl.prefetch_policy != 'NONE':
                # share of prefetched experts that were then actually used
                useful = pl.useful_prefetches / pl.prefetches * 100 if pl.prefetches else 0.0
                print_markup(f"{key}: prefetched {pl.prefetches} experts "
                             f"({pl.prefetched_bytes / GB_TO_BYTE:.2f} GB), used (%) {useful:.2f}")
            for line in pl.extra_summary():      # policy-specific lines (e.g. DSA)
                print_markup(f"{key}: {line}")
        print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results[/]")
        print_markup(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print_markup(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        if total_requested_tokens > 0:
            print_markup(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
            if prefix_storage != "None":
                print_markup(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
                print_markup(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
            print_markup(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        else:
            print_markup("NPU prefix hit ratio (%):                                           N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {log_interval:g} sec (W): {power_model.power_time_series}")
        print_rule()
    # Each instacne results
    for i in range(num_instances):
        print_rule(f"[sim.tagline]Instance \\[{i}][/]")
        schedulers[i].print_result()
        print_rule()
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)

    if args.cleanup_inputs:
        _cleanup_inputs_root(run_paths, logger)
    

if __name__ == "__main__": 
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
