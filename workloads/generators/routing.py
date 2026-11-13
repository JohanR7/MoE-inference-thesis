"""Expert routing traces for ``python -m serving --expert-routing-policy TRACE``.

A routing trace records, for every request of a workload JSONL, which
experts each token position picked in each MoE layer. The simulator
replays it per scheduled token, so its requests must line up with the
workload's: request ``i`` is the ``i``-th flat line, and an agentic
session takes one id per sub-request, in order (the same ids
``serving/core/router.py`` assigns).

Output is one ``.npz``:

    meta        0-d string array, JSON: model, num_layers, num_experts, top_k, has_gates
    req_<id>    [input_toks + output_toks, num_layers, top_k] expert ids, each token's
                experts ordered from highest to lowest gating weight
    gates_<id>  same shape, float16: the router's gating weight for each chosen expert,
                normalised over the token's top-k (sums to 1), as Qwen3/Mixtral do

Two generators:

**routing-capture** — real router decisions. Runs the HF model over each
request's ``input_tok_ids + output_tok_ids`` in one teacher-forced forward
pass with ``output_router_logits=True`` and keeps the top-k per position.
A causal model routes position ``t`` the same way whether the token was
prefilled or decoded, up to kernel numerics. The workload must come from
the sharegpt generator run with the same ``--model``, so the token ids
belong to this model's tokenizer. Needs torch + transformers (vLLM
container or Kaggle).

**routing-synthetic** — controlled skew without a GPU. Each layer gets a
Zipf popularity over a random expert order (``--skew 0`` is uniform),
reshuffled every ``--phase-reqs`` requests to model drift, and a token
reuses the previous token's experts with probability ``--stickiness``.

Usage:
    python -m workloads.generators routing-capture --model allenai/OLMoE-1B-7B-0924 \
        --dataset workloads/<workload>.jsonl --output workloads/<workload>.routing.npz
    python -m workloads.generators routing-synthetic --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --dataset workloads/<workload>.jsonl --skew 1.0 --output workloads/<workload>.zipf1.npz
"""

from __future__ import annotations   # lets type hints like argparse.ArgumentParser stay unevaluated

import argparse                      # command-line flag definitions
import json                          # workload lines and the trace's meta record are JSON
import os                            # file paths

import numpy as np                   # traces are numpy arrays saved in one .npz


def register_capture_args(p: argparse.ArgumentParser) -> None:
    """Flags of ``routing-capture`` (real router decisions from an HF model)."""
    p.add_argument("--model", required=True,                    # which model's router to record
                   help="HF model id or local path; must expose output_router_logits")
    p.add_argument("--dataset", required=True,                  # the workload to replay later
                   help="workload JSONL with input_tok_ids / output_tok_ids")
    p.add_argument("--output", required=True, help="output .npz path")
    p.add_argument("--num-reqs", type=int, default=0,           # keep in sync with the simulation
                   help="capture only the first N lines (0 = all); match the simulator's --num-reqs")
    p.add_argument("--dtype", default="bfloat16",               # T4/P100 have no bfloat16
                   help="torch dtype for the model (float16 on T4/P100, which lack bfloat16)")
    p.add_argument("--device", default="cuda")                  # GPU by default


def register_synthetic_args(p: argparse.ArgumentParser) -> None:
    """Flags of ``routing-synthetic`` (made-up skewed routing, no GPU needed)."""
    p.add_argument("--model", required=True,                    # only its shape is used
                   help="model name under configs/model/ (supplies layers, experts, top-k)")
    p.add_argument("--dataset", required=True, help="workload JSONL")
    p.add_argument("--output", required=True, help="output .npz path")
    p.add_argument("--num-reqs", type=int, default=0,
                   help="cover only the first N lines (0 = all); match the simulator's --num-reqs")
    p.add_argument("--skew", type=float, default=1.0,           # higher = a few experts get most tokens
                   help="Zipf exponent of per-layer expert popularity (0 = uniform)")
    p.add_argument("--phase-reqs", type=int, default=0,         # models popularity drifting over time
                   help="reshuffle each layer's popularity order every N requests (0 = never)")
    p.add_argument("--stickiness", type=float, default=0.0,     # models neighbouring tokens picking alike
                   help="probability a token reuses the previous token's experts in a layer")
    p.add_argument("--seed", type=int, default=0)               # same seed = same trace


def _iter_requests(path, num_reqs):
    """Yield (request_id, row) with the ids serving/core/router.py assigns."""
    next_id = 0                                  # request ids count up from 0, like the simulator's router
    with open(path) as f:
        for line_no, line in enumerate(f):       # one JSON object per line
            if num_reqs > 0 and line_no >= num_reqs:   # stop after the first num_reqs lines
                break
            row = json.loads(line)
            # An agentic session holds several sub-requests, each getting its own id;
            # a flat request is treated as a session of one
            for sub in row.get("sub_requests", [row]):
                yield next_id, sub
                next_id += 1


def _save(path, arrays, model, num_layers, num_experts, top_k, gates=None):
    """Write all requests' expert choices (and gating weights, if given) plus a
    meta record into one .npz."""
    # Shape information the simulator checks against the model it runs
    meta = json.dumps({"model": model, "num_layers": num_layers,
                       "num_experts": num_experts, "top_k": top_k,
                       "has_gates": gates is not None})
    # Smallest integer type that holds every expert id: 1 byte for up to 256 experts
    dtype = np.uint8 if num_experts <= 256 else np.uint16
    out = {f"req_{rid}": a.astype(dtype) for rid, a in arrays.items()}   # one array per request
    if gates is not None:                                     # gating weights, half precision
        out.update({f"gates_{rid}": g.astype(np.float16) for rid, g in gates.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)   # create the output folder
    np.savez_compressed(path, meta=np.array(meta), **out)               # compressed: traces are large
    print(f"wrote {len(arrays)} requests to {path}")


def run_capture(args) -> int:
    """Record the real router's top-k choices for every token of every request."""
    import torch                                              # only needed here, so imported lazily
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model)           # model shape
    num_layers = config.num_hidden_layers
    # Mixtral-style configs say num_local_experts, Qwen/OLMoE-style num_experts
    num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts")
    top_k = config.num_experts_per_tok                        # experts each token picks
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()   # inference mode

    arrays = {}                                               # request id -> [tokens, layers, top_k]
    gates = {}                                                # request id -> matching gating weights
    with torch.inference_mode():                              # no gradients: faster, less memory
        for rid, row in _iter_requests(args.dataset, args.num_reqs):
            # The whole conversation: prompt tokens followed by the answer tokens
            ids = list(row["input_tok_ids"]) + list(row["output_tok_ids"])
            expected = int(row["input_toks"]) + int(row["output_toks"])
            if len(ids) != expected:                          # token ids must match the counts
                raise ValueError(
                    f"request {rid}: {len(ids)} token ids but input_toks + output_toks = {expected}")
            # One forward pass over all tokens; also return each layer's router scores
            out = model(torch.tensor([ids], device=args.device), output_router_logits=True)
            logits = out.router_logits                        # one entry per MoE layer
            if len(logits) != num_layers:                     # the simulator assumes every layer is MoE
                raise ValueError(
                    f"{args.model} returned router logits for {len(logits)} layers, but "
                    f"num_hidden_layers = {num_layers}. Models with dense layers mixed in "
                    f"(e.g. DeepSeek first_k_dense_replace) are not supported yet.")
            # Each entry is [tokens, num_experts]; softmax is monotonic, so
            # top-k of the logits is the router's top-k.
            # topk returns them sorted, highest first
            top = [l.reshape(len(ids), -1).float().topk(top_k, dim=-1) for l in logits]
            picks = torch.stack([t.indices for t in top], dim=1)          # -> [tokens, layers, top_k]
            # Gating weights: softmax over the chosen experts' logits (top-k normalised)
            weights = torch.stack([t.values.softmax(dim=-1) for t in top], dim=1)
            arrays[rid] = picks.cpu().numpy()                 # move to CPU as a numpy array
            gates[rid] = weights.cpu().numpy()
            if (rid + 1) % 50 == 0:                           # progress message every 50 requests
                print(f"captured {rid + 1} requests")
    _save(args.output, arrays, args.model, num_layers, num_experts, top_k, gates)
    return 0


def run_synthetic(args) -> int:
    """Generate skewed, drifting, token-correlated routing without a GPU."""
    # The model's shape comes from the simulator's own config file
    with open(os.path.join("configs", "model", f"{args.model}.json")) as f:
        config = json.load(f)
    L = config["num_hidden_layers"]                           # layers
    E = config.get("num_local_experts", config.get("num_experts"))   # experts per layer
    k = config["num_experts_per_tok"]                         # experts per token
    rng = np.random.default_rng(args.seed)                    # seeded, so the trace is repeatable

    # Zipf popularity: the r-th most popular expert gets weight 1 / r^skew
    weights = 1.0 / np.arange(1, E + 1) ** args.skew
    log_p = np.log(weights / weights.sum())                   # normalised, as log-probabilities

    def popularity():
        # [L, E] log-probabilities, each layer with its own hot experts
        # (a random permutation decides which expert is most popular in each layer)
        return np.stack([log_p[rng.permutation(E)] for _ in range(L)])

    layer_log_p = popularity()                                # initial popularity
    arrays = {}                                               # request id -> [tokens, layers, k]
    gates = {}                                                # request id -> matching gating weights
    for n, (rid, row) in enumerate(_iter_requests(args.dataset, args.num_reqs)):
        # Drift: every phase_reqs requests, every layer gets a new set of hot experts
        if args.phase_reqs > 0 and n > 0 and n % args.phase_reqs == 0:
            layer_log_p = popularity()
        T = int(row["input_toks"]) + int(row["output_toks"])  # token positions in this request
        # Gumbel top-k: k distinct experts per (token, layer), drawn by popularity
        # (adding Gumbel noise to log-probabilities and taking the top k samples
        # k experts without replacement, in proportion to their popularity)
        scores = layer_log_p[None, :, :] + rng.gumbel(size=(T, L, E))
        # Indices of the k largest scores; argpartition skips a full sort (order among the k doesn't matter)
        picks = np.argpartition(-scores, k - 1, axis=-1)[:, :, :k]
        # Order each token's experts from highest to lowest score, and turn the scores
        # into gating weights (softmax over the chosen k, i.e. top-k normalised).
        # This uses no random numbers, so the chosen experts are the same as before.
        top = np.take_along_axis(scores, picks, axis=-1)
        order = np.argsort(-top, axis=-1)
        picks = np.take_along_axis(picks, order, axis=-1)
        top = np.take_along_axis(top, order, axis=-1)
        g = np.exp(top - top.max(axis=-1, keepdims=True))
        g /= g.sum(axis=-1, keepdims=True)
        if args.stickiness > 0:
            # For each (token, layer), decide whether it copies the previous token's experts
            reuse = rng.random((T, L)) < args.stickiness
            for t in range(1, T):                             # token 0 has no previous token
                picks[t, reuse[t]] = picks[t - 1, reuse[t]]   # copy whole expert sets, so they stay distinct
                g[t, reuse[t]] = g[t - 1, reuse[t]]           # ... with their gating weights
        arrays[rid] = picks
        gates[rid] = g
    _save(args.output, arrays, args.model, L, E, k, gates)
    return 0
