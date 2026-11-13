"""Profile the MoE block of Qwen3-30B-A3B on a CPU, in the simulator's format.

Writes <out>/<hardware>/Qwen/Qwen3-30B-A3B-Instruct-2507/<variant>/{meta.yaml, tp1/moe.csv}
and zips it as <hardware>_profile.zip. Needs only PyTorch; runs on any machine.

The measurement mirrors the GPU profiler's MoE category (profiler/core/categories.py,
hooks/moe_hook.py): the same (tokens, activated_experts) grid, and token i's j-th
expert is (i * top_k + j) % activated_experts, each with weight 1/top_k. One call
= one layer's experts for that batch: group tokens per expert, gate_up projection,
SiLU-and-multiply, down projection, weighted sum back into the output.

    python profile_cpu_moe.py --hardware LAPTOP_CPU
    python profile_cpu_moe.py --hardware KAGGLE_CPU --dtype bfloat16
"""
import argparse, os, platform, statistics, time, zipfile
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def grid(num_experts, top_k, max_tokens):
    """(tokens, activated) pairs: tokens 1..max_tokens in powers of two; activated
    a power of two between top_k and min(num_experts, tokens * top_k)."""
    pts, t = [], 1
    while t <= max_tokens:
        a = 1
        while a <= num_experts:
            if top_k <= a <= min(num_experts, t * top_k):
                pts.append((t, a))
            a *= 2
        t *= 2
    return pts


def moe_forward(x, w13, w2, ids, weights, inter):
    """Reference MoE block: per activated expert, its tokens through gate_up -> SiLU*up -> down."""
    out = torch.zeros_like(x)
    flat_ids = ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)          # group (token, slot) pairs by expert
    sorted_ids = flat_ids[order]
    experts, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    tok = order // ids.shape[1]
    wts = weights.reshape(-1)[order]
    start = 0
    for e, c in zip(experts.tolist(), counts.tolist()):
        idx = tok[start:start + c]
        h = x[idx] @ w13[e].t()                            # [c, 2 * inter]
        a = F.silu(h[:, :inter]) * h[:, inter:]
        y = a @ w2[e].t()                                  # [c, hidden]
        out.index_add_(0, idx, y * wts[start:start + c, None])
        start += c
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hardware", required=True, help="profile folder name, e.g. LAPTOP_CPU")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)       # moe_intermediate_size
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    p.add_argument("--out", default="perf")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    variant = {"bfloat16": "bf16", "float32": "fp32"}[args.dtype]
    torch.manual_seed(0)
    H, I, E, K = args.hidden, args.inter, args.experts, args.top_k
    # Weights of one layer's experts, laid out as in vLLM (w13 = gate and up stacked)
    w13 = (torch.randn(E, 2 * I, H) * 0.02).to(dtype)
    w2 = (torch.randn(E, H, I) * 0.02).to(dtype)
    print(f"CPU: {cpu_name()} | threads {torch.get_num_threads()} | torch {torch.__version__} | {args.dtype}")
    print(f"expert weights: {(w13.numel() + w2.numel()) * w13.element_size() / 1e9:.2f} GB")

    rows = []
    pts = grid(E, K, args.max_tokens)
    t_start = time.time()
    with torch.inference_mode():
        for n, (tokens, act) in enumerate(pts, 1):
            x = torch.randn(tokens, H).to(dtype)
            slot = torch.arange(K)
            ids = (torch.arange(tokens)[:, None] * K + slot[None, :]) % act
            weights = torch.full((tokens, K), 1.0 / K, dtype=dtype)
            for _ in range(args.warmup):
                moe_forward(x, w13, w2, ids, weights, I)
            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                moe_forward(x, w13, w2, ids, weights, I)
                times.append((time.perf_counter() - t0) * 1e6)
            us = statistics.median(times)
            rows.append((tokens, act, us))
            print(f"[{n:2d}/{len(pts)}] tokens={tokens:5d} experts={act:4d}  {us:12.1f} us"
                  f"   ({time.time() - t_start:.0f}s elapsed)", flush=True)

    root = os.path.join(args.out, args.hardware, MODEL, variant)
    os.makedirs(os.path.join(root, "tp1"), exist_ok=True)
    with open(os.path.join(root, "tp1", "moe.csv"), "w") as f:
        f.write("tokens,activated_experts,time_us\n")
        for t, a, us in rows:
            f.write(f"{t},{a},{us:.6g}\n")
    with open(os.path.join(root, "meta.yaml"), "w") as f:
        f.write(f"""profiler_version: cpu-moe-1.0
torch_version: '{torch.__version__}'
gpu: 'CPU: {cpu_name()}'
cpu_threads: {torch.get_num_threads()}
hardware: {args.hardware}
profiled_at: '{datetime.now(timezone.utc).isoformat(timespec="seconds")}'
architecture: qwen3_moe
model: {MODEL}
variant: {variant}
tp_degrees: [1]
note: MoE block only (reference PyTorch implementation on CPU); no dense or attention tables
engine_effective:
  dtype: {args.dtype}
  max_num_batched_tokens: {args.max_tokens}
  max_num_seqs: 256
measurement_iterations: {args.iters}
skew_profile:
  enabled: false
skew_fit:
  enabled: false
""")
    zname = f"{args.hardware}_profile.zip"
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(os.path.join(args.out, args.hardware)):
            for fn in files:
                path = os.path.join(d, fn)
                z.write(path, os.path.relpath(path, args.out))
    print(f"DONE: {root} written, download {zname}")


if __name__ == "__main__":
    main()
