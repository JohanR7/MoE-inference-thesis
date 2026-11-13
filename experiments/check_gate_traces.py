import sys, numpy as np, json
for old, new in [("outputs/zipf_qwen30.npz", "outputs/zipf_qwen30_g.npz"),
                 ("outputs/zipf_seed1.npz", "outputs/zipf_seed1_g.npz"),
                 ("outputs/zipf_seed2.npz", "outputs/zipf_seed2_g.npz")]:
    a, b = np.load(old), np.load(new)
    meta = json.loads(str(b["meta"]))
    reqs = [k for k in a.files if k.startswith("req_")]
    same = all((np.sort(a[k], axis=-1) == np.sort(b[k], axis=-1)).all() for k in reqs)
    g = b["gates_0"].astype(float)
    print(f"{new}: {len(reqs)} requests, same experts as before: {same}, has_gates: {meta['has_gates']}, "
          f"gate sums ~1: {np.allclose(g.sum(-1), 1, atol=1e-2)}, sorted desc: {(np.diff(g, axis=-1) <= 1e-3).all()}, "
          f"mean top-1 weight {g[..., 0].mean():.2f}")
