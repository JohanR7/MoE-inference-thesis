import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.dsa_scheduler import DSAScheduler
from serving.core.expert_placement import HOST, REDUCED, FULL
L, E = 4, 16
d = np.array([100, 90, 80, 70, 60, 50, 40, 30, 5, 4, 3, 2, 2, 1, 1, 1.])
def run(imb, lat, ep=2, frac=1.0):
    pl = DSAScheduler(L, E, ep, 100.0, policy="DSA", residency_policy="DSA", rank_latency_ns=lat,
                      host_bw=1e9, switch_threshold_ns=0, memory_fraction=frac, reduced_bytes=0.4,
                      overhead_ns=0, quality_fill=True, max_imbalance=imb)
    for _ in range(3):
        for l in range(L): pl.observe(l, d)
        pl.begin_iteration()
        assert (pl._rank_bytes() <= pl.budget + 1e-6).all()
    return pl
eq = run(0.0, [1.0, 2.0]); un = run(0.5, [1.0, 2.0])
print("1 equal counts  : experts per GPU", np.bincount(eq.owner[0], minlength=2).tolist(),
      "| bounds", eq.min_owned, eq.max_owned)
print("  imbalance 0.5 : experts per GPU", np.bincount(un.owner[0], minlength=2).tolist(),
      "| bounds", un.min_owned, un.max_owned)
t = lambda pl: max((np.bincount(pl.owner[0], weights=d, minlength=2) * pl.rank_latency))
print(f"  predicted slowest-GPU time: equal {t(eq):.0f}, imbalance {t(un):.0f}")
assert np.bincount(un.owner[0], minlength=2)[0] > E // 2 and t(un) < t(eq)
# fast GPU owns more than fits at FULL: precision engine fits it (budget asserted inside run)
print("  fast GPU tiers after fitting:", np.bincount(un.tier[0][un.owner[0] == 0], minlength=3).tolist(), "(HOST, REDUCED, FULL)")
# 4 ranks: minimum ownership respected
pl = run(0.5, [1.0, 1.0, 1.0, 4.0], ep=4)
c = np.bincount(pl.owner[0], minlength=4)
print("2 four GPUs, one 4x slower: experts per GPU", c.tolist(), "bounds", pl.min_owned, pl.max_owned)
assert c.min() >= pl.min_owned and c.max() <= pl.max_owned
try:
    DSAScheduler(L, E, 2, 100.0, policy="DSA", residency_policy="LRU", max_imbalance=0.5); raise SystemExit("expected error")
except ValueError as e: print("3 ok: rejected without DSA residency:", str(e)[:60], "...")
print("all ok")
