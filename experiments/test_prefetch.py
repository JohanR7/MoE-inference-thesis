import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.expert_placement import ExpertPlacement, HOST, FULL
B = 100.0; L, E, ep = 3, 8, 2      # budget 50%: 6 of 12 experts per rank resident
def make(acc=1.0, frac=0.5):
    return ExpertPlacement(L, E, ep, B, memory_fraction=frac, prefetch_policy="ORACLE", prefetch_accuracy=acc)
pl = make()
nxt = np.flatnonzero(pl.tier[1] == HOST)[:3]         # 3 experts of layer 1 that would miss
pf = pl.plan_prefetch(0, np.ones(E), next_active=nxt)
assert (pl.tier[1, nxt] == FULL).all() and pf.sum() == 3 * B, pf
assert (pl._rank_bytes() <= pl.budget + 1e-6).all()
c = np.zeros(E); c[nxt] = 1; pl.serve(1, c)
assert pl.misses == 0 and pl.useful_prefetches == 3
print("1 ok: oracle prefetched", nxt.tolist(), "bytes per rank", pf.tolist(), "-> 0 misses, 3 used")

pl = make(acc=0.0)
nxt = np.flatnonzero(pl.tier[1] == HOST)[:2]
pf = pl.plan_prefetch(0, np.ones(E), next_active=nxt)
c = np.zeros(E); c[nxt] = 1; pl.serve(1, c)
assert pl.prefetches == 2 and pl.useful_prefetches == 0 and pl.misses == 2
print("2 ok: accuracy 0 -> 2 wrong prefetches, 0 used, still 2 misses")

pl = make(frac=1.0)
assert pl.plan_prefetch(0, np.ones(E), next_active=np.arange(E)).sum() == 0
print("3 ok: nothing to prefetch when everything fits")
assert make().plan_prefetch(L - 1, np.ones(E), next_active=np.arange(E)).sum() == 0
print("4 ok: no prefetch past the last layer")
print("all ok")
