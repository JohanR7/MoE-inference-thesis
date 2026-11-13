import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.expert_placement import ExpertPlacement, HOST, FULL
B = 100.0; L, E, ep = 2, 8, 2          # 4 experts/rank/layer, 8 per rank overall; budget 50% = 4
pl = ExpertPlacement(L, E, ep, B, memory_fraction=0.5, residency_policy="FREQUENCY", overhead_ns=0)
load = np.array([[1, 9, 2, 8, 7, 1, 6, 0], [5, 0, 3, 4, 0, 9, 2, 8]], float)
for l in range(L): pl.observe(l, load[l])
pl.begin_iteration()
for r in range(ep):
    mine = [(l, e) for l in range(L) for e in range(E) if pl.owner[l, e] == r]
    want = sorted(mine, key=lambda x: -load[x])[:4]
    got = [x for x in mine if pl.tier[x] == FULL]
    assert sorted(got) == sorted(want), (r, got, want)
print("1 ok: refill keeps each rank's top-4 by load; fetched", pl.fetched_bytes, "B, rank bytes", pl._rank_bytes().tolist())
f0 = pl.fetched_bytes
for l in range(L): pl.observe(l, load[l])
pl.begin_iteration()
assert pl.fetched_bytes == f0, "steady load must not churn"
print("2 ok: steady load -> no new fetches")
# miss on a cold expert: evicts the lowest-load resident on that rank, not the least recent
r0_res = [(l, e) for l in range(L) for e in range(E) if pl.owner[l, e] == 0 and pl.tier[l, e] == FULL]
coldest = min(r0_res, key=lambda x: pl.load[x])
c = np.zeros(E); c[coldest[1]] = 1; pl.serve(coldest[0], c)        # touch the coldest: most recent now
miss = next((l, e) for l in range(L) for e in range(E) if pl.owner[l, e] == 0 and pl.tier[l, e] == HOST)
c = np.zeros(E); c[miss[1]] = 1; pl.serve(miss[0], c)
assert pl.tier[coldest] == HOST and pl.tier[miss] == FULL, (coldest, miss)
print("3 ok: miss evicted the lowest-load expert", coldest, "even though it was just used")
print("all ok")
