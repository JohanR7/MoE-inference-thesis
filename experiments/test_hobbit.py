import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.hobbit_policy import HobbitPolicy
from serving.core.expert_placement import HOST, REDUCED, FULL
L, E = 4, 16
def fresh(**kw):
    pl = HobbitPolicy(L, E, 2, 100.0, memory_fraction=0.5, residency_policy="HOBBIT", reduced_bytes=0.4, **kw)
    pl.high[:] = False; pl.low[:] = False; pl._sync_tier()      # start empty to see every decision
    return pl

# 1 per-token Eq. 2: token A = (e0 .7, e1 .2), token B = (e2 .5, e3 .4), k=2, T1=0.6
pl = fresh()
assign = np.array([[1, 0], [2, 3]]); gates = np.array([[.2, .7], [.5, .4]])
c = np.bincount(assign.ravel(), minlength=E).astype(float)
red = pl.serve(0, c, assign=assign, gates=gates)
# s: e0=0 FULL, e1=.7 low; e2=0 FULL, e3=.5 FULL (<= .6)
assert pl.high[0, [0, 2, 3]].all() and pl.low[0, 1] and not pl.high[0, 1], (pl.high[0], pl.low[0])
assert pl.high_misses == 3 and pl.low_misses == 1
print("1 ok: per-token precision", "reduced share per rank", red)

# 2 an expert held at high precision serves low-precision tokens without a new load
a = np.array([[2, 0]]); pl.serve(0, np.bincount(a.ravel(), minlength=E).astype(float), assign=a, gates=np.array([[.9, .1]]))   # e0 now unimportant
assert pl.high_misses == 3 and pl.low_misses == 1
print("2 ok: high copy reused for a low-precision token")

# 3 a low copy needed at high precision gets a second (high) copy; LHU only counts high uses
a = np.array([[1, 0]]); pl.serve(0, np.bincount(a.ravel(), minlength=E).astype(float), assign=a, gates=np.array([[.9, .1]]))    # e1 now important
assert pl.high[0, 1] and pl.high_misses == 4
assert pl.high_uses[0, 1] == 1 and pl.uses[0, 1] == 2
print("3 ok: low copy upgraded; uses", pl.uses[0, 1], "high uses", pl.high_uses[0, 1])

# 4 fallback without gates (batch shares) still works
pl2 = fresh()
c = np.zeros(E); c[[0, 1, 2, 3, 8]] = [50, 20, 15, 10, 5]
pl2.serve(0, c)
assert [pl2.tier[0, e] for e in [0, 1, 2, 3, 8]] == [FULL, FULL, REDUCED, REDUCED, REDUCED]
print("4 ok: no-gates fallback")

# 5 each cache stays within its share of the budget; start-up fill respects it too
for split in (0.75, 0.5):
    pl3 = HobbitPolicy(L, E, 2, 100.0, memory_fraction=0.5, residency_policy="HOBBIT",
                       reduced_bytes=0.4, high_cache=split)
    rng = np.random.default_rng(0)
    for step in range(60):
        for l in range(L):
            a = np.stack([rng.choice(E, 2, replace=False) for _ in range(8)])
            g = rng.dirichlet([1, 1], size=8)
            pl3.serve(l, np.bincount(a.ravel(), minlength=E).astype(float), assign=a, gates=g)
            hi = np.bincount(pl3.owner.ravel(), weights=pl3.high.ravel() * pl3.tier_bytes[FULL], minlength=2)
            lo = np.bincount(pl3.owner.ravel(), weights=pl3.low.ravel() * pl3.tier_bytes[REDUCED], minlength=2)
            assert ((hi <= pl3.high_budget + 1e-6).all() and (lo <= pl3.low_budget + 1e-6).all()) or pl3._over_budget
    assert (pl3._rank_bytes() <= pl3.budget + 1e-6).all() or pl3._over_budget
    print(f"5 ok: split {split}: misses {pl3.misses}, evictions {pl3.evictions} |", pl3.extra_summary()[0])
print("all ok")
