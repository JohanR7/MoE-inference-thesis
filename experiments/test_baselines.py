import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.expert_placement import ExpertPlacement
rng = np.random.default_rng(0)
E, L, ep = 128, 4, 2
skew = 1.0 / np.arange(1, E + 1) ** 1.2

# SWAP: at most 2*K moves per layer per re-placement, each re-placement lowers the gap
for K in (1, 4):
    pl = ExpertPlacement(L, E, ep, 100.0, "SWAP", max_swaps=K, overhead_ns=0)
    for it in range(6):
        for l in range(L): pl.observe(l, rng.poisson(skew * 500))
        before = pl.total_moves; pl.begin_iteration()
        assert pl.total_moves - before <= 2 * K * L, (K, pl.total_moves - before)
    gap = [np.ptp(np.bincount(pl.owner[l], weights=pl.load[l], minlength=ep)) for l in range(L)]
    print(f"SWAP K={K}: moves {pl.total_moves} over 6 iterations, per-iter max {2*K*L}, final rank gaps {np.round(gap, 1)}")
    assert (np.bincount(pl.owner[0], minlength=ep) == E // ep).all()

# uniform load: no swap is useful, nothing moves
pl = ExpertPlacement(L, E, ep, 100.0, "SWAP", max_swaps=8)
for it in range(3):
    for l in range(L): pl.observe(l, np.full(E, 10.0))
    pl.begin_iteration()
print("SWAP uniform load moves:", pl.total_moves); assert pl.total_moves == 0

# interval: POPULARITY re-places only on iterations 1, 1+N, ...
pl = ExpertPlacement(L, E, ep, 100.0, "POPULARITY", interval=5, overhead_ns=7)
charged = []
for it in range(12):
    for l in range(L): pl.observe(l, rng.poisson(skew[rng.permutation(E)] * 500))
    pl.begin_iteration(); charged.append(pl.decision_ns)
print("interval=5 decision charged on iterations:", [i + 1 for i, c in enumerate(charged) if c])
assert [i + 1 for i, c in enumerate(charged) if c] == [1, 6, 11]
print("all ok")
