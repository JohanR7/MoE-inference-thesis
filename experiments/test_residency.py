import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.expert_placement import ExpertPlacement, HOST, REDUCED, FULL
B = 100.0  # bytes per expert
L, E, ep = 4, 8, 2      # 4 experts per rank, 16 per rank over layers

# 1. no pressure: nothing tracked, all FULL
pl = ExpertPlacement(L, E, ep, B)
assert not pl.tracks_residency and pl.is_static and (pl.tier == FULL).all()
print("1 ok: fraction 1.0 tracks nothing")

# 2. preload fills exactly the budget (half of 16 experts per rank = 8)
pl = ExpertPlacement(L, E, ep, B, memory_fraction=0.5)
rb = pl._rank_bytes(); assert (rb == pl.budget).all() and pl.budget == 800, rb
print("2 ok: preload =", rb.tolist(), "budget", pl.budget, "| layer0 tiers", pl.tier[0].tolist())

# 3. miss: route layer 0 to expert 3 (rank 0, HOST after preload?) -> fetched, cached, budget kept
counts = np.zeros(E); counts[[2, 3]] = 5
before = pl.tier[0].copy()
pl.serve(0, counts)
assert (pl.tier[0, [2, 3]] == FULL).all()
newly = int(((before[[2, 3]] == HOST)).sum())
assert pl.misses == newly and pl.take_loads(0, 0) == newly * B
assert (pl._rank_bytes() <= pl.budget + 1e-6).all()
print(f"3 ok: {newly} misses fetched {newly*B:.0f} B, evictions {pl.evictions}, rank bytes {pl._rank_bytes().tolist()}")

# 4. LRU: touch everything resident on rank 0 except one; next miss evicts that one
pl = ExpertPlacement(L, E, ep, B, memory_fraction=0.5)
res = [(l, e) for l in range(L) for e in range(4) if pl.tier[l, e] == FULL]
stale = res[0]
for (l, e) in res[1:]:
    c = np.zeros(E); c[e] = 1; pl.serve(l, c)
target = next((l, e) for l in range(L) for e in range(4) if pl.tier[l, e] == HOST)
c = np.zeros(E); c[target[1]] = 1; pl.serve(target[0], c)
assert pl.tier[stale] == HOST and pl.tier[target] == FULL, (stale, target)
print("4 ok: LRU evicted the stale expert", stale, "to admit", target)

# 5. protect: a layer whose active set exceeds the budget keeps them (streams)
pl = ExpertPlacement(1, 8, 1, B, memory_fraction=0.25)   # budget 2 experts
c = np.ones(8); pl.serve(0, c)
assert (pl.tier[0] == FULL).all(), pl.tier
c2 = np.zeros(8); c2[0] = 1; pl.serve(0, c2)                 # next call evicts down to budget
assert pl._rank_bytes()[0] <= pl.budget + 1e-6
print("5 ok: over-budget active layer streams, then evicted to", pl._rank_bytes().tolist())

# 6. CUSTOM residency: REDUCED share + raise costs fetch, lower is free
pl = ExpertPlacement(L, E, ep, B, residency_policy="CUSTOM", overhead_ns=0)
pl._custom_residency = lambda l, load, owner, tier, now: np.array([FULL, REDUCED, REDUCED, HOST] * 2)
for l in range(L): pl.observe(l, np.ones(E))
pl.begin_iteration()
assert pl.fetched_bytes == 0, pl.fetched_bytes          # all lowered: free
frac = pl.serve(0, np.array([1, 1, 1, 0, 1, 1, 0, 0.]))
print("6 reduced fraction per rank:", frac, "pairs reduced", pl.reduced_pairs, "/", pl.total_pairs)
assert frac == [2/3, 1/2]
pl._custom_residency = lambda l, load, owner, tier, now: np.full(E, FULL)
for l in range(L): pl.observe(l, np.ones(E))
pl.begin_iteration()
# per layer: raise 2 REDUCED + 1 HOST... (HOST at e=3 was a miss? no, 3 had 0 tokens) -> 4 REDUCED + 2 HOST raised per layer
print("  raise to FULL fetched", pl.fetched_bytes, "B")
assert pl.fetched_bytes == L * 6 * B

# 7. move of a HOST expert costs nothing, of a FULL expert costs full bytes
pl = ExpertPlacement(1, 4, 2, B, policy="CUSTOM", memory_fraction=0.5, overhead_ns=0)
pl._custom_placement = lambda l, load, now: np.array([1, 0, 0, 1])   # swap e0 <-> e2? -> e0->1, e2->0
pl.observe(0, np.ones(4)); pl.begin_iteration()
print("7 tiers", pl.tier[0].tolist(), "moved bytes owed per rank", pl.move_due[0].tolist())

# 8. memory model: half the experts -> more KV
from serving.core.memory_model import MemoryModel
import os; os.chdir('astra-sim')
M = "Qwen/Qwen3-30B-A3B-Instruct-2507"
w = [MemoryModel(M, 0, 0, 2, 2, 96, 512, 16, 16, True, False, None, None, ep_size=2,
                 npu_memory_utilization=0.9, expert_memory_fraction=f).weight for f in (1.0, 0.5, 0.25)]
print("8 weight per GPU (GB) at fraction 1.0/0.5/0.25:", [round(x / 2**30, 2) for x in w])
assert w[0] > w[1] > w[2]
print("all ok")
