import sys, numpy as np
sys.path.insert(0, '.')
from serving.core.dsa_scheduler import DSAScheduler
from serving.core.expert_placement import HOST, REDUCED, FULL
B = 100.0
def obs(pl, counts):                       # feed one iteration's routing to every layer
    for l in range(pl.L): pl.observe(l, counts[l] if counts.ndim == 2 else counts)
    pl.begin_iteration()

# --- Stage B: Hedge ---
rng = np.random.default_rng(0)
a, b = np.array([50, 30, 10, 5, 3, 1, 1, 0.]), np.array([0, 1, 1, 3, 5, 10, 30, 50.])
pl = DSAScheduler(1, 8, 2, B, residency_policy="DSA", overhead_ns=0)
for t in range(30): obs(pl, a if t % 2 == 0 else b)          # flips every iteration: newest is a bad guide
print("B1 alternating routing -> beta (trust in newest):", round(float(pl.beta[0]), 3)); assert pl.beta[0] < 0.2
pl = DSAScheduler(1, 8, 2, B, residency_policy="DSA", overhead_ns=0)
for t in range(30): obs(pl, a if t < 15 else b)              # sudden shift, then stable: newest is a good guide
print("B2 shift then stable  -> beta:", round(float(pl.beta[0]), 3)); assert pl.beta[0] > 0.8

# --- Stages C+D: placement + migration ---
hot = np.array([100, 90, 80, 70, 5, 4, 3, 2.])               # EVEN puts all hot experts on rank 0
pl = DSAScheduler(1, 8, 2, B, policy="DSA", rank_latency_ns=[1.0, 1.0], host_bw=1e9, switch_threshold_ns=0, overhead_ns=0)
obs(pl, hot); obs(pl, hot)
print("C1 equal GPUs, cheap transfers  -> owner", pl.owner[0].tolist(), "accepted", pl.placements_accepted)
per_rank = np.bincount(pl.owner[0], weights=hot, minlength=2); assert per_rank.max() < 200
pl = DSAScheduler(1, 8, 2, B, policy="DSA", rank_latency_ns=[1.0, 1.0], host_bw=1e-9, switch_threshold_ns=0, overhead_ns=0)
obs(pl, hot); obs(pl, hot)
print("C2 equal GPUs, huge transfer cost -> owner", pl.owner[0].tolist(), "rejected", pl.placements_rejected)
assert pl.owner[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
flat = np.array([10.] * 8)
pl = DSAScheduler(1, 8, 2, B, policy="DSA", rank_latency_ns=[1.0, 3.0], host_bw=1e9, switch_threshold_ns=0, overhead_ns=0)
obs(pl, hot[::-1].copy()); obs(pl, hot[::-1].copy())          # hot experts start on the SLOW rank 1
print("C3 rank 1 is 3x slower -> hot experts 4-7 move to rank", pl.owner[0][[7, 6]].tolist())
assert pl.owner[0][7] == 0

# --- Stage E: precision / residency ---
L, E = 4, 16
pl = DSAScheduler(L, E, 2, B, memory_fraction=0.5, residency_policy="DSA", reduced_bytes=0.4, overhead_ns=0)
zipf = 1.0 / np.arange(1, E + 1) ** 1.2 * 100
start = pl.tier.copy()
obs(pl, np.tile(zipf, (L, 1)))
jump = np.abs(pl.tier - start).max()
print("E1 one decision: max tier change", int(jump), "| rank bytes", pl._rank_bytes().tolist(), "budget", pl.budget)
assert jump <= 1 and (pl._rank_bytes() <= pl.budget + 1e-6).all()
for _ in range(5): obs(pl, np.tile(zipf, (L, 1)))
t0 = pl.tier[0]
print("E2 steady state, layer 0 tiers (expert 0 hottest):", t0.tolist())
print("   resident experts per layer:", int((t0 > HOST).sum()), "of", E, "(FULL-only budget would hold", int(pl.budget / B / L), "per rank)")
assert t0[0] == FULL and (t0 > HOST).sum() > E // 2
# misses load at REDUCED
pl2 = DSAScheduler(1, 8, 2, B, memory_fraction=0.5, residency_policy="DSA", overhead_ns=0)
miss = np.flatnonzero(pl2.tier[0] == HOST)[:1]; c = np.zeros(8); c[miss] = 1
pl2.serve(0, c); print("E3 a miss is loaded at tier", int(pl2.tier[0, miss[0]]), "(REDUCED = 1)"); assert pl2.tier[0, miss[0]] == REDUCED
print(pl.extra_summary()[0])
print("all ok")
