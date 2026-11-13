"""HOBBIT-style mixed-precision expert offloading, as a simulator baseline.

Adapted from Tang et al., "HOBBIT: A Mixed Precision Expert Offloading System
for Fast MoE Inference" (arXiv 2411.01433). Selected with
``--expert-residency-policy HOBBIT``.

What is kept from the paper:
  * Token-level dynamic precision (Sec. 3.2, Eq. 2): for every token, its
    top-k experts are ranked by gating weight, and an expert's unimportance
    score s is the summed weight of the experts ranked above it. s <= T1
    means the token needs the high-precision version (default T1 = 0.6, the
    paper's Mixtral value), so a token's first expert always needs it.
  * Two caches (Sec. 3.4): a high-precision and a low-precision cache, each
    with its own share of the memory budget, so an expert can be held in
    both. A token that needs high precision uses the high-precision copy; one
    that needs low precision uses the high-precision copy if present,
    otherwise the low-precision one. Missing copies are loaded into the
    matching cache.
  * Multidimensional eviction (Eq. 3): priority = w_lru * recency +
    w_lfu * frequency + w_lhu * high-precision frequency + w_fld *
    layer-distance term; the lowest priority is evicted from the cache that
    is over budget. Only high-precision uses update the LHU count.

Adaptations to the simulator (it schedules batches, not single tokens):
  * In a batch, an expert is loaded at high precision if any token needs it
    at high precision; otherwise at low precision.
  * Gating weights come from the routing trace (``gates_<id>``). Without
    them, a token's experts are weighted by their share of the batch.
  * Skipping experts (T2) is disabled: it changes model output, and accuracy
    is not simulated, so it would appear as a free speed-up.
  * The cache split is a setting (the paper only says the high-precision
    cache is typically larger); Eq. 3 weights are equal (the paper tunes
    them on calibration data and does not report them); counts are not
    reset per sequence, since a batch mixes many sequences.
  * No prefetching, matching the DSA configuration it is compared with.
"""

import numpy as np                                                  # all state is numpy arrays
from .expert_placement import ExpertPlacement, HOST, REDUCED, FULL     # base class and tier labels


class HobbitPolicy(ExpertPlacement):
    """HOBBIT-style residency on top of ExpertPlacement (see module docstring)."""

    def __init__(self, *args, t1=0.6, high_cache=0.75,
                 weights=(0.25, 0.25, 0.25, 0.25), **kwargs):
        # t1          precision threshold on the unimportance score (paper: 0.6 for Mixtral)
        # high_cache  share of the expert memory budget given to the high-precision cache
        # weights     Eq. 3 weights (w_lru, w_lfu, w_lhu, w_fld); must sum to 1
        super().__init__(*args, **kwargs)
        if not 0 <= t1 <= 1:
            raise ValueError(f"HOBBIT T1 must be in [0, 1], got {t1}")
        if not 0 < high_cache < 1:
            raise ValueError(f"HOBBIT high-precision cache share must be in (0, 1), got {high_cache}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"HOBBIT weights must sum to 1, got {weights}")
        self.t1 = float(t1)
        self.w_lru, self.w_lfu, self.w_lhu, self.w_fld = weights
        # Each rank's byte budget split between the two caches
        self.high_budget = high_cache * self.budget
        self.low_budget = self.budget - self.high_budget
        self.high = np.zeros((self.L, self.E), dtype=bool)   # high-precision copy cached
        self.low = np.zeros((self.L, self.E), dtype=bool)    # low-precision copy cached
        self.uses = np.zeros((self.L, self.E))        # F_t: times each expert was used
        self.high_uses = np.zeros((self.L, self.E))   # H_t: times it was used at high precision
        self.layer_serves = np.zeros(self.L)          # how often each layer has run (normalises F, H)
        self._current_layer = 0                       # l_i in the layer-distance term
        self.high_misses = 0                          # high-precision copies loaded on demand
        self.low_misses = 0                           # low-precision copies loaded on demand
        self._fill_caches()

    # -------------------- start-up --------------------

    def _fill_caches(self):
        """Start with each cache filled in expert-slot order (as the base
        preload): high-precision copies first, then low-precision copies of
        the next experts."""
        for cache, size, budget in ((self.high, self.tier_bytes[FULL], self.high_budget),
                                    (self.low, self.tier_bytes[REDUCED], self.low_budget)):
            used = np.zeros(self.ep)
            for slot in range(self.experts_per_rank):
                for r in range(self.ep):
                    e = r * self.experts_per_rank + slot       # EVEN placement
                    for layer in range(self.L):
                        if self.high[layer, e] or self.low[layer, e]:
                            continue                           # already cached
                        if used[r] + size > budget + 1e-6:
                            break
                        cache[layer, e] = True
                        used[r] += size
        self._sync_tier()

    def _sync_tier(self):
        """``tier`` = best copy held (FULL, else REDUCED, else HOST), for the base class."""
        self.tier[:] = np.where(self.high, FULL, np.where(self.low, REDUCED, HOST))

    def _rank_bytes(self):
        """Bytes each rank holds: its high-precision plus its low-precision copies."""
        held = self.high * self.tier_bytes[FULL] + self.low * self.tier_bytes[REDUCED]
        return np.bincount(self.owner.ravel(), weights=held.ravel(), minlength=self.ep)

    # -------------------- runtime: one layer's routing --------------------

    def _needs_high(self, active, expert_counts, assign, gates):
        """Experts that at least one token needs at high precision (Eq. 2)."""
        if assign is None or gates is None or len(assign) == 0:
            # No per-token gating weights: rank the batch's experts by their share
            share = expert_counts[active] / expert_counts[active].sum()
            ranked = np.argsort(-share, kind="stable")
            s = np.empty(len(active))
            s[ranked] = np.concatenate([[0.0], np.cumsum(share[ranked])[:-1]])
            return set(active[s <= self.t1].tolist())
        order = np.argsort(-gates, axis=1, kind="stable")      # each token's experts, heaviest first
        g = np.take_along_axis(gates, order, axis=1)
        g = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-12)   # normalise per token
        s = np.cumsum(g, axis=1) - g                           # Eq. 2: weight of the experts above
        experts = np.take_along_axis(assign, order, axis=1)
        return set(np.unique(experts[s <= self.t1]).tolist())

    def serve(self, layer_num, expert_counts, assign=None, gates=None):
        """Decide each active expert's precision, load missing copies into the
        matching cache, update the Eq. 3 records, and evict by priority.
        Returns the share of each rank's activated experts served at REDUCED."""
        active = np.flatnonzero(expert_counts > 0)   # experts this layer uses
        owner = self.owner[layer_num]
        self._clock += 1                             # recency clock (R_t)
        self._current_layer = layer_num
        self.last_used[layer_num, active] = self._clock
        self.layer_serves[layer_num] += 1
        self.uses[layer_num, active] += 1

        needs_high = self._needs_high(active, expert_counts, assign, gates)
        served_full = np.zeros(self.E, dtype=bool)   # which experts ran at high precision
        for e in active.tolist():
            if e in needs_high:
                if not self.high[layer_num, e]:      # high-precision miss
                    self.high[layer_num, e] = True
                    self.fetch_due[layer_num, owner[e]] += self.tier_bytes[FULL]
                    self.fetched_bytes += self.tier_bytes[FULL]
                    self.misses += 1
                    self.high_misses += 1
                served_full[e] = True
            elif self.high[layer_num, e]:            # low precision is enough, but the high copy is here
                served_full[e] = True
            elif not self.low[layer_num, e]:         # low-precision miss
                self.low[layer_num, e] = True
                self.fetch_due[layer_num, owner[e]] += self.tier_bytes[REDUCED]
                self.fetched_bytes += self.tier_bytes[REDUCED]
                self.misses += 1
                self.low_misses += 1
        self.high_uses[layer_num, served_full] += 1  # LHU: high-precision uses only

        self._over_budget = self._enforce_budget(protect=[(layer_num, active)])

        reduced = active[~served_full[active]]       # experts that ran at low precision
        self.reduced_pairs += expert_counts[reduced].sum()
        self.total_pairs += expert_counts[active].sum()
        n_active = np.bincount(owner[active], minlength=self.ep)
        n_reduced = np.bincount(owner[reduced], minlength=self.ep)
        return (n_reduced / np.maximum(n_active, 1)).tolist()

    # -------------------- Eq. 3 eviction --------------------

    def _priority(self):
        """Eq. 3 priority of every (layer, expert); the lowest is evicted first."""
        T = max(self._clock, 1)
        serves = np.maximum(self.layer_serves, 1)[:, None]
        p_lru = self.last_used / T                    # recency
        p_lfu = self.uses / serves                    # frequency
        p_lhu = self.high_uses / serves               # high-precision frequency
        # Layer distance: layers about to run next get the highest value
        layers = np.arange(self.L)[:, None]
        p_fld = 1.0 - ((layers - self._current_layer + self.L) % self.L) / self.L
        return (self.w_lru * p_lru + self.w_lfu * p_lfu + self.w_lhu * p_lhu
                + self.w_fld * np.broadcast_to(p_fld, (self.L, self.E)))

    def _enforce_budget(self, protect=None):
        """Evict the lowest-priority copies from whichever cache of a rank is
        over its share of the budget; copies the current layer is running are
        protected. Returns whether any cache is still over."""
        priority = None
        still_over = False
        for cache, size, budget in ((self.high, self.tier_bytes[FULL], self.high_budget),
                                    (self.low, self.tier_bytes[REDUCED], self.low_budget)):
            used = np.bincount(self.owner.ravel(), weights=cache.ravel() * size, minlength=self.ep)
            for r in np.flatnonzero(used > budget + 1e-6):
                if priority is None:
                    priority = self._priority()
                candidates = (self.owner == r) & cache
                for layer, experts in protect or ():
                    candidates[layer, experts] = False
                order = np.argsort(np.where(candidates, priority, np.inf), axis=None, kind="stable")
                for flat in order[:int(candidates.sum())]:
                    if used[r] <= budget + 1e-6:
                        break
                    layer, e = divmod(int(flat), self.E)
                    cache[layer, e] = False
                    used[r] -= size
                    self.evictions += 1
                still_over |= used[r] > budget + 1e-6
        self._sync_tier()
        return bool(still_over)

    def extra_summary(self):
        """HOBBIT lines for the end-of-run summary."""
        return [f"HOBBIT: misses loaded at high precision {self.high_misses}, "
                f"at low precision {self.low_misses}"]
