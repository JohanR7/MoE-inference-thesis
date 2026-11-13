"""DSA-Aware Resource Scheduling for MoE inference, run inside the simulator.

Implements the team's algorithm (DSA_Aware_MoE_Resource_Scheduling_Algorithm.docx)
on top of ``ExpertPlacement``, which already models placement, residency tiers,
memory budgets, misses, migrations and decision time. Selected with
``--expert-placement-policy DSA`` and/or ``--expert-residency-policy DSA``.

Stages, and how they differ from the document (the "fix" notes):

  A. EWMA profiler      demand x̂ = 0.3 x + 0.7 x̂_prev (the base class's ``load``,
                        with alpha = 0.3 as in the document).
  B. Hedge predictor    blends a short-term signal r (the newest observed routing
                        distribution) and a long-term signal h (the EWMA), with
                        weight beta updated by exp(-eta * cross-entropy).
                        Fix: the document's r is the router softmax of the batch
                        being scheduled, which would be the answer itself; the
                        newest *past* routing is used instead. beta is kept in
                        [0.01, 0.99] so neither signal is ever ignored for good.
  C. Placement          experts from highest predicted demand down, each onto the
                        rank with the lowest load(a) + D(e) * L(a).
                        Fix: the document's D(e) * L(e, a) ignores what a rank
                        already holds, so every expert would pick the fastest GPU;
                        an MoE layer waits for its slowest rank, so balance counts.
  D. Migration          per layer: accept the new placement only if
                        (slowest-rank time saved) x horizon - transfer time > threshold.
                        Fix: measures the layer-level gain, keeps units in ns, and
                        prices transfers at the host link (cpu_mem.mem_bw), which
                        is where the simulator loads expert weights from. A lower
                        bound on the slowest rank's time skips layers where no
                        placement could pass the test (same decisions, less work).
                        Option max_imbalance: ranks may own unequal numbers of experts
                        (bounded by memory at REDUCED size), so a faster GPU can take more.
  E. Precision          sensitivity S = 0.7 D_now + 0.3 D_forecast, bands
                        S >= 0.7 FULL, 0.3-0.7 REDUCED, 0.1-0.3 keep, < 0.1 HOST,
                        at most one tier change per expert per decision, then
                        fit each rank's memory budget (lowest S degraded first).
                        Fixes: demand is normalised to the layer's hottest expert
                        (absolute bands evicted almost everything); the middle
                        band means "REDUCED" (repeated step-downs evicted it);
                        spare memory is filled with the next-best experts one
                        step up; misses are fetched at REDUCED (one step from
                        HOST) since the token has to run now.
                        Option quality_fill: spare memory first brings more experts
                        on at REDUCED, then raises REDUCED experts to FULL (experts
                        already FULL first, so keeping them costs no reload).
  F. Communication      not simulated (point-to-point expert traffic is out of
                        scope for the simulated setup).
"""

import numpy as np                                                # all state is numpy arrays
from .expert_placement import ExpertPlacement, HOST, REDUCED, FULL   # base class and tier labels


class DSAScheduler(ExpertPlacement):
    """The team's scheduler, as an ExpertPlacement subclass (see module docstring)."""

    _EPS = 1e-3     # probability floor mixed into every distribution, so log() never sees 0

    def __init__(self, *args, rank_latency_ns=None, host_bw=64.0, alpha=0.3, hedge_eta=1.0,
                 horizon=20, switch_threshold_ns=100_000, bands=(0.7, 0.3, 0.1),
                 sensitivity_weights=(0.7, 0.3), quality_fill=False, max_imbalance=0.0, **kwargs):
        # Arguments (on top of ExpertPlacement's):
        #   rank_latency_ns      ns per (token, expert) pair on each EP rank, from the GPU
        #                        profiles (stage C/D's L(a)); None = all ranks equal
        #   host_bw              host -> GPU bandwidth in bytes per ns (= GB/s), for transfer time
        #   alpha                stage A smoothing factor (document default 0.3)
        #   hedge_eta            stage B learning rate
        #   horizon              stage D: iterations a placement is expected to last
        #   switch_threshold_ns  stage D: minimum net gain to migrate (document: 0.10 ms)
        #   bands                stage E thresholds (FULL, REDUCED, keep), document defaults
        #   sensitivity_weights  stage E weights of current vs forecast demand
        #   quality_fill         stage E: spend spare memory on FULL precision too (see _dsa_residency)
        #   max_imbalance        stage C: let a rank own up to (1 + x) times its even share of experts
        super().__init__(*args, **kwargs)                 # all the generic state and checks
        if not 0 < alpha <= 1:
            raise ValueError(f"DSA alpha must be in (0, 1], got {alpha}")
        self._EMA_DECAY = 1.0 - alpha                     # base EWMA keeps (1 - alpha) of history
        if rank_latency_ns is None:                       # no profiles given: identical ranks
            rank_latency_ns = [1.0] * self.ep
        self.rank_latency = np.asarray(rank_latency_ns, dtype=float)   # L(a) per rank
        if self.rank_latency.shape != (self.ep,):         # one value per EP rank
            raise ValueError(f"rank_latency_ns needs {self.ep} values, got {len(self.rank_latency)}")
        self.host_bw = float(host_bw)                     # bytes per ns
        self.eta = float(hedge_eta)
        self.horizon = int(horizon)
        self.switch_threshold_ns = float(switch_threshold_ns)
        self.band_full, self.band_reduced, self.band_evict = bands   # e.g. 0.7, 0.3, 0.1
        if not 1 >= self.band_full > self.band_reduced > self.band_evict >= 0:
            raise ValueError(f"DSA bands must satisfy 1 >= full > reduced > evict >= 0, got {bands}")
        self.w_now, self.w_forecast = sensitivity_weights  # 0.7, 0.3 in the document
        self.quality_fill = bool(quality_fill)
        # Unequal ownership: a rank may own (1 +- max_imbalance) x its even share of
        # experts per layer, and never more than its budget holds at REDUCED size, so
        # the precision engine can always fit what it owns (the document's memory
        # feasibility rule). Needs DSA residency to do that fitting.
        if max_imbalance < 0:
            raise ValueError(f"DSA max imbalance must be >= 0, got {max_imbalance}")
        if max_imbalance > 0:
            if self.residency_policy != "DSA":
                raise ValueError("--dsa-max-imbalance needs --expert-residency-policy DSA, "
                                 "which fits the extra experts into memory")
            fits = int(self.budget // (self.L * self.tier_bytes[REDUCED]))   # per layer, at REDUCED
            self.max_owned = max(self.experts_per_rank,
                                 min(self.E, int(self.experts_per_rank * (1 + max_imbalance)), fits))
            self.min_owned = max(int(np.ceil(self.experts_per_rank * (1 - max_imbalance))),
                                 self.E - (self.ep - 1) * self.max_owned, 0)

        # Stage B state, one entry per layer
        self.beta = np.full(self.L, 0.5)                   # trust in the short-term signal; starts neutral
        self.latest = np.zeros((self.L, self.E))           # newest observed distribution (sums to 1)
        self.forecast = np.zeros((self.L, self.E))         # blended prediction for the next iteration
        self._r_pred = np.zeros((self.L, self.E))          # short-term prediction made last time
        self._h_pred = np.zeros((self.L, self.E))          # long-term prediction made last time
        self._has_pred = np.zeros(self.L, dtype=bool)      # whether a prediction exists to score

        self.miss_tier = REDUCED       # fix: a miss loads one tier up from HOST (cheaper, one-step rule)

        # Counters for the end-of-run summary
        self.placements_accepted = 0   # layer re-placements that passed the migration test
        self.placements_rejected = 0   # ... that were proposed but not worth their transfer
        self.placements_pruned = 0     # ... skipped: even a perfect rebalance could not pay off

    # -------------------- stage B: Hedge demand prediction --------------------

    def _norm(self, counts):
        """Counts -> probability distribution over experts, with a small floor."""
        total = counts.sum()
        p = counts / total if total > 0 else np.full(self.E, 1.0 / self.E)   # nothing seen: uniform
        return (1 - self._EPS) * p + self._EPS / self.E    # keep every probability above 0

    def _on_observations(self, observed):
        """Score last iteration's two predictions against what really happened,
        reweight them (Hedge), and predict the next iteration."""
        for l in np.flatnonzero(observed):                 # each layer that got tokens
            y = self._norm(self._pending[l])               # what the router actually did this iteration
            if self._has_pred[l]:                          # there is an earlier prediction to score
                ce_r = -(y * np.log(self._r_pred[l])).sum()   # cross-entropy of the short-term signal
                ce_h = -(y * np.log(self._h_pred[l])).sum()   # cross-entropy of the long-term signal
                w_r = self.beta[l] * np.exp(-self.eta * ce_r)        # lower error -> bigger weight
                w_h = (1 - self.beta[l]) * np.exp(-self.eta * ce_h)
                # New trust in the short-term signal, kept away from 0 and 1 so it can recover
                self.beta[l] = np.clip(w_r / (w_r + w_h), 0.01, 0.99)
            self._r_pred[l] = y                            # short-term: "next looks like the newest"
            self._h_pred[l] = self._norm(self.load[l])     # long-term: the EWMA (already includes y)
            # Blended demand d = beta * r + (1 - beta) * h
            self.forecast[l] = self.beta[l] * self._r_pred[l] + (1 - self.beta[l]) * self._h_pred[l]
            self.latest[l] = y
            self._has_pred[l] = True

    # -------------------- stages C + D: placement and migration --------------------

    def _slowest_rank_ns(self, owner, demand):
        """Predicted MoE time of a layer: the busiest rank's pairs x its ns per pair."""
        per_rank = np.bincount(owner, weights=demand, minlength=self.ep)   # pairs per rank
        return (per_rank * self.rank_latency).max()       # the layer waits for the slowest rank

    def _balanced_assignment(self, demand, current):
        """Hottest expert first onto the rank with the lowest load(a) + D(e) * L(a),
        keeping each rank within [min_owned, max_owned] experts (equal by default);
        ties keep the current owner."""
        # Plain Python lists: with a handful of ranks this is much faster than
        # many tiny numpy operations (this runs for every layer, every decision)
        lat = self.rank_latency.tolist()                  # L(a) per rank
        dem, cur = demand.tolist(), current.tolist()
        acc = [0.0] * self.ep                             # predicted time each rank has so far
        count = [0] * self.ep                             # experts each rank has so far
        owner = [0] * self.E
        remaining = self.E                                # experts not yet assigned
        for e in np.argsort(-demand, kind="stable").tolist():   # highest demand first
            # If the experts left are only just enough to bring every rank up to its
            # minimum, they must go to ranks still below it
            short = [r for r in range(self.ep) if count[r] < self.min_owned]
            must_fill = sum(self.min_owned - count[r] for r in short) >= remaining
            best, best_cost = -1, float("inf")
            for r in range(self.ep):
                if count[r] >= self.max_owned:            # full ranks are not allowed
                    continue
                if must_fill and r not in short:          # reserve the rest for short ranks
                    continue
                cost = acc[r] + dem[e] * lat[r]           # finishing time if e goes to rank r
                # Strictly better wins; an equal cost wins only if it is the current owner
                if cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and r == cur[e]):
                    best, best_cost = r, cost
            owner[e] = best
            acc[best] += dem[e] * lat[best]
            count[best] += 1
            remaining -= 1
        return np.array(owner)

    def _dsa_placement(self, layer):
        """Propose a balanced placement for ``layer`` and keep it only if the
        time it saves over the horizon beats the time its transfers cost."""
        # Predicted (token, expert) pairs per expert next iteration: the forecast's
        # shape scaled to the EWMA's total volume
        demand = self.forecast[layer] * self.load[layer].sum()
        current = self.owner[layer]
        # Shortcut: no placement can make the slowest rank faster than
        #   all work spread perfectly over the ranks' speeds, or the hottest expert on the fastest rank.
        # If even that saving cannot beat the threshold, skip building a candidate.
        t_cur = self._slowest_rank_ns(current, demand)
        lower = max(demand.sum() / (1.0 / self.rank_latency).sum(),
                    demand.max() * self.rank_latency.min())
        if (t_cur - lower) * self.horizon <= self.switch_threshold_ns:
            self.placements_pruned += 1
            return current
        candidate = self._balanced_assignment(demand, current)
        moved = np.flatnonzero(candidate != current)      # experts that would change rank
        if not len(moved):
            return current                                # nothing to decide
        # Stage D: value = slowest-rank time saved per iteration x iterations it lasts
        saved_ns = t_cur - self._slowest_rank_ns(candidate, demand)
        value_ns = saved_ns * self.horizon
        # Transfer time: bytes of the moved experts that are on a GPU (HOST ones cost 0)
        transfer_ns = self.tier_bytes[self.tier[layer, moved]].sum() / self.host_bw
        if value_ns - transfer_ns > self.switch_threshold_ns:   # worth it: migrate
            self.placements_accepted += 1
            return candidate
        self.placements_rejected += 1                     # not worth it: stay put
        return current

    # -------------------- stage E: precision / residency --------------------

    def _sensitivity(self):
        """S = w_now * D_now + w_forecast * D_forecast, each normalised to its
        layer's hottest expert (so S is in [0, 1] and the bands mean the same
        thing for 16 or 128 experts)."""
        S = np.zeros((self.L, self.E))
        for l in np.flatnonzero(self._seen):              # layers with observations
            now = self.latest[l] / self.latest[l].max()
            fc = self.forecast[l] / self.forecast[l].max()
            S[l] = self.w_now * now + self.w_forecast * fc
        return S

    def _dsa_residency(self):
        """Choose every expert's tier: bands, one-step limit, then each rank's budget."""
        S = self._sensitivity()
        start = self.tier.copy()                          # tiers before this decision
        # 1. Bands: what each expert would like to be
        band = start.copy()                               # 0.1 <= S < 0.3: keep current
        band[S >= self.band_full] = FULL
        band[(S >= self.band_reduced) & (S < self.band_full)] = REDUCED
        band[S < self.band_evict] = HOST
        band[~self._seen] = start[~self._seen]            # layers never observed: leave alone
        # 2. One-step rule: at most one tier up or down from where each expert was
        floor = np.maximum(start - 1, HOST)
        ceil = np.minimum(start + 1, FULL)
        target = np.clip(band, floor, ceil)
        ceil_one_step = ceil.copy()                       # quality fill may go up to this
        # Spare memory may keep an expert resident, but at most at REDUCED unless its
        # band asks for FULL: filling must not undo the bands' precision choices
        ceil = np.minimum(ceil, np.maximum(band, REDUCED))
        # 3. Fit each rank's memory budget, all its layers pooled
        tb = self.tier_bytes
        flat_t, flat_S = target.ravel(), S.ravel()        # views: editing flat_t edits target
        flat_floor, flat_ceil = floor.ravel(), ceil.ravel()
        flat_ceil1, flat_start = ceil_one_step.ravel(), start.ravel()
        owner = self.owner.ravel()
        for r in range(self.ep):
            mine = np.flatnonzero(owner == r)             # every (layer, expert) on rank r
            used = tb[flat_t[mine]].sum()                 # bytes the targets would take
            if used > self.budget:
                # Over budget: lower the least sensitive experts first, one tier at a time
                order = mine[np.argsort(flat_S[mine], kind="stable")]
                changed = True
                while used > self.budget and changed:     # repeat passes until it fits or nothing can drop
                    changed = False
                    for i in order:
                        if used <= self.budget:
                            break
                        if flat_t[i] > flat_floor[i]:     # may still go down one tier
                            used -= tb[flat_t[i]] - tb[flat_t[i] - 1]
                            flat_t[i] -= 1
                            changed = True
            elif self.quality_fill:
                # Spare memory, quality fill. Pass 1, coverage: bring the most sensitive
                # experts onto the GPU at REDUCED (fewer misses).
                for i in mine[np.argsort(-flat_S[mine], kind="stable")]:
                    if flat_t[i] == HOST and flat_ceil1[i] >= REDUCED:
                        extra = tb[REDUCED]
                        if used + extra <= self.budget:
                            flat_t[i] = REDUCED
                            used += extra
                # Pass 2, quality: raise REDUCED experts to FULL. Experts that were FULL
                # before this decision go first (keeping them costs no reload), then by
                # sensitivity. lexsort: the last key is the primary one.
                order = mine[np.lexsort((-flat_S[mine], flat_start[mine] != FULL))]   # False (was FULL) sorts first
                for i in order:
                    if flat_t[i] == REDUCED and flat_ceil1[i] == FULL:
                        extra = tb[FULL] - tb[REDUCED]
                        if used + extra <= self.budget:
                            flat_t[i] = FULL
                            used += extra
            else:
                # Spare memory: raise the most sensitive experts one tier while they fit
                for i in mine[np.argsort(-flat_S[mine], kind="stable")]:
                    if flat_t[i] < flat_ceil[i]:
                        extra = tb[flat_t[i] + 1] - tb[flat_t[i]]
                        if used + extra <= self.budget:
                            flat_t[i] += 1
                            used += extra
        # 4. Apply: raises are charged as fetches, lowerings are free
        for layer in range(self.L):
            self._set_tiers(layer, target[layer])

    # -------------------- reporting --------------------

    def extra_summary(self):
        """DSA lines for the end-of-run summary."""
        seen = self._seen
        beta = self.beta[seen].mean() if seen.any() else 0.5
        return [f"DSA: re-placements accepted {self.placements_accepted}, "
                f"rejected (not worth the transfer) {self.placements_rejected}, "
                f"skipped (could not pay off) {self.placements_pruned}, "
                f"mean Hedge beta (trust in newest routing) {beta:.2f}"]
