import heapq                          # priority queue: used by POPULARITY to always pick the least-loaded rank
from time import perf_counter_ns      # nanosecond wall clock: used to measure how long a policy decision takes
import numpy as np                    # all per-(layer, expert) state is stored in numpy arrays
from .logger import get_logger        # the simulator's shared logger

# Residency tier of one (layer, expert) on its owner rank. These integer labels
# say where an expert's weights are stored:
#   HOST    = 0: not on the GPU, only in host (CPU) memory; using it is a miss
#   REDUCED = 1: on the GPU in a smaller, quantized form (e.g. int8)
#   FULL    = 2: on the GPU at full model precision
# Ordering matters: a higher number means "more resident", so "new > old" is a tier raise.
HOST, REDUCED, FULL = 0, 1, 2


def expert_weight_bytes(config, fp_bytes):
    """Bytes of one expert's gate, up and down projections."""
    # Width of one expert's feed-forward layer. MoE configs name it
    # moe_intermediate_size; older ones only have intermediate_size.
    moe_ffn = config.get('moe_intermediate_size', config.get('intermediate_size'))
    # An expert has three weight matrices (gate, up, down), each hidden x moe_ffn,
    # each element fp_bytes wide. The router's own weights are counted separately
    # in memory_model, since they are replicated and always resident.
    return 3 * config['hidden_size'] * moe_ffn * fp_bytes


class ExpertPlacement:
    """Per-layer expert -> EP rank map, and where each expert's weights live,
    persisting across iterations.

    The trace generator builds a fresh ``GateRouter`` every iteration, so
    anything a placement policy learns has to live here instead. One object
    is shared by every instance of a DP group, because EP spans the group
    and all members must agree on where each expert lives.

    Lifecycle, once per scheduled batch (or once per DP wave):
        1. ``serving/__main__.py`` calls ``begin_iteration()``. Observations
           from earlier iterations are folded into ``load``, the placement
           policy recomputes ``owner`` and the residency policy ``tier``.
        2. ``GateRouter.route_ep`` reads ``owner[layer]``, reports what it
           routed through ``observe()`` and runs it through ``serve()``.
           Observations only reach the policies at the next
           ``begin_iteration()``, so a policy never sees the routing of the
           iteration it is deciding for.

    Placement policies:
        EVEN        (default) — ``expert_id * ep // num_experts``, the split
                                the simulator always used. Never changes.
        POPULARITY            — greedy longest-processing-time packing of an
                                EMA of observed per-expert token counts, kept
                                only when it lowers the hottest rank's load.
                                Rebuilds the whole map, so it can move up to
                                every expert at once.
        SWAP                  — incremental: up to ``max_swaps`` swaps per
                                layer between the hottest and coldest rank,
                                each the pair that best evens them out. A swap
                                moves exactly two experts.
        CUSTOM                — user-supplied ``_custom_placement``.

    ``interval`` re-runs the policies only every N-th iteration that has
    observations; load is still tracked every iteration.

    Every placement must give each rank exactly ``num_experts // ep_size``
    experts: the expert memory budget is an even share per rank.

    Residency. Each (layer, expert) is FULL, REDUCED or HOST on its owner.
    A rank may hold ``memory_fraction`` of its experts' full-precision bytes
    (the rest of its experts' share goes to KV cache, see ``memory_model``).
    With a fraction below 1, experts start resident in expert-index order
    until the budget is full. Routing a token to a HOST expert is a miss:
    the expert is fetched at FULL before the rank's expert compute and stays
    cached. Whenever a rank is over budget, experts are evicted to HOST,
    except the ones the current layer is running.
        LRU         (default) — only that, evicting the least recently used:
                                a reactive cache at full precision.
        FREQUENCY             — evicts the lowest load EMA instead, and every
                                re-placement refills each rank's budget with
                                its highest-load experts, all layers pooled.
        CUSTOM                — ``_custom_residency`` sets each layer's tiers
                                every re-placement; LRU still evicts on
                                overflow. Raising a tier loads the expert at
                                the new tier; lowering one is free (quantized
                                or dropped in place).
    A REDUCED expert takes ``reduced_bytes`` of the full size, and a rank's
    expert compute is scaled toward ``reduced_latency`` by the share of its
    activated experts that are REDUCED. Neither factor is profiled yet, and
    the accuracy cost of REDUCED is not modelled: ``reduced_pairs`` counts
    the (token, expert) pairs it touched.

    Loads are charged. A moved expert is reloaded on its new rank at its
    tier; misses and tier raises are fetched. The trace generator turns the
    bytes a rank is owed into a weight load in that layer's MoE block, after
    the gate and before the rank's compute (``take_loads``). There is no
    prefetch and no overlap with earlier layers. ``charge_migration=False``
    makes moves free to bound what placement can gain; fetches stay charged.

    Prefetch. After a layer's MoE block, ``plan_prefetch`` may load experts
    of the next layer ahead of time; the trace emits the load right after
    the block, so it overlaps the combine and the next layer's attention,
    and only the next layer's expert compute waits for it. Loads share the
    node's host-memory queue (FIFO) with on-demand fetches.
        NONE        (default) — no prefetch.
        ORACLE                — the next layer's would-be misses, read from
                                the routing that is about to run: the upper
                                bound for a perfect predictor (TRACE only).
                                ``prefetch_accuracy`` < 1 swaps each correct
                                expert, with that probability's complement,
                                for a wrong one, keeping the volume.
        CUSTOM                — ``_custom_prefetch`` names the experts.
    A prefetched expert is admitted at FULL and protected from the eviction
    it causes; a wrong one costs its bandwidth and the residents it evicts.

    Deciding is charged too. ``decision_ns`` is the time ``begin_iteration``
    spent in the policies (measured wall clock, or ``overhead_ns`` when
    set), and the trace generator delays the iteration by it.
    """

    # Allowed names for each kind of policy (checked in __init__)
    # "DSA" is the team's scheduler; its logic lives in dsa_scheduler.DSAScheduler
    _PLACEMENT_POLICIES = ("EVEN", "POPULARITY", "SWAP", "DSA", "CUSTOM")   # which GPU owns each expert
    # QUANTIZED = every expert at REDUCED precision, statically (the "just quantize all" baseline)
    _RESIDENCY_POLICIES = ("LRU", "FREQUENCY", "QUANTIZED", "HOBBIT", "DSA", "CUSTOM")   # which experts stay on the GPU
    _PREFETCH_POLICIES = ("NONE", "ORACLE", "CUSTOM")                # which experts to load early
    # Weight of the history in the load EMA. 0.8 tracks a shift in
    # popularity within ~5 iterations without chasing single-batch noise.
    _EMA_DECAY = 0.8

    def __init__(self, num_layers, num_experts, ep_size, expert_bytes, policy="EVEN",
                 overhead_ns=None, interval=1, max_swaps=1, charge_migration=True,
                 memory_fraction=1.0, residency_policy="LRU",
                 reduced_bytes=0.5, reduced_latency=1.0,
                 prefetch_policy="NONE", prefetch_accuracy=1.0, seed=0,
                 node_id=None, instance_id=None):
        # Arguments:
        #   num_layers        number of MoE layers in the model
        #   num_experts       experts per layer
        #   ep_size           number of EP ranks (GPUs) the experts are spread over
        #   expert_bytes      size of one expert at full precision, in bytes
        #   policy            placement policy: which rank owns each expert
        #   overhead_ns       fixed decision time to charge; None = measure the real one
        #   interval          run the policies only every N-th iteration
        #   max_swaps         SWAP: most swaps per layer per re-placement
        #   charge_migration  False makes moves free (to bound what placement can gain)
        #   memory_fraction   share of a rank's expert bytes that fits on its GPU
        #   residency_policy  which experts stay on the GPU when not all fit
        #   reduced_bytes     size of a REDUCED expert relative to FULL
        #   reduced_latency   compute time of a REDUCED expert relative to FULL
        #   prefetch_policy   which experts of the next layer to load early
        #   prefetch_accuracy ORACLE: chance each correct prediction is kept
        #   seed              random seed (ORACLE's wrong guesses)
        #   node_id/instance_id  only used to tag log messages
        self.L = int(num_layers)                 # number of layers
        self.E = int(num_experts)                # number of experts per layer
        self.ep = max(1, int(ep_size))           # number of EP ranks, at least 1
        if self.E % self.ep != 0:                # experts must split evenly across ranks
            raise ValueError(f"ep_size ({self.ep}) must divide num_experts ({self.E})")
        self.experts_per_rank = self.E // self.ep    # every rank owns this many experts per layer
        # Allowed experts per rank per layer: equal by default; DSA may widen this
        self.min_owned = self.max_owned = self.experts_per_rank
        self.policy = policy.upper()             # placement policy name, case-insensitive
        if self.policy not in self._PLACEMENT_POLICIES:   # reject unknown names early
            raise ValueError(
                f"Unknown expert placement policy {policy!r}. "
                f"Supported: {', '.join(self._PLACEMENT_POLICIES)}")
        self.residency_policy = residency_policy.upper()  # residency policy name
        if self.residency_policy not in self._RESIDENCY_POLICIES:
            raise ValueError(
                f"Unknown expert residency policy {residency_policy!r}. "
                f"Supported: {', '.join(self._RESIDENCY_POLICIES)}")
        self.prefetch_policy = prefetch_policy.upper()    # prefetch policy name
        if self.prefetch_policy not in self._PREFETCH_POLICIES:
            raise ValueError(
                f"Unknown expert prefetch policy {prefetch_policy!r}. "
                f"Supported: {', '.join(self._PREFETCH_POLICIES)}")
        if not 0 <= prefetch_accuracy <= 1:      # a probability must be in [0, 1]
            raise ValueError(f"prefetch accuracy must be in [0, 1], got {prefetch_accuracy}")
        self.prefetch_accuracy = float(prefetch_accuracy)
        self._rng = np.random.default_rng(seed)  # seeded random generator, so runs are repeatable
        if int(interval) < 1 or int(max_swaps) < 1:       # both are counts, so at least 1
            raise ValueError(f"interval ({interval}) and max_swaps ({max_swaps}) must be >= 1")
        if not 0 < memory_fraction <= 1:         # some memory must exist, and at most all experts fit
            raise ValueError(f"expert memory fraction must be in (0, 1], got {memory_fraction}")
        if not 0 < reduced_bytes < 1:            # a reduced expert must be smaller than a full one
            raise ValueError(f"reduced expert bytes must be in (0, 1), got {reduced_bytes}")
        self.interval = int(interval)
        self.max_swaps = int(max_swaps)
        self.charge_migration = bool(charge_migration)
        self.overhead_ns = overhead_ns    # None = measure the policies' wall clock
        self.reduced_latency = float(reduced_latency)
        # Bytes one expert occupies at each tier, indexed by tier: [HOST, REDUCED, FULL]
        self.tier_bytes = np.array([0.0, expert_bytes * reduced_bytes, float(expert_bytes)])
        # Each rank's GPU byte budget for experts: its share of all layers' experts,
        # scaled by memory_fraction (1.0 = everything fits)
        self.budget = memory_fraction * self.experts_per_rank * self.L * expert_bytes
        # Residency only needs tracking when something can leave the GPU.
        self.tracks_residency = memory_fraction < 1 or self.residency_policy != "LRU"
        # Block copy replays one layer for all; only safe when nothing changes.
        self.is_static = self.policy == "EVEN" and not self.tracks_residency

        # Starting placement = EVEN: expert e goes to rank e * ep // E
        # (e.g. 128 experts on 2 ranks: experts 0-63 -> rank 0, 64-127 -> rank 1).
        # min(..., ep - 1) only guards against rounding past the last rank.
        even = np.minimum(np.arange(self.E) * self.ep // self.E, self.ep - 1)
        self.owner = np.tile(even, (self.L, 1))                 # [layer, expert] -> rank (same split in every layer)
        self.tier = np.full((self.L, self.E), FULL, dtype=int)  # [layer, expert] -> tier; all FULL unless memory_fraction < 1
        # [layer, expert] -> "clock" value of its last use; higher = more recent.
        # LRU evicts the smallest value first.
        self.last_used = np.zeros((self.L, self.E), dtype=np.int64)
        self._clock = 0             # ticks once per layer served; stamps last_used
        self._over_budget = False   # a layer's active set overflowed a rank
        self.miss_tier = FULL       # tier a missed expert is fetched at (DSA uses REDUCED)
        if self.residency_policy == "QUANTIZED":
            # Every expert is quantized: start with as many REDUCED experts as fit
            # (all of them when budget allows), and fetch misses at REDUCED too
            self.miss_tier = REDUCED
            self._preload(REDUCED)
        elif memory_fraction < 1:   # not everything fits: start with only some experts resident
            self._preload()

        self.load = np.zeros((self.L, self.E))        # EMA of tokens per expert: the policies' demand estimate
        # Routing counts collected during the current iteration; folded into
        # load at the next begin_iteration (so policies only see the past)
        self._pending = np.zeros((self.L, self.E))
        self._has_pending = False                     # any observation since the last begin_iteration?
        self._seen = np.zeros(self.L, dtype=bool)     # layers with any observation
        self.iteration = 0                # number of iterations whose observations were folded in
        self.decision_ns = 0              # charge for the current iteration
        # Bytes owed to [layer, rank]: loads that rank must do before its next
        # expert compute in that layer. Kept separately so --no-charge-expert-migration
        # can waive moves while fetches stay charged.
        self.move_due = np.zeros((self.L, self.ep))   # owed because an expert was moved onto the rank
        self.fetch_due = np.zeros((self.L, self.ep))  # owed because of a miss or a tier raise

        # Totals for the end-of-run summary. These counters accumulate over the
        # whole simulation and are printed by serving/__main__.py.
        self.total_moves = 0              # experts that changed rank
        self.total_decision_ns = 0        # time charged for policy decisions
        self.moved_bytes = 0.0            # bytes charged for moves
        self.fetched_bytes = 0.0          # bytes fetched for misses, tier raises and refills
        self.loaded_bytes = 0.0           # charged by take_loads: moves, misses, tier raises
        self.misses = 0                   # experts used while on HOST (fetched on demand)
        self.evictions = 0                # experts dropped from the GPU to make room
        self.prefetched = np.zeros((self.L, self.E), dtype=bool)  # loaded ahead, not yet used
        self.prefetches = 0               # experts loaded early
        self.prefetched_bytes = 0.0       # bytes loaded early
        self.useful_prefetches = 0        # early loads that were then actually used
        self.reduced_pairs = 0.0          # (token, expert) pairs served at REDUCED precision
        self.total_pairs = 0.0            # all (token, expert) pairs served
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    # -------------------- per-iteration policy step --------------------

    def observe(self, layer_num, expert_counts):
        """Record the tokens routed to each expert of ``layer_num`` this iteration."""
        self._pending[layer_num] += expert_counts   # add this layer's tokens per expert
        self._has_pending = True                    # tell begin_iteration there is something to fold in

    def begin_iteration(self, current_ns=0):
        """Fold pending observations into ``load`` and re-run the policies.
        Idempotent when nothing was observed since the previous call.

        Iteration N:   begin_iteration() -> routing -> observe() -> serve()
                       -> loads and compute emitted into the trace
        Iteration N+1: begin_iteration() uses iteration N's observations to
                       update placement and residency, and so on.
        """
        self.decision_ns = 0        # nothing charged unless a policy actually runs below
        if not self._has_pending:   # no new observations: nothing to update
            return
        start = perf_counter_ns()   # start timing the decision 
        observed = self._pending.any(axis=1)     # layers that got any tokens this time
        first = observed & ~self._seen           # layers observed for the very first time
        self.load[first] = self._pending[first]  # first observation: take it as the estimate directly
        rest = observed & self._seen             # layers seen before: blend into the history
        # EMA update: new_load = 0.8 * old_load + 0.2 * this iteration's counts
        self.load[rest] = (self._EMA_DECAY * self.load[rest]
                           + (1 - self._EMA_DECAY) * self._pending[rest])
        # Hook: let a subclass (DSA's demand predictor) see this iteration's raw
        # counts before they are cleared. load is already updated at this point.
        self._on_observations(observed)
        self._seen |= observed       # remember which layers now have an estimate
        self._pending[:] = 0         # observations consumed: clear for the next iteration
        self._has_pending = False
        self.iteration += 1          # one more iteration folded in

        # Static setups never change; otherwise only every interval-th iteration re-decides
        if self.is_static or (self.iteration - 1) % self.interval != 0:
            return
        if self.policy == "EVEN" and self.residency_policy in ("LRU", "QUANTIZED", "HOBBIT"):
            return  # LRU / QUANTIZED / HOBBIT act in serve(); nothing to decide here

        moves = 0                                 # experts moved in this decision
        for layer in np.flatnonzero(self._seen):  # every layer that has a load estimate
            if self.policy != "EVEN":             # EVEN never moves anything
                moves += self._replace(layer, current_ns)   # re-place this layer's experts
            if self.residency_policy == "CUSTOM": # user's residency rule: ask for this layer's tiers
                new = np.asarray(self._custom_residency(
                    layer, self.load[layer], self.owner[layer], self.tier[layer].copy(), current_ns))
                self._set_tiers(layer, new)       # apply them, charging any tier raises
        if self.residency_policy == "FREQUENCY":  # FREQUENCY decides across all layers at once
            self._frequency_residency()
        elif self.residency_policy == "DSA":      # so does DSA's precision engine
            self._dsa_residency()
        self._enforce_budget()      # make sure no rank exceeds its memory budget after the changes
        self.total_moves += moves
        # Charge either the measured wall-clock time or the fixed overhead_ns
        self.decision_ns = (perf_counter_ns() - start if self.overhead_ns is None
                            else int(self.overhead_ns))
        self.total_decision_ns += self.decision_ns
        if moves:                   # log only when placement actually changed
            self.logger.info(
                "iteration=%d policy=%s re-placed %d experts (total %d)",
                self.iteration, self.policy, moves, self.total_moves,
            )

    def _replace(self, layer, current_ns):
        """Run the placement policy for ``layer``; moved resident experts are
        reloaded on their new rank at their tier. Returns the move count."""
        if self.policy == "POPULARITY":           # greedy full repack
            new = self._popularity_placement(self.load[layer], self.owner[layer])
        elif self.policy == "SWAP":               # a few incremental swaps
            new = self._swap_placement(self.load[layer], self.owner[layer])
        elif self.policy == "DSA":                # DSA's placement + migration engines
            new = self._dsa_placement(layer)
        else:                                     # CUSTOM: the user's rule
            new = self._custom_placement(layer, self.load[layer], current_ns)
        new = self._validate(np.asarray(new), layer)       # reject malformed placements
        moved = np.flatnonzero(new != self.owner[layer])   # experts whose rank changed
        # Each moved expert must be reloaded on its new rank at its current tier
        # (a HOST expert costs 0 bytes: nothing to move)
        np.add.at(self.move_due[layer], new[moved], self.tier_bytes[self.tier[layer, moved]])
        self.owner[layer] = new     # commit the new placement
        return len(moved)

    def _set_tiers(self, layer, new):
        """Apply a residency decision (one tier per expert) to ``layer``."""
        # Must be one tier per expert, each a valid tier value
        if new.shape != (self.E,) or new.min() < HOST or new.max() > FULL:
            raise ValueError(
                f"Residency for layer {layer} must be {self.E} tiers in [HOST, FULL]")
        raised = np.flatnonzero(new > self.tier[layer])    # experts moved to a higher tier
        # A raised expert has to be fetched at its new tier by its owner rank.
        # Lowered experts cost nothing (quantized or dropped in place).
        np.add.at(self.fetch_due[layer], self.owner[layer, raised],
                  self.tier_bytes[new[raised]])
        self.fetched_bytes += self.tier_bytes[new[raised]].sum()
        self.tier[layer] = new      # commit the new tiers

    # -------------------- runtime: routing hits the experts --------------------

    def serve(self, layer_num, expert_counts, assign=None, gates=None):
        """Run one layer's routing against residency.

        ``assign`` ([tokens, k] expert ids) and ``gates`` (matching gating
        weights) are passed for policies that decide per token; this base
        version only needs ``expert_counts``.

        Misses are fetched at FULL and cached, then the owners are brought
        back under budget without evicting this layer's active experts.
        Returns the share of each rank's activated experts that are REDUCED."""
        active = np.flatnonzero(expert_counts > 0)   # experts that got at least one token
        owner = self.owner[layer_num]                # rank of every expert in this layer
        self._clock += 1                             # advance the LRU clock
        self.last_used[layer_num, active] = self._clock   # mark the active experts as just used
        # Count prefetched experts that turned out to be needed, then clear the marks
        self.useful_prefetches += int(self.prefetched[layer_num, active].sum())
        self.prefetched[layer_num] = False

        missed = active[self.tier[layer_num, active] == HOST]   # needed but not on the GPU
        if len(missed):
            # Fetch each miss at miss_tier (FULL unless a policy says otherwise):
            # its owner rank is owed those bytes
            np.add.at(self.fetch_due[layer_num], owner[missed], self.tier_bytes[self.miss_tier])
            self.tier[layer_num, missed] = self.miss_tier   # the fetched experts are now cached
            self.misses += len(missed)
            self.fetched_bytes += len(missed) * self.tier_bytes[self.miss_tier]
        # Make room if a fetch overflowed a rank (or an earlier layer left one over),
        # but never evict the experts this layer is running
        if len(missed) or self._over_budget:
            self._over_budget = self._enforce_budget(protect=[(layer_num, active)])

        reduced = active[self.tier[layer_num, active] == REDUCED]   # active experts at REDUCED
        self.reduced_pairs += expert_counts[reduced].sum()     # tokens served at reduced precision
        self.total_pairs += expert_counts[active].sum()        # all tokens served
        n_active = np.bincount(owner[active], minlength=self.ep)    # active experts per rank
        n_reduced = np.bincount(owner[reduced], minlength=self.ep)  # of which REDUCED, per rank
        # Share per rank; max(..., 1) avoids dividing by zero on a rank with no active expert.
        # The trace generator scales that rank's expert compute time by this.
        return (n_reduced / np.maximum(n_active, 1)).tolist()

    def plan_prefetch(self, layer_num, expert_counts, next_active=None):
        """Load experts of ``layer_num + 1`` ahead of time, after ``layer_num``
        ran with ``expert_counts``. ``next_active`` is the next layer's real
        expert set, which only ORACLE may read. Returns the bytes each rank
        prefetches."""
        nxt = layer_num + 1                      # the layer being prefetched for
        # Nothing to do: no prefetch policy, past the last layer, or everything is resident anyway
        if self.prefetch_policy == "NONE" or nxt >= self.L or not self.tracks_residency:
            return np.zeros(self.ep)
        if self.prefetch_policy == "ORACLE":     # perfect (or degraded) knowledge of the next layer
            predicted = self._oracle_prefetch(nxt, next_active)
        else:                                    # CUSTOM: the user's predictor
            predicted = np.asarray(self._custom_prefetch(
                layer_num, expert_counts, self.tier[nxt].copy()), dtype=int)
        load = predicted[self.tier[nxt, predicted] == HOST]   # only predicted experts not already on the GPU
        per_rank = np.zeros(self.ep)             # bytes each rank will prefetch
        if len(load):
            # Loaded by their owner at the tier a miss would use (FULL, or REDUCED under
            # DSA), so prefetching never moves bigger copies than fetching on demand
            np.add.at(per_rank, self.owner[nxt, load], self.tier_bytes[self.miss_tier])
            self.tier[nxt, load] = self.miss_tier   # they will be resident when the next layer runs
            self.last_used[nxt, load] = self._clock   # fresh stamp so LRU doesn't evict them at once
            self.prefetched[nxt, load] = True    # remember them, to count useful prefetches later
            self.prefetches += len(load)
            self.prefetched_bytes += len(load) * self.tier_bytes[self.miss_tier]
            # Make room, but never evict what was just predicted
            self._over_budget = self._enforce_budget(protect=[(nxt, predicted)])
        return per_rank

    def _oracle_prefetch(self, layer, next_active):
        """The next layer's would-be misses; with accuracy p each one is kept
        with probability p and otherwise replaced by a non-resident expert
        the layer will not use."""
        if next_active is None:                  # ORACLE needs the real upcoming routing
            raise ValueError("ORACLE prefetch needs the next layer's routing")
        would_miss = next_active[self.tier[layer, next_active] == HOST]   # needed and not resident
        if self.prefetch_accuracy >= 1 or not len(would_miss):   # perfect oracle, or nothing to predict
            return would_miss
        keep = self._rng.random(len(would_miss)) < self.prefetch_accuracy   # each kept with probability p
        # Wrong guesses: experts not on the GPU that the layer will NOT use
        decoys = np.setdiff1d(np.flatnonzero(self.tier[layer] == HOST), next_active)
        # One wrong guess per dropped correct one (as many as decoys allow), so the volume stays the same
        wrong = self._rng.choice(decoys, size=min(int((~keep).sum()), len(decoys)), replace=False)
        return np.concatenate([would_miss[keep], wrong]).astype(int)

    def take_loads(self, layer_num, rank):
        """Bytes ``rank`` must load in ``layer_num`` before its expert compute.
        The caller charges them, so the debt is cleared."""
        moved = self.move_due[layer_num, rank] if self.charge_migration else 0.0   # moves may be waived
        fetched = self.fetch_due[layer_num, rank]   # misses and tier raises are always charged
        self.move_due[layer_num, rank] = 0          # debt paid: clear it
        self.fetch_due[layer_num, rank] = 0
        self.moved_bytes += moved
        self.loaded_bytes += moved + fetched
        return int(moved + fetched)                 # the trace generator turns this into a weight load

    # -------------------- memory budget --------------------

    def _rank_bytes(self):
        """Bytes of experts each rank currently holds on its GPU."""
        # Sum tier_bytes[tier] over all (layer, expert), grouped by owner rank
        return np.bincount(self.owner.ravel(), weights=self.tier_bytes[self.tier].ravel(),
                           minlength=self.ep)

    def _preload(self, tier=FULL):
        """Fill each rank's budget at ``tier`` (FULL unless QUANTIZED), expert
        slot first, then layer, so every layer starts with the same share of residents."""
        self.tier[:] = HOST                      # start with nothing on the GPU
        used = np.zeros(self.ep)                 # bytes filled so far per rank
        for slot in range(self.experts_per_rank):        # the rank's 1st expert, then its 2nd, ...
            for r in range(self.ep):
                e = r * self.experts_per_rank + slot     # EVEN placement
                for layer in range(self.L):              # ... in every layer
                    if used[r] + self.tier_bytes[tier] > self.budget + 1e-6:   # this rank is full
                        break
                    self.tier[layer, e] = tier           # make it resident
                    used[r] += self.tier_bytes[tier]

    def _enforce_budget(self, protect=None):
        """Evict experts to HOST until every rank fits: least recently used
        first, or lowest load EMA first under FREQUENCY (ties: least recent).

        ``protect`` = [(layer, experts), ...] are never evicted: they are
        running now, or were just prefetched. If those alone exceed the budget
        the rank stays over (the layer streams through) and the next ``serve``
        evicts again. Returns whether any rank is still over."""
        rank_bytes = self._rank_bytes()          # current usage per rank
        # 1e-6 tolerates floating-point rounding when a rank is exactly full
        for r in np.flatnonzero(rank_bytes > self.budget + 1e-6):   # each over-budget rank
            candidates = (self.owner == r) & (self.tier != HOST)    # its resident experts
            for layer, experts in protect or ():                    # minus the protected ones
                candidates[layer, experts] = False
            # Non-candidates get the largest stamp so they sort last
            stamps = np.where(candidates, self.last_used, np.iinfo(np.int64).max).ravel()
            if self.residency_policy in ("FREQUENCY", "DSA"):
                # Evict lowest demand first; ties broken by least recent use
                demand = np.where(candidates, self.load, np.inf).ravel()
                order = np.lexsort((stamps, demand))   # lexsort: the last key is the primary one
            else:
                order = np.argsort(stamps, kind="stable")   # LRU: oldest stamp first
            for flat in order[:int(candidates.sum())]:      # walk the candidates in eviction order
                if rank_bytes[r] <= self.budget + 1e-6:     # fits again: stop evicting
                    break
                layer, e = divmod(int(flat), self.E)        # flat index -> (layer, expert)
                rank_bytes[r] -= self.tier_bytes[self.tier[layer, e]]   # free its bytes
                self.tier[layer, e] = HOST                  # evicted to host memory (free to do)
                self.evictions += 1
        return bool((rank_bytes > self.budget + 1e-6).any())   # still over (only protected experts left)?

    # -------------------- policies --------------------

    def _frequency_residency(self):
        """Each rank keeps its highest-load experts at FULL, all layers pooled,
        as many as its budget holds; ties keep what is already resident."""
        slots = int(self.budget // self.tier_bytes[FULL] + 1e-9)   # FULL experts that fit per rank
        load, resident = self.load.ravel(), (self.tier != HOST).ravel()   # flattened over (layer, expert)
        new = np.full(self.L * self.E, HOST)     # default: not resident
        for r in range(self.ep):
            mine = np.flatnonzero(self.owner.ravel() == r)   # all (layer, expert) owned by rank r
            # Sort by load, highest first; among equal loads keep the already resident ones
            # (avoids pointless reloads). lexsort: the last key is the primary one.
            order = np.lexsort((~resident[mine], -load[mine]))
            new[mine[order[:slots]]] = FULL      # the top `slots` become resident
        new = new.reshape(self.L, self.E)        # back to [layer, expert]
        for layer in range(self.L):
            self._set_tiers(layer, new[layer])   # apply; newly resident experts are charged as fetches

    def _popularity_placement(self, load, current):
        """Hottest expert first onto the least-loaded rank that still has room.

        The packing is kept only if it lowers the hottest rank's load below
        what ``current`` gives; otherwise nothing moves. Moves cost a weight
        load each, and greedy packing of a near-uniform load reshuffles
        experts without helping."""
        owner = np.empty(self.E, dtype=int)        # the new placement being built
        heap = [(0.0, r) for r in range(self.ep)]  # (load, rank) of ranks with room
        count = [0] * self.ep                      # experts assigned per rank so far
        loads = load.tolist()                      # plain floats: faster in the loop below
        for e in np.argsort(-load, kind="stable").tolist():   # experts from hottest to coldest
            rank_load, r = heapq.heappop(heap)     # the least-loaded rank with room left
            owner[e] = r
            count[r] += 1
            if count[r] < self.experts_per_rank:   # rank still has room: put it back with its new load
                heapq.heappush(heap, (rank_load + loads[e], r))
        # Load of the busiest rank under a placement
        hottest = lambda o: np.bincount(o, weights=load, minlength=self.ep).max()
        return owner if hottest(owner) < hottest(current) else current   # keep only if it helps

    def _swap_placement(self, load, current):
        """Swap experts between the hottest and coldest rank, ``max_swaps``
        times at most.

        Moving load ``d`` from the hot rank to the cold one helps only for
        ``0 < d < gap``, and evens them out best at ``d = gap / 2``; each
        swap takes the hot/cold expert pair closest to that."""
        owner = current.copy()                     # start from the current placement
        for _ in range(self.max_swaps):
            rank_load = np.bincount(owner, weights=load, minlength=self.ep)   # load per rank
            hot, cold = int(rank_load.argmax()), int(rank_load.argmin())      # busiest and idlest
            gap = rank_load[hot] - rank_load[cold]  # imbalance between them
            on_hot, on_cold = np.flatnonzero(owner == hot), np.flatnonzero(owner == cold)
            # d[i, j] = load moved from hot to cold by swapping on_hot[i] with on_cold[j]
            d = load[on_hot][:, None] - load[on_cold][None, :]
            useful = (d > 0) & (d < gap)            # swaps that actually reduce the imbalance
            if not useful.any():                    # no swap helps: stop early
                break
            score = np.where(useful, np.abs(d - gap / 2), np.inf)   # distance from the ideal d = gap/2
            i, j = np.unravel_index(np.argmin(score), score.shape)  # best pair
            owner[on_hot[i]], owner[on_cold[j]] = cold, hot          # swap them
        return owner

    def _custom_placement(self, layer_num, load, current_ns):
        """Return an owner array of length ``num_experts`` for ``layer_num``.

        ``load`` is the EMA of tokens per expert from earlier iterations.
        Each rank must receive exactly ``num_experts // ep_size`` experts."""
        raise NotImplementedError("Implement custom expert placement.")   # hook for the thesis algorithm

    def _custom_residency(self, layer_num, load, owner, tier, current_ns):
        """Return a tier array (HOST / REDUCED / FULL) of length
        ``num_experts`` for ``layer_num``.

        ``tier`` is the current residency (a copy), ``owner`` the placement
        just decided. ``self.budget`` is each rank's byte budget and
        ``self.tier_bytes[t]`` the bytes of one expert at tier ``t``; LRU
        evicts whatever does not fit."""
        raise NotImplementedError("Implement custom expert residency.")   # hook for the precision engine

    def _custom_prefetch(self, layer_num, expert_counts, next_tier):
        """Return the expert ids of layer ``layer_num + 1`` to load now.

        ``expert_counts`` is the tokens each expert of ``layer_num`` just
        took (the router's choices at this layer), ``next_tier`` the next
        layer's current residency (a copy). Ids already resident cost
        nothing."""
        raise NotImplementedError("Implement custom expert prefetch.")    # hook for the demand predictor

    # -------------------- hooks filled in by DSAScheduler --------------------

    def _on_observations(self, observed):
        """Called with the layers observed this iteration, while ``_pending``
        still holds their raw counts. Nothing to do in the base class."""
        pass

    def _dsa_placement(self, layer):
        """DSA placement; implemented in dsa_scheduler.DSAScheduler."""
        raise NotImplementedError("DSA placement needs DSAScheduler (serving/core/dsa_scheduler.py)")

    def _dsa_residency(self):
        """DSA precision/residency; implemented in dsa_scheduler.DSAScheduler."""
        raise NotImplementedError("DSA residency needs DSAScheduler (serving/core/dsa_scheduler.py)")

    def extra_summary(self):
        """Extra lines for the end-of-run summary (a subclass may add some)."""
        return []

    def _validate(self, owner, layer_num):
        """Check a placement before it is applied; returns it as ints."""
        if owner.shape != (self.E,):             # exactly one rank per expert
            raise ValueError(
                f"Placement for layer {layer_num} has shape {owner.shape}, expected ({self.E},)")
        if owner.min() < 0 or owner.max() >= self.ep:    # every rank id must exist
            raise ValueError(f"Placement for layer {layer_num} names a rank outside [0, {self.ep})")
        counts = np.bincount(owner, minlength=self.ep)   # experts per rank
        # Equal split unless a policy widened [min_owned, max_owned] (DSA --dsa-max-imbalance)
        if (counts < self.min_owned).any() or (counts > self.max_owned).any():
            raise ValueError(
                f"Placement for layer {layer_num} gives ranks {counts.tolist()} experts; "
                f"each rank must hold between {self.min_owned} and {self.max_owned}")
        return owner.astype(int)
