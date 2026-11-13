import json                      # new: reads the routing trace's meta record
import random
from dataclasses import dataclass
import numpy as np               # new: per-token routing is done on numpy arrays
from .logger import get_logger


@dataclass
class RoutingResult:
    """Per-EP-rank routing information for a single MoE layer."""
    local_tokens: list     # [rank] -> token count routed to this rank
    activated_experts: list # [rank] -> number of distinct experts activated on this rank
    source_tokens: list    # [rank] -> token count originating from this rank before dispatch
    # The three fields below are new:
    compute_tokens: list   # [rank] -> token count the MoE latency lookup charges (see rank_cost)
    reduced_fraction: list # [rank] -> share of the rank's activated experts at REDUCED precision
    expert_counts: object  # [expert] -> tokens routed to it (numpy array); used for prefetch decisions


class RoutingTrace:
    """Recorded router decisions, replayed by the TRACE routing policy.

    File format: one ``.npz`` holding
        meta          0-d string array, JSON with ``num_layers``,
                      ``num_experts`` and ``top_k`` (``model`` optional)
        req_<id>      int array ``[num_tokens, num_layers, top_k]`` of expert
                      ids for every token position of request ``<id>``

    ``<id>`` is the id ``router.load_requests`` assigns: the line index of a
    flat request in the workload JSONL, with an agentic session taking one
    id per sub-request. Positions cover prompt and output tokens, so a
    request needs ``input_toks + output_toks`` rows.
    """

    def __init__(self, path):
        self.path = path                     # kept for error messages
        self._npz = np.load(path)  # lazy: members are read on first access
        meta = json.loads(str(self._npz["meta"]))   # the shape the trace was recorded for
        self.num_layers = int(meta["num_layers"])
        self.num_experts = int(meta["num_experts"])
        self.top_k = int(meta["top_k"])
        self.has_gates = bool(meta.get("has_gates", False))   # gating weights stored as gates_<id>
        self._requests = {}                  # request id -> its array, cached after first read
        self._gates = {}                     # request id -> its gating weights, cached

    def check(self, num_layers, num_experts, top_k):
        """Fail early if the trace was recorded for a different model shape."""
        got = (self.num_layers, self.num_experts, self.top_k)                   # from the trace
        want = (int(num_layers), int(num_experts), int(top_k))                  # from the model config
        if got != want:
            raise ValueError(
                f"Routing trace {self.path} has (layers, experts, top_k) = {got}, "
                f"but the model config needs {want}")

    def tokens(self, req_id, start, end):
        """Expert ids for positions ``[start, end)`` of request ``req_id``."""
        arr = self._requests.get(req_id)     # already loaded?
        if arr is None:
            key = f"req_{req_id}"            # each request is stored under its own name
            if key not in self._npz.files:   # request missing: trace and workload don't match
                raise ValueError(
                    f"Routing trace {self.path} has no entry for request {req_id}. "
                    f"Capture the trace from the same workload file the simulation runs.")
            arr = self._npz[key]             # decompress it from the .npz (only once)
            self._requests[req_id] = arr
        if end > len(arr):                   # asked for a token position the trace doesn't have
            raise ValueError(
                f"Request {req_id} reached token position {end - 1}, but routing trace "
                f"{self.path} records only {len(arr)} positions for it")
        return arr[start:end]                # [end - start, layers, top_k] expert ids

    def gates(self, req_id, start, end):
        """Gating weights for positions ``[start, end)`` of ``req_id`` (same shape as
        ``tokens``), or None when the trace has none."""
        if not self.has_gates:
            return None
        arr = self._gates.get(req_id)
        if arr is None:
            arr = self._npz[f"gates_{req_id}"].astype(np.float32)   # decompress once
            self._gates[req_id] = arr
        return arr[start:end]


_ROUTING_TRACES = {}   # path -> RoutingTrace; the router is rebuilt every iteration, the trace is not


def load_routing_trace(path):
    """Load a routing trace once per process and reuse it every iteration."""
    if path not in _ROUTING_TRACES:          # first use: open the file
        _ROUTING_TRACES[path] = RoutingTrace(path)
    return _ROUTING_TRACES[path]             # later uses: the same object


class GateRouter:
    """Simulator-side model of the MoE gate + expert dispatch.

    Policies:
        BALANCED  (default) — closed-form pigeonhole approximation of
                              a trained learned gate with load-balancing
                              auxiliary loss. Deterministic.
        RR                  — deterministic round-robin per token.
        RAND                — uniform random per token (seedable).
        TRACE               — replays a recorded ``RoutingTrace``, per
                              request, per token position, per layer.
        CUSTOM              — user-supplied ``_custom_gate_function``.

    Experts are mapped to EP ranks by ``placement`` (an ``ExpertPlacement``),
    falling back to the even split ``expert_owner`` when none is given.

    ``rank_cost`` picks the token count a rank's MoE latency is looked up
    with. The profile times ``tokens`` tokens that each run top-k experts,
    i.e. ``tokens * k`` (token, expert) pairs.
        tokens  (default) — tokens that reached the rank. A token that hits
                            one local expert is charged like one that hits
                            k, so pair imbalance is invisible.
        pairs             — ``ceil(local pairs / k)``: the profiled shot with
                            the same pair count as the rank actually runs.

    ``block_copy``: simulator-side optimization that emits one
    transformer block's trace and replays it ``num_hidden_layers``
    times instead of re-computing the routing every layer. Cuts
    trace-generation time by roughly ``num_hidden_layers`` × on MoE
    models. Safe whenever every layer's routing produces the same
    (local_tokens, activated_experts) pair — which is true for
    BALANCED (deterministic), and a harmless approximation for
    RR / RAND (per-layer variance in activated-count is small once
    the batch is at saturation). Default True for speed; CUSTOM
    policies that legitimately need per-layer variance can set
    ``block_copy=False`` in the constructor. TRACE always disables it:
    recorded routing differs per layer, which is the point of replaying it.
    So does any placement that is not static (a policy other than EVEN,
    or tracked residency), which decides and loads each layer on its own.
    """

    _SUPPORTED_POLICIES = ("BALANCED", "RR", "RAND", "TRACE", "CUSTOM")

    def __init__(
        self,
        node_id,
        instance_id,
        num_local_experts,
        num_experts_per_tok=1,
        routing_policy="BALANCED",
        seed=42,
        block_copy=True,
        placement=None,          # new: ExpertPlacement, the expert -> rank map and residency state
        routing_trace=None,      # new: RoutingTrace, required by TRACE routing
        rank_cost="tokens",      # new: "tokens" (original behaviour) or "pairs"
    ):
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.rnd = random.Random(seed) if seed is not None else random
        self.placement = placement
        # Block copy builds one layer and repeats it, which is only valid when every
        # layer routes and places the same way: not for TRACE (each layer has its own
        # recorded routing), and not when placement or residency can change per layer
        self.block_copy = (bool(block_copy) and self.routing_policy != "TRACE"
                           and (placement is None or placement.is_static))
        # Residency must know which experts each token picks; BALANCED only produces counts
        if (self.routing_policy == "BALANCED" and placement is not None
                and placement.tracks_residency):
            raise ValueError(
                "Expert residency (--expert-memory-fraction < 1 or a CUSTOM residency policy) "
                "needs to know which experts each token picks; BALANCED only gives counts. "
                "Use RR, RAND, TRACE or CUSTOM expert routing.")
        if rank_cost not in ("tokens", "pairs"):     # reject unknown cost modes
            raise ValueError(f"Unknown MoE rank cost {rank_cost!r}. Supported: tokens, pairs")
        self.rank_cost = rank_cost
        self.routing_trace = routing_trace
        self._trace_rows = {}  # id(batch) -> [tokens, layers, k] expert ids
        self._trace_gate_rows = {}  # id(batch) -> [tokens, layers, k] gating weights
        if self.routing_policy == "TRACE" and routing_trace is None:   # TRACE has nothing to replay
            raise ValueError("TRACE expert routing needs a routing trace (--expert-routing-trace)")

        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "BALANCED":
            # ``route_ep`` bypasses ``routing_fn`` for this policy —
            # the per-rank (local_tokens, activated_experts) pair is
            # computed analytically. ``routing_fn`` still points at a
            # valid function for the few non-route_ep call sites.
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_gate_function
        elif self.routing_policy == "TRACE":
            # ``route_ep`` reads the trace directly; ``routing_fn`` only
            # serves ``route``, which has no batch to look positions up in.
            self.routing_fn = self._rand_routing
        else:
            raise ValueError(
                f"Unknown routing_policy {routing_policy!r}. "
                f"Supported: {', '.join(self._SUPPORTED_POLICIES)}"
            )
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    @staticmethod
    def expert_owner(expert_id, ep_size, num_experts):
        """Determine which EP rank owns a given expert. Even distribution across ranks."""
        return min(int(expert_id * ep_size // num_experts), ep_size - 1)

    def _rr_routing(self, token_idx, E, k):
        base = token_idx % E
        return [(base + o) % E for o in range(k)]

    def _rand_routing(self, token_idx, E, k):
        return self.rnd.sample(range(E), k)

    def _custom_gate_function(self, token_idx, E, k):
        raise NotImplementedError("Implement custom gate function.")

    def route(self, layer_num, batch_id, total_len):
        """Returns flat token counts per expert (used when EP=1)."""
        counts = [0] * self.E
        for t in range(int(total_len)):
            exps = self.routing_fn(t, self.E, self.k)
            for e in exps:
                counts[e] += 1

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d batch=%s tokens=%d assigned=%s",
            layer_num, self.routing_policy, self.E, self.k,
            batch_id, total_len, counts,
        )
        return counts

    def route_ep(self, layer_num, batch_id, total_len, ep_size, batch=None):
        """EP-aware routing: returns per-rank token counts and activated experts.

        Tokens are distributed evenly across EP ranks before dispatch
        (matching vLLM's EP execution model). Each token selects k
        experts; the owning rank receives the token for local
        execution. Expert-to-rank assignment comes from ``placement``
        (``owner[layer_num]``), or the even split without one.

        BALANCED short-circuits the per-token draw and uses the
        pigeonhole expression — trained MoE gates (Qwen3, Mixtral,
        DeepSeek, …) are load-balance-regularised so per-expert
        traffic is approximately uniform at serving time. Every policy
        places an equal number of experts on each rank, so the
        pigeonhole result does not depend on which experts those are.

        TRACE needs ``batch`` to know which request positions ran.
        """
        total_len = int(total_len)
        ep_size = max(1, int(ep_size))

        # Distribute source tokens evenly across EP ranks
        base = total_len // ep_size
        remainder = total_len % ep_size
        source_tokens = [base + (1 if r < remainder else 0) for r in range(ep_size)]

        if self.routing_policy == "BALANCED":
            local_tokens, activated_counts = self._balanced_route_ep(
                total_len, ep_size, source_tokens,
            )
            # Uniform routing: every expert gets an equal share of the token-expert pairs
            expert_counts = np.full(self.E, total_len * self.k / self.E)
            rank_pairs = np.full(ep_size, total_len * self.k / ep_size)   # and so does every rank
        else:
            # Per-token routing (RR, RAND, TRACE, CUSTOM): which k experts each token picks
            assign = self._assign_experts(layer_num, total_len, batch)    # [tokens, k] expert ids
            owner = self._owner(layer_num, ep_size)                       # [expert] -> rank, this layer
            local_tokens, activated_counts, expert_counts = self._aggregate(
                assign, owner, ep_size,
            )
            # (token, expert) pairs each rank computes: count the owner rank of every pick
            rank_pairs = np.bincount(owner[assign].ravel(), minlength=ep_size)

        if self.rank_cost == "pairs":
            # The profile charges k pairs per token, so pairs / k = the equivalent token count
            compute_tokens = np.ceil(rank_pairs / self.k).astype(int).tolist()
        else:
            compute_tokens = list(local_tokens)      # original behaviour: tokens that reached the rank

        reduced_fraction = [0.0] * ep_size           # default: nothing runs at reduced precision
        if self.placement is not None:
            self.placement.observe(layer_num, expert_counts)   # record demand for the next decision
            if self.placement.tracks_residency:
                # Check the routing against what is on the GPU: misses, LRU stamps, reduced share
                # Per-token choices and gating weights too, for policies that decide per
                # token (HOBBIT); gating weights only exist for TRACE routing with gates
                gates = self._trace_gates(layer_num, total_len, batch) if self.routing_policy == "TRACE" else None
                reduced_fraction = self.placement.serve(
                    layer_num, expert_counts,
                    assign=None if self.routing_policy == "BALANCED" else assign, gates=gates)

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d ep=%d batch=%s tokens=%d local=%s activated=%s",
            layer_num, self.routing_policy, self.E, self.k, ep_size,
            batch_id, total_len, local_tokens, activated_counts,
        )

        return RoutingResult(
            local_tokens=local_tokens,
            activated_experts=activated_counts,
            source_tokens=source_tokens,
            compute_tokens=compute_tokens,
            reduced_fraction=reduced_fraction,
            expert_counts=expert_counts,
        )

    def _owner(self, layer_num, ep_size):
        """Expert -> EP rank array for ``layer_num``."""
        if self.placement is None:           # no placement state: the original even split
            return np.minimum(np.arange(self.E) * ep_size // self.E, ep_size - 1)
        if self.placement.ep != ep_size:     # placement built for a different number of ranks
            raise ValueError(
                f"Expert placement covers {self.placement.ep} EP ranks, "
                f"but routing was asked for {ep_size}")
        return self.placement.owner[layer_num]   # the policy's current map for this layer

    def _assign_experts(self, layer_num, total_len, batch):
        """Expert ids chosen by each of ``total_len`` tokens: ``[total_len, k]``."""
        if self.routing_policy == "TRACE":       # replay the recorded choices
            return self._trace_assign(layer_num, total_len, batch)
        if self.routing_policy == "RR":          # round robin, vectorised
            return self._rr_block(0, total_len)
        assign = np.empty((total_len, self.k), dtype=int)   # RAND / CUSTOM: ask per token
        for t in range(total_len):
            # Pass the real token index t (the old code always passed 0, which
            # sent every RR token to experts 0..k-1)
            assign[t] = self.routing_fn(t, self.E, self.k)
        return assign

    def _rr_block(self, start, count):
        """Round robin: token t picks experts t, t+1, ..., t+k-1 (mod E)."""
        t = np.arange(start, start + count)[:, None]   # token indices as a column
        return (t + np.arange(self.k)) % self.E        # broadcast to [count, k]

    def _trace_assign(self, layer_num, total_len, batch):
        """Recorded expert ids for the tokens this batch runs in ``layer_num``."""
        if batch is None:                        # positions come from the batch, so it is required
            raise ValueError("TRACE expert routing needs the batch being routed")
        rows = self._trace_rows.get(id(batch))   # built once per batch, reused for every layer
        if rows is None:
            # q_list / k_list are this iteration's (scheduled, computed-before)
            # per request, aligned with batch.requests. The requests' own
            # num_computed_tokens has already moved past this step.
            # For each request: the trace rows of the positions it computes this step,
            # i.e. [already computed, already computed + scheduled)
            parts = [
                self.routing_trace.tokens(req.id, k_before, k_before + q)
                for req, q, k_before in zip(batch.requests, batch.q_list, batch.k_list)
            ]
            # Stack the requests in batch order; an empty batch gets an empty array
            rows = (np.concatenate(parts) if parts
                    else np.empty((0, self.routing_trace.num_layers, self.k), dtype=int))
            self._trace_rows[id(batch)] = rows
        assign = rows[:total_len, layer_num, :].astype(int)   # this layer's picks: [tokens, k]
        # DP padding adds dummy tokens that ran through the router but belong
        # to no request; spread them round-robin.
        pad = total_len - len(assign)            # how many padding tokens there are
        if pad > 0:
            assign = np.concatenate([assign, self._rr_block(0, pad)])
        return assign

    def _trace_gates(self, layer_num, total_len, batch):
        """Gating weights matching ``_trace_assign`` for this layer, or None."""
        if not self.routing_trace.has_gates:
            return None
        rows = self._trace_gate_rows.get(id(batch))
        if rows is None:
            parts = [
                self.routing_trace.gates(req.id, k_before, k_before + q)
                for req, q, k_before in zip(batch.requests, batch.q_list, batch.k_list)
            ]
            rows = (np.concatenate(parts) if parts
                    else np.empty((0, self.routing_trace.num_layers, self.k), dtype=np.float32))
            self._trace_gate_rows[id(batch)] = rows
        g = rows[:total_len, layer_num, :]
        pad = total_len - len(g)                 # padding tokens: equal weights
        if pad > 0:
            g = np.concatenate([g, np.full((pad, self.k), 1.0 / self.k, dtype=np.float32)])
        return g

    def _aggregate(self, assign, owner, ep_size):
        """Per-rank token and activated-expert counts for ``assign`` under ``owner``.

        A token that picks several experts on one rank is sent there once."""
        T = len(assign)                                  # number of tokens
        hits = np.zeros((T, ep_size), dtype=bool)        # hits[t, r]: token t needs rank r
        if T:
            # Mark the owner rank of each of a token's k experts (duplicates collapse to one)
            hits[np.arange(T)[:, None], owner[assign]] = True
        local_tokens = hits.sum(axis=0).tolist()         # tokens each rank receives
        expert_counts = np.bincount(assign.ravel(), minlength=self.E)   # tokens per expert
        # Distinct experts used, grouped by their owner rank
        activated_counts = np.bincount(owner[expert_counts > 0], minlength=ep_size).tolist()
        return local_tokens, activated_counts, expert_counts

    def _balanced_route_ep(self, total_len, ep_size, source_tokens):
        """Closed-form per-rank load for a perfectly-balanced learned gate.

        Pigeonhole model:
          * total expert-token pairs = ``total_len * k``
          * split evenly across EP ranks
          * ``pairs_per_rank       = total_len * k / ep_size``
          * ``activated_per_rank   = min(pairs_per_rank, experts_per_rank)``
            — each owned expert fires as long as there are enough
            pairs to go around; beyond saturation the count is
            capped at ``E / ep_size``.

        Per-rank token count uses the probability a given token hits
        at least one of rank r's experts:

          P(token hits r) = 1 − ((ep − 1) / ep) ** k

        which collapses to ~1 when ``k ≫ ep`` (so every token reaches
        every rank) and degrades gracefully for small-k / large-ep.
        """
        k = self.k
        E_rank = max(1, self.E // ep_size)

        pairs_per_rank = (total_len * k) / ep_size
        activated_per_rank = min(int(round(pairs_per_rank)), E_rank)
        activated_counts = [activated_per_rank] * ep_size

        if ep_size <= 1:
            hit_prob = 1.0
        else:
            hit_prob = 1.0 - ((ep_size - 1) / ep_size) ** k
        tokens_to_r = int(round(total_len * hit_prob))
        local_tokens = [tokens_to_r] * ep_size
        return local_tokens, activated_counts
