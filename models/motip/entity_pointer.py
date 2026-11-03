import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# The single mask convention of this module: anything not a real score is
# written as exactly -1.0e4, so "is there a real value here?" is the strict test
# `value > INVALID_LOGIT` (real scores live in the +-sqrt(d)*logit_scale band).
INVALID_LOGIT = -1.0e4


class CausalEntityPointer(nn.Module):
    """Resolve each current observation to a prior entity mention or NEW.

    The returned tensor keeps MOTIP's vocabulary-shaped interface only as a
    sparse address table.  There is no learned classifier weight per identity:
    each entity logit is a causal pointer score aggregated over that entity's
    past mentions, so renaming identity slots merely permutes output addresses.

    Aggregation modes
    ------------------
    "max"        – original behaviour: s(i,e) = max_j a_ij.  A single mention
                   can dominate; this is the P0 baseline.
    "logsumexp"  – Module A: s(i,e) = tau * (logsumexp_j(a_ij/tau) - log|M_e|).
                   At low tau this approximates max (smooth init from a max
                   checkpoint); as tau grows the aggregation softens toward the
                   mean, but the -log|M_e| correction keeps it from degenerating
                   to the unconditional H1 mean prototype.  tau is learnable via
                   log_tau so the model can discover the right temperature.
    """

    def __init__(
        self,
        feature_dim: int,
        num_id_vocabulary: int,
        aggregation_mode: str = "max",
        init_tau: float = 0.5,
        consensus_residual_scale: float = 2.0,
        entity_state_residual_scale: float = 2.0,
        entity_state_hidden_dim: int = 64,
        id_memory_dim: int = 128,
        new_head_context: bool = False,
        new_context_init: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.aggregation_mode = aggregation_mode
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        # E19c scale-freedom fix (architecture-audit ladder step 1):
        # learnable scalar s on the bounded cos*sqrt(d) logits.  Init
        # 1.0 keeps step-0 geometry identical to E5/E19b; s restores the
        # one-dimensional logit scale freedom that the official
        # unbounded linear head gets for free.
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.new_pointer = nn.Linear(feature_dim, 1)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.key_norm = nn.LayerNorm(feature_dim)
        self._last_mention_audit = None

        if aggregation_mode == "logsumexp":
            # Learnable log-temperature; exp() guarantees tau > 0.
            # Initialised to log(init_tau) so that low tau ≈ max at start,
            # enabling smooth initialisation from a max-aggregation checkpoint.
            self.log_tau = nn.Parameter(torch.tensor(float(math.log(init_tau))))
        if aggregation_mode == "consensus_residual":
            # Module A v2 never replaces P0 max.  It learns only a bounded,
            # shared residual from causal top-k mention agreement.  A zero last
            # layer makes checkpoint bootstrap exactly P0 at step zero.
            self.consensus_residual_scale = float(consensus_residual_scale)
            self.consensus_mlp = nn.Sequential(
                nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 1),
            )
            nn.init.zeros_(self.consensus_mlp[-1].weight)
            nn.init.zeros_(self.consensus_mlp[-1].bias)
        if aggregation_mode in ("entity_state", "equivariant_memory"):
            # CRR-E1: build a query-conditional entity state from strictly past
            # mentions.  The shared modules never see the numeric identity slot,
            # so renaming slots only permutes the output columns.
            # CRR-E5 (equivariant_memory) inherits all E1 components as the
            # appearance channel P_ptr and adds an equivariant identity-memory
            # channel P_id behind a zero-initialised gate.
            self.entity_state_residual_scale = float(entity_state_residual_scale)
            self.entity_state_time_bias = nn.Embedding(8, 1)
            self.entity_state_reliability = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, entity_state_hidden_dim),
                nn.GELU(),
                nn.Linear(entity_state_hidden_dim, 1),
            )
            self.entity_state_residual = nn.Sequential(
                nn.Linear(10, entity_state_hidden_dim),
                nn.GELU(),
                nn.Linear(entity_state_hidden_dim, 1),
            )
            self.log_entity_state_tau = nn.Parameter(torch.tensor(0.0))
            nn.init.zeros_(self.entity_state_time_bias.weight)
            # E19c cold-start fix (peer round-2 verdict): the reliability
            # and residual last layers keep their DEFAULT init.  Zero-init
            # was a warm-start bootstrap constraint (step-0 == P0) that
            # only delays module wake-up when training from scratch.
        if aggregation_mode == "equivariant_memory":
            # CRR-E5: generative identity embedding.  h_e is produced by a
            # shared GRUCell recurrence over entity e's OWN mention history
            # (dense [V, T] grid, gaps masked to no-ops), so h_e is a function
            # of the entity's own observations only -- the Deep-Sets-style
            # equivariance constraint.  The dual channel
            #     P = P_ptr + g * (P_id - P_ptr)
            # uses a zero-initialised gate (ControlNet zero-conv principle):
            # at step zero g == 0 exactly, so the output is bit-identical to
            # the E1 appearance channel and checkpoint bootstrap is safe.
            self.id_memory_dim = int(id_memory_dim)
            self.id_memory_gru = nn.GRUCell(feature_dim, self.id_memory_dim)
            self.id_memory_proj = nn.Linear(self.id_memory_dim, feature_dim, bias=False)
            self.id_gate = nn.Sequential(
                nn.Linear(4, entity_state_hidden_dim),
                nn.GELU(),
                nn.Linear(entity_state_hidden_dim, 1),
            )
            # E19c cold-start fix: id_gate last layer keeps its DEFAULT
            # init, so the id-memory channel is awake from step 0.  tanh
            # gating is kept: negative gates are a legitimate CE solution
            # (peer round-2 rebuttal 3).

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.new_pointer.weight)
        nn.init.zeros_(self.new_pointer.bias)

        # E19d (2): context-aware NEW head.  `new_pointer` sees only the query
        # feature, so the NEW logit stays at magnitude O(1) while every entity
        # logit is a cos*sqrt(d)*logit_scale score at magnitude ~20.  The two
        # therefore live on different scales, NEW can never win, and the
        # measured pointer_new_fraction collapses to ~0.004.
        #     s_new = new_pointer(x) + lam * logmeanexp_e(s_e)
        # puts NEW in the same reference frame as the entity it competes with.
        # lam is zero-initialised, so step 0 is bit-identical to E19c; unlike a
        # zero-product gate its gradient dL/dlam = sum_e (p_new - y) * s_ctx is
        # order 20 * 0.5 != 0, so it wakes up within a few hundred steps.
        self.new_head_context = bool(new_head_context)
        if self.new_head_context:
            self.new_context_scale = nn.Parameter(
                torch.tensor(float(new_context_init))
            )

    @property
    def tau(self):
        if self.aggregation_mode == "logsumexp":
            return self.log_tau.exp()
        return None

    def forward(self, seq_info, use_decoder_checkpoint=False):
        del use_decoder_checkpoint
        history = seq_info["trajectory_features"]
        current = seq_info["unknown_features"]
        history_labels = seq_info["trajectory_id_labels"]
        history_times = seq_info["trajectory_times"]
        current_times = seq_info["unknown_times"]
        history_masks = seq_info["trajectory_masks"]
        current_masks = seq_info["unknown_masks"]
        current_labels = seq_info.get("unknown_id_labels")

        b, g, _, _, c = history.shape
        current_shape = current.shape[:-1]
        history_flat = history.reshape(b * g, -1, c)
        current_flat = current.reshape(b * g, -1, c)
        labels_flat = history_labels.reshape(b * g, -1)
        history_times_flat = history_times.reshape(b * g, -1)
        current_times_flat = current_times.reshape(b * g, -1)
        history_masks_flat = history_masks.reshape(b * g, -1)

        queries = F.normalize(self.query_norm(self.query_proj(current_flat)), dim=-1)
        keys = F.normalize(self.key_norm(self.key_proj(history_flat)), dim=-1)
        mention_scores = torch.matmul(queries, keys.transpose(1, 2)) * math.sqrt(self.feature_dim) * self.logit_scale

        # Strictly causal: a current mention may only point to an earlier one.
        invalid = history_masks_flat[:, None, :] | (
            history_times_flat[:, None, :] >= current_times_flat[:, :, None]
        )
        mention_scores = mention_scores.masked_fill(invalid, -1.0e4)

        if os.environ.get("MOTIP_RECORD_ATTRIBUTION_V2", "0") == "1":
            self._last_mention_audit = self._build_mention_audit(
                mention_scores=mention_scores, labels_flat=labels_flat,
                history_times_flat=history_times_flat,
                current_times_flat=current_times_flat, invalid=invalid,
            )
        else:
            self._last_mention_audit = None

        entity_scores = self._aggregate_mentions(
            mention_scores=mention_scores,
            labels_flat=labels_flat,
            invalid=invalid,
            n_current=current_flat.shape[1],
            history_times_flat=history_times_flat,
            current_times_flat=current_times_flat,
            queries=queries,
            keys=keys,
            history_flat=history_flat,
            history_masks_flat=history_masks_flat,
            history_grid_shape=(history.shape[2], history.shape[3]),
        )
        new_score = self.new_pointer(current_flat)
        if self.new_head_context:
            new_score = new_score + self.new_context_scale * self._entity_context(entity_scores)
        logits = torch.cat([entity_scores, new_score], dim=-1)
        logits = logits.reshape(*current_shape, self.num_id_vocabulary + 1)
        return logits, current_labels, current_masks

    def _entity_context(self, entity_scores):
        """log-mean-exp over the valid entity logits, per query.

        `entity_scores` is [bg, N, V] with every "no evidence" column already
        filled to INVALID_LOGIT, so validity is the strict `> INVALID_LOGIT`
        test.  Squashing the masked -1.0e4 columns through logsumexp would
        contribute exp(-1e4 - max) == 0 anyway; the mask only matters for the
        count.  When a query has no valid entity at all the context is 0, i.e.
        the head falls back to the plain `new_pointer` score.
        """
        valid = entity_scores > INVALID_LOGIT
        n_valid = valid.sum(dim=-1, keepdim=True).to(entity_scores.dtype)
        log_mean_exp = (
            torch.logsumexp(entity_scores, dim=-1, keepdim=True)
            - torch.log(n_valid.clamp(min=1.0))
        )
        return torch.where(
            valid.any(dim=-1, keepdim=True),
            log_mean_exp,
            torch.zeros_like(log_mean_exp),
        )

    def pop_last_mention_audit(self):
        audit = self._last_mention_audit
        self._last_mention_audit = None
        return audit

    @torch.no_grad()
    def _build_mention_audit(self, mention_scores, labels_flat,
                             history_times_flat, current_times_flat, invalid):
        """Record actual top-scoring causal mentions; never reads GT."""
        audits = []
        for batch_index in range(mention_scores.shape[0]):
            batch_rows = []
            for query_index in range(mention_scores.shape[1]):
                valid = ~invalid[batch_index, query_index]
                labels = labels_flat[batch_index]
                candidates = []
                for label in torch.unique(labels[valid]).tolist():
                    label = int(label)
                    if label < 0 or label >= self.num_id_vocabulary:
                        continue
                    idx = torch.nonzero(valid & (labels == label), as_tuple=False).flatten()
                    values = mention_scores[batch_index, query_index, idx]
                    best_local = int(torch.argmax(values).item())
                    best_idx = int(idx[best_local].item())
                    age = int(current_times_flat[batch_index, query_index].item()
                              - history_times_flat[batch_index, best_idx].item())
                    candidates.append({
                        "label": label,
                        "history_count": int(idx.numel()),
                        "best_mention_score": float(values[best_local].item()),
                        "best_mention_age": age,
                    })
                batch_rows.append(candidates)
            audits.append(batch_rows)
        return audits

    # ------------------------------------------------------------------ #
    #  Entity-level score aggregation                                     #
    # ------------------------------------------------------------------ #
    def _aggregate_mentions(
        self, mention_scores, labels_flat, invalid, n_current,
        history_times_flat=None, current_times_flat=None,
        queries=None, keys=None, history_flat=None,
        history_masks_flat=None, history_grid_shape=None,
    ):
        """Reduce per-mention scores to per-entity scores.

        mention_scores : [bg, N, H]  query-mention similarity (already masked)
        labels_flat    : [bg, H]     entity label for each history mention
        invalid        : [bg, N, H]  True for non-causal / padded mentions
        n_current      : int         number of current-frame queries
        """
        bg = mention_scores.shape[0]
        address = labels_flat[:, None, :].expand(-1, n_current, -1)
        valid_address = address.clamp(min=0, max=self.num_id_vocabulary - 1)

        if self.aggregation_mode == "max":
            entity_scores = mention_scores.new_full(
                (bg, n_current, self.num_id_vocabulary), -1.0e4,
            )
            entity_scores.scatter_reduce_(
                dim=-1,
                index=valid_address,
                src=mention_scores,
                reduce="amax",
                include_self=True,
            )
            return entity_scores

        elif self.aggregation_mode == "logsumexp":
            return self._logsumexp_aggregate(
                mention_scores, valid_address, invalid, bg, n_current,
            )

        elif self.aggregation_mode == "consensus_residual":
            if history_times_flat is None or current_times_flat is None:
                raise ValueError("consensus_residual requires causal times")
            return self._consensus_residual_aggregate(
                mention_scores, valid_address, invalid, bg, n_current,
                history_times_flat, current_times_flat,
            )

        elif self.aggregation_mode == "entity_state":
            required = (history_times_flat, current_times_flat, queries, keys, history_flat)
            if any(value is None for value in required):
                raise ValueError("entity_state requires causal times, queries, keys, and history features")
            return self._entity_state_aggregate(
                mention_scores=mention_scores,
                valid_address=valid_address,
                invalid=invalid,
                queries=queries,
                keys=keys,
                history_flat=history_flat,
                history_times_flat=history_times_flat,
                current_times_flat=current_times_flat,
            )

        elif self.aggregation_mode == "equivariant_memory":
            required = (
                history_times_flat, current_times_flat, queries, keys, history_flat,
                history_masks_flat, history_grid_shape,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "equivariant_memory requires causal times, queries, keys, "
                    "history features, history masks, and the history grid shape"
                )
            return self._equivariant_memory_aggregate(
                mention_scores=mention_scores,
                valid_address=valid_address,
                labels_flat=labels_flat,
                invalid=invalid,
                queries=queries,
                keys=keys,
                history_flat=history_flat,
                history_times_flat=history_times_flat,
                history_masks_flat=history_masks_flat,
                current_times_flat=current_times_flat,
                history_grid_shape=history_grid_shape,
            )

        else:
            raise ValueError(
                f"Unknown aggregation mode: {self.aggregation_mode}"
            )

    def _equivariant_memory_aggregate(
        self, mention_scores, valid_address, labels_flat, invalid, queries, keys,
        history_flat, history_times_flat, history_masks_flat, current_times_flat,
        history_grid_shape,
    ):
        """CRR-E5 equivariant identity memory: P = P_ptr + g * (P_id - P_ptr).

        P_ptr is the full E1 entity-state aggregation (appearance channel).
        P_id comes from a generative identity embedding h_e: a shared GRUCell
        recurrence over entity e's own strictly-past mentions.  The gate g is
        produced by a zero-initialised MLP, so at initialisation g == 0 and the
        output is exactly the E1 appearance channel (safe bootstrap).
        Numeric identity labels are used only as scatter addresses.

        The aggregation materialises [bg, Nq, V, c] and [bg, Nq, H, c] tensors
        which dominate activation memory on long multi-entity sequences; the
        whole dual-channel aggregation is therefore computed in bg-row chunks
        under gradient checkpointing during training (bit-identical outputs,
        transient per-chunk recompute in backward).
        """
        bg = mention_scores.shape[0]
        chunk = int(os.environ.get("E5_BG_CHUNK", "4"))
        outputs = []
        for start in range(0, bg, chunk):
            stop = min(start + chunk, bg)
            sl = slice(start, stop)
            chunk_args = (
                mention_scores[sl], valid_address[sl], labels_flat[sl],
                invalid[sl], queries[sl], keys[sl], history_flat[sl],
                history_times_flat[sl], history_masks_flat[sl],
                current_times_flat[sl], history_grid_shape,
            )
            if self.training and torch.is_grad_enabled():
                outputs.append(
                    grad_checkpoint(
                        self._equivariant_memory_chunk, *chunk_args,
                        use_reentrant=False,
                    )
                )
            else:
                outputs.append(self._equivariant_memory_chunk(*chunk_args))
        return torch.cat(outputs, dim=0)

    def _equivariant_memory_chunk(
        self, mention_scores, valid_address, labels_flat, invalid, queries, keys,
        history_flat, history_times_flat, history_masks_flat, current_times_flat,
        history_grid_shape,
    ):
        """One bg-row chunk of the E5 dual-channel aggregation."""
        p_ptr = self._entity_state_aggregate(
            mention_scores=mention_scores,
            valid_address=valid_address,
            invalid=invalid,
            queries=queries,
            keys=keys,
            history_flat=history_flat,
            history_times_flat=history_times_flat,
            current_times_flat=current_times_flat,
        )

        bg, n_current = mention_scores.shape[0], mention_scores.shape[1]
        n_entities = self.num_id_vocabulary
        valid = ~invalid
        valid_float = valid.to(mention_scores.dtype)
        count = mention_scores.new_zeros((bg, n_current, n_entities))
        count.scatter_add_(dim=-1, index=valid_address, src=valid_float)

        # Generative identity embedding per (query, entity).  Queries at
        # different times receive different prefixes of the same recurrence,
        # preserving strict causality for every query.
        h_q = self._entity_memory_prefixes(
            history_flat=history_flat,
            labels_flat=labels_flat,
            history_times_flat=history_times_flat,
            history_masks_flat=history_masks_flat,
            current_times_flat=current_times_flat,
            history_grid_shape=history_grid_shape,
        )   # [bg, n_current, V, id_memory_dim]
        id_keys = F.normalize(self.id_memory_proj(h_q), dim=-1)
        id_scores = torch.einsum(
            "bqd,bqvd->bqv", queries.to(id_keys.dtype), id_keys,
        ) * math.sqrt(self.feature_dim)

        # Zero-initialised dual-channel gate on bounded pairwise features.
        scale = math.sqrt(self.feature_dim)
        gate_features = torch.stack([
            p_ptr.clamp(min=-2 * scale, max=2 * scale) / (2 * scale),
            id_scores.clamp(min=-2 * scale, max=2 * scale) / (2 * scale),
            (id_scores - p_ptr).clamp(min=-4 * scale, max=4 * scale) / (4 * scale),
            torch.log1p(count.clamp(min=0.0)).clamp(max=math.log(31.0)) / math.log(31.0),
        ], dim=-1)
        gate = torch.tanh(self.id_gate(gate_features.to(mention_scores.dtype)).squeeze(-1))
        output = p_ptr + gate * (id_scores.to(mention_scores.dtype) - p_ptr)
        return output.masked_fill(count <= 0, -1.0e4)

    def _entity_memory_prefixes(
        self, history_flat, labels_flat, history_times_flat, history_masks_flat,
        current_times_flat, history_grid_shape,
    ):
        """Per-entity GRU memory over the [T, N] history grid.

        The flat mention list is scattered into a dense [V, T] grid keyed by
        (entity label, time row).  A shared GRUCell runs over the T rows; rows
        without an observation for that entity are masked to no-ops, so the
        recurrence only updates on real observations.  For every query, the
        prefix consisting of strictly-past rows is gathered.  All operations
        are label-order agnostic (renaming slots only permutes the V axis).
        Returns h_q: [bg, n_current, V, id_memory_dim].
        """
        bg, _, c = history_flat.shape
        n_current = current_times_flat.shape[1]
        v = self.num_id_vocabulary
        t_steps, n_slots = history_grid_shape
        device = history_flat.device
        if t_steps == 0 or n_slots == 0 or n_current == 0:
            return history_flat.new_zeros(
                (bg, n_current, v, self.id_memory_dim),
            )

        grid_labels = labels_flat.view(bg, t_steps, n_slots).clamp(min=0, max=v - 1)
        grid_times = history_times_flat.view(bg, t_steps, n_slots)
        grid_valid = (~history_masks_flat.view(bg, t_steps, n_slots))
        grid_feat = history_flat.view(bg, t_steps, n_slots, c)

        # Dense per-entity grid: [bg * V, T, c].
        b_idx = torch.arange(bg, device=device).view(bg, 1, 1).expand(bg, t_steps, n_slots)
        t_idx = torch.arange(t_steps, device=device).view(1, t_steps, 1).expand(bg, t_steps, n_slots)
        dest = ((b_idx * v + grid_labels) * t_steps + t_idx).reshape(-1)
        src_valid = grid_valid.reshape(-1)
        dense = history_flat.new_zeros(bg * v * t_steps, c)
        dense_valid = torch.zeros(bg * v * t_steps, dtype=torch.bool, device=device)
        dense[dest[src_valid]] = grid_feat.reshape(-1, c)[src_valid]
        dense_valid[dest[src_valid]] = True
        dense = dense.view(bg * v, t_steps, c)
        dense_valid = dense_valid.view(bg * v, t_steps)

        # GRUCell recurrence with gap masking (gaps are exact no-ops).
        h = history_flat.new_zeros(bg * v, self.id_memory_dim)
        prefixes = []
        for t in range(t_steps):
            h_new = self.id_memory_gru(dense[:, t], h)
            h = torch.where(dense_valid[:, t:t + 1], h_new, h)
            prefixes.append(h.view(bg, v, self.id_memory_dim))
        prefixes = torch.stack(prefixes, dim=2)     # [bg, V, T, d]

        # For each query, take the memory prefix of strictly-past rows.
        row_times = grid_times[:, :, 0]             # [bg, T] (constant across slots)
        past = row_times[:, None, :] < current_times_flat[:, :, None]   # [bg, Nq, T]
        has_past = past.any(dim=-1)                 # [bg, Nq]
        steps = torch.arange(t_steps, device=device)
        last_row = (past.long() * steps).argmax(dim=-1)                 # [bg, Nq]
        gather_idx = last_row[:, None, :, None].expand(
            bg, v, n_current, self.id_memory_dim,
        )
        h_q = prefixes.gather(dim=2, index=gather_idx)                   # [bg, V, Nq, d]
        h_q = h_q * has_past[:, None, :, None].to(h_q.dtype)
        return h_q.permute(0, 2, 1, 3)              # [bg, Nq, V, d]

    def _entity_state_aggregate(
        self, mention_scores, valid_address, invalid, queries, keys,
        history_flat, history_times_flat, current_times_flat,
    ):
        """CRR-E1 query-conditional causal entity-state aggregation.

        For every current query i and dynamic entity e, strictly past mentions
        are weighted by appearance compatibility, learned relative-time bias,
        and a learned observation reliability.  Their normalized key vectors
        form z_{i,e}; a shared bounded residual then corrects the P0 maximum.
        Numeric identity labels are used only as scatter addresses.
        """
        bg, n_current, history_len = mention_scores.shape
        n_entities = self.num_id_vocabulary
        feature_dim = keys.shape[-1]
        shape = (bg, n_current, n_entities)
        valid = ~invalid
        valid_float = valid.to(mention_scores.dtype)

        base_max = mention_scores.new_full(shape, -1.0e4)
        base_max.scatter_reduce_(
            dim=-1, index=valid_address, src=mention_scores,
            reduce="amax", include_self=True,
        )

        ages = (current_times_flat[:, :, None] - history_times_flat[:, None, :]).clamp(min=1)
        boundaries = torch.tensor([1, 2, 3, 5, 8, 12, 20], device=ages.device)
        age_bucket = torch.bucketize(ages, boundaries)
        time_bias = self.entity_state_time_bias(age_bucket).squeeze(-1)

        reliability = torch.sigmoid(
            self.entity_state_reliability(history_flat).squeeze(-1)
        ).clamp(min=1.0e-4, max=1.0)
        tau = self.log_entity_state_tau.exp().clamp(min=0.1, max=10.0)
        attention_logits = mention_scores / tau + time_bias + reliability[:, None, :].log()
        attention_logits = attention_logits.masked_fill(invalid, -1.0e4)

        attention_max = mention_scores.new_full(shape, -1.0e4)
        attention_max.scatter_reduce_(
            dim=-1, index=valid_address, src=attention_logits,
            reduce="amax", include_self=True,
        )
        shifted = attention_logits - attention_max.gather(dim=-1, index=valid_address)
        exp_weight = shifted.clamp(min=-30.0, max=0.0).exp() * valid_float
        denom = mention_scores.new_zeros(shape)
        denom.scatter_add_(dim=-1, index=valid_address, src=exp_weight)
        weights = exp_weight / denom.gather(dim=-1, index=valid_address).clamp(min=1.0e-8)

        # Dynamic entity states z_{i,e}; this is deliberately query-dependent.
        state = keys.new_zeros((bg, n_current, n_entities, feature_dim))
        state_index = valid_address[..., None].expand(-1, -1, -1, feature_dim)
        weighted_keys = weights[..., None] * keys[:, None, :, :]
        state.scatter_add_(dim=2, index=state_index, src=weighted_keys)
        state = F.normalize(state, dim=-1)
        state_score = (queries[:, :, None, :] * state).sum(dim=-1) * math.sqrt(feature_dim)

        count = mention_scores.new_zeros(shape)
        count.scatter_add_(dim=-1, index=valid_address, src=valid_float)
        entropy_src = -weights * weights.clamp(min=1.0e-8).log()
        entropy = mention_scores.new_zeros(shape)
        entropy.scatter_add_(dim=-1, index=valid_address, src=entropy_src)
        entropy_norm = entropy / torch.log(count.clamp(min=2.0))

        log_age = torch.log1p(ages.to(mention_scores.dtype)) / math.log(31.0)
        expected_age = mention_scores.new_zeros(shape)
        expected_age.scatter_add_(dim=-1, index=valid_address, src=weights * log_age)
        recent_mass = mention_scores.new_zeros(shape)
        recent_mass.scatter_add_(
            dim=-1, index=valid_address,
            src=weights * (ages <= 3).to(weights.dtype),
        )
        reliability_mean = mention_scores.new_zeros(shape)
        reliability_mean.scatter_add_(
            dim=-1, index=valid_address,
            src=weights * reliability[:, None, :],
        )
        reliability_max = mention_scores.new_full(shape, -1.0e4)
        reliability_max.scatter_reduce_(
            dim=-1, index=valid_address,
            src=reliability[:, None, :].expand(-1, n_current, -1).masked_fill(invalid, -1.0e4),
            reduce="amax", include_self=True,
        )
        concentration = mention_scores.new_zeros(shape)
        concentration.scatter_add_(dim=-1, index=valid_address, src=weights.square())

        scale = math.sqrt(feature_dim)
        features = torch.stack([
            base_max.clamp(min=-scale, max=scale) / scale,
            state_score.clamp(min=-scale, max=scale) / scale,
            (state_score - base_max).clamp(min=-2 * scale, max=2 * scale) / scale,
            torch.log1p(count).clamp(max=math.log(31.0)) / math.log(31.0),
            entropy_norm.clamp(min=0.0, max=1.0),
            expected_age.clamp(min=0.0, max=1.0),
            recent_mass.clamp(min=0.0, max=1.0),
            reliability_mean.clamp(min=0.0, max=1.0),
            reliability_max.clamp(min=0.0, max=1.0),
            concentration.clamp(min=0.0, max=1.0),
        ], dim=-1)
        residual = self.entity_state_residual_scale * torch.tanh(
            self.entity_state_residual(features).squeeze(-1)
        )
        output = base_max + residual
        return output.masked_fill(count <= 0, -1.0e4)


    def _consensus_residual_aggregate(
        self, mention_scores, valid_address, invalid, bg, n_current,
        history_times_flat, current_times_flat,
    ):
        """P0 max plus a bounded zero-init residual from top-3 causal evidence.

        Unlike normalized logsumexp, this preserves P0's entity/NEW calibration
        at initialization and does not average every historical token.  The
        shared residual sees only agreement gaps, causal ages, and support size;
        identity slot names never enter the computation.
        """
        h = mention_scores.shape[-1]
        work = mention_scores
        history_index = torch.arange(h, device=work.device).view(1, 1, h).expand(bg, n_current, h)
        ages = current_times_flat[:, :, None] - history_times_flat[:, None, :]
        valid = ~invalid
        top_scores, top_ages, present = [], [], []
        for _ in range(3):
            score = work.new_full((bg, n_current, self.num_id_vocabulary), -1.0e4)
            score.scatter_reduce_(dim=-1, index=valid_address, src=work,
                                  reduce="amax", include_self=True)
            gathered = score.gather(dim=-1, index=valid_address)
            is_top = valid & (work == gathered)
            index_src = torch.where(is_top, history_index, torch.full_like(history_index, h))
            arg = torch.full((bg, n_current, self.num_id_vocabulary), h,
                             dtype=torch.long, device=work.device)
            arg.scatter_reduce_(dim=-1, index=valid_address, src=index_src,
                                reduce="amin", include_self=True)
            has_value = arg < h
            safe_arg = arg.clamp(max=max(h - 1, 0))
            age = ages.gather(dim=-1, index=safe_arg).masked_fill(~has_value, 0)
            top_scores.append(score)
            top_ages.append(age)
            present.append(has_value)
            chosen_for_mention = arg.gather(dim=-1, index=valid_address)
            work = work.masked_fill(history_index == chosen_for_mention, -1.0e4)

        shape = (bg, n_current, self.num_id_vocabulary)
        valid_float = valid.to(mention_scores.dtype)
        count = mention_scores.new_zeros(shape)
        count.scatter_add_(dim=-1, index=valid_address, src=valid_float)
        valid_scores = torch.where(valid, mention_scores, torch.zeros_like(mention_scores))
        score_sum, score_sq_sum = mention_scores.new_zeros(shape), mention_scores.new_zeros(shape)
        score_sum.scatter_add_(dim=-1, index=valid_address, src=valid_scores)
        score_sq_sum.scatter_add_(dim=-1, index=valid_address, src=valid_scores.square())
        score_mean = score_sum / count.clamp(min=1.0)
        score_std = (score_sq_sum / count.clamp(min=1.0) - score_mean.square()).clamp(min=0.0).sqrt()

        m1, m2, m3 = top_scores
        m1_for_mentions = m1.gather(dim=-1, index=valid_address)
        support1, support2 = mention_scores.new_zeros(shape), mention_scores.new_zeros(shape)
        support1.scatter_add_(dim=-1, index=valid_address,
                              src=(valid & (mention_scores >= m1_for_mentions - 1.0)).to(mention_scores.dtype))
        support2.scatter_add_(dim=-1, index=valid_address,
                              src=(valid & (mention_scores >= m1_for_mentions - 2.0)).to(mention_scores.dtype))

        def temporal_bucket(bucket_mask):
            bucket_src = mention_scores.masked_fill(~(valid & bucket_mask), -1.0e4)
            value = mention_scores.new_full(shape, -1.0e4)
            value.scatter_reduce_(dim=-1, index=valid_address, src=bucket_src,
                                  reduce="amax", include_self=True)
            return value, value > INVALID_LOGIT

        recent, has_recent = temporal_bucket(ages <= 3)
        middle, has_middle = temporal_bucket((ages >= 4) & (ages <= 10))
        old, has_old = temporal_bucket(ages >= 11)
        scale = math.sqrt(self.feature_dim)
        age_scale = math.log(31.0)
        bucket_coverage = (has_recent.to(mention_scores.dtype)
                           + has_middle.to(mention_scores.dtype)
                           + has_old.to(mention_scores.dtype)) / 3.0
        recent_old_gap = torch.where(has_recent & has_old, recent - old, torch.zeros_like(recent))
        features = torch.stack([
            m1.clamp(min=-scale, max=scale) / scale,
            (m2 - m1).clamp(min=-2 * scale, max=0.0) / scale,
            (m3 - m1).clamp(min=-2 * scale, max=0.0) / scale,
            score_mean.clamp(min=-scale, max=scale) / scale,
            score_std.clamp(max=scale) / scale,
            torch.log1p(count).clamp(max=age_scale) / age_scale,
            support1 / count.clamp(min=1.0),
            support2 / count.clamp(min=1.0),
            torch.log1p(top_ages[0].clamp(min=0)).clamp(max=age_scale) / age_scale,
            torch.log1p(top_ages[1].clamp(min=0)).clamp(max=age_scale) / age_scale,
            torch.log1p(top_ages[2].clamp(min=0)).clamp(max=age_scale) / age_scale,
            present[1].to(mention_scores.dtype),
            present[2].to(mention_scores.dtype),
            has_recent.to(mention_scores.dtype),
            has_middle.to(mention_scores.dtype),
            has_old.to(mention_scores.dtype),
            recent_old_gap.clamp(min=-2 * scale, max=2 * scale) / scale,
            bucket_coverage,
        ], dim=-1)
        residual = self.consensus_residual_scale * torch.tanh(
            self.consensus_mlp(features).squeeze(-1)
        )
        output = m1 + residual
        return output.masked_fill(~present[0], -1.0e4)

    def _logsumexp_aggregate(
        self, mention_scores, valid_address, invalid, bg, n_current,
    ):
        """s(i,e) = tau * (logsumexp_j(a_ij/tau) - log|M_e|)

        Implemented with max-subtraction for numerical stability:
            logsumexp_j(a/tau) = max_j(a)/tau + log sum_j exp((a - max)/tau)
        so
            s(i,e) = max_j(a_ij) + tau * log( sum_j exp((a_ij - max)/tau) / |M_e| )

        At tau → 0  this reduces to max_j a_ij (P0 baseline).
        At tau → ∞  this approaches mean_j a_ij (H1 prototype, disallowed by design
        but the model can learn to keep tau low if max is better).
        """
        tau = self.log_tau.exp()  # scalar > 0
        valid_mask = (~invalid).float()  # [bg, N, H]

        # --- max per entity (for stability + as the leading term) ----------
        max_per_entity = mention_scores.new_full(
            (bg, n_current, self.num_id_vocabulary), -1.0e4,
        )
        max_per_entity.scatter_reduce_(
            dim=-1,
            index=valid_address,
            src=mention_scores,
            reduce="amax",
            include_self=True,
        )
        # Gather max back to the mention dimension for shift-subtraction.
        max_gathered = max_per_entity.gather(dim=-1, index=valid_address)

        # --- exp((a - max) / tau) * valid_mask ------------------------------
        shifted = (mention_scores - max_gathered) / tau
        exp_scores = torch.exp(shifted) * valid_mask

        # --- sum exp and count valid mentions, per entity -------------------
        sum_exp = mention_scores.new_zeros((bg, n_current, self.num_id_vocabulary))
        sum_exp.scatter_add_(dim=-1, index=valid_address, src=exp_scores)

        count = mention_scores.new_zeros((bg, n_current, self.num_id_vocabulary))
        count.scatter_add_(dim=-1, index=valid_address, src=valid_mask)

        # --- s(i,e) = max + tau * log(sum_exp / count) ----------------------
        log_count = torch.log(count.clamp(min=1.0))
        entity_scores = max_per_entity + tau * (
            torch.log(sum_exp.clamp(min=1e-8)) - log_count
        )

        # Entities with zero valid mentions: revert to -1e4 (same as max mode).
        no_mentions = count <= 0
        entity_scores = entity_scores.masked_fill(no_mentions, -1.0e4)
        return entity_scores
