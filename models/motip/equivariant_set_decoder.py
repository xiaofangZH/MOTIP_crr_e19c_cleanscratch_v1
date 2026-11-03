import math

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.ffn import FFN
from models.misc import _get_clones


INVALID_LOGIT = -1.0e4


class EquivariantSetDecoder(nn.Module):
    """Decode causal entity sets without identity-specific classifier weights."""

    def __init__(
            self,
            feature_dim: int,
            id_dim: int,
            ffn_dim_ratio: int,
            num_layers: int,
            head_dim: int,
            num_id_vocabulary: int,
            rel_pe_length: int,
            use_aux_loss: bool,
            use_shared_aux_head: bool,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.hidden_dim = feature_dim + id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = self.hidden_dim // head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length
        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head
        if self.hidden_dim % head_dim != 0:
            raise ValueError("feature_dim + id_dim must be divisible by head_dim")

        self.query_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.history_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.new_token = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.new_token, std=0.02)

        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((num_layers, rel_pe_length, self.n_heads), dtype=torch.float)
        )

        self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        ffn = FFN(
            d_model=self.hidden_dim,
            d_ffn=self.hidden_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.self_attn_layers = _get_clones(self_attn, num_layers - 1)
        self.self_attn_norm_layers = _get_clones(nn.LayerNorm(self.hidden_dim), num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, num_layers)
        self.cross_attn_norm_layers = _get_clones(nn.LayerNorm(self.hidden_dim), num_layers)
        self.ffn_layers = _get_clones(ffn, num_layers)
        self.ffn_norm_layers = _get_clones(nn.LayerNorm(self.hidden_dim), num_layers)

        query_readout = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        entity_readout = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        if use_aux_loss and not use_shared_aux_head:
            self.query_readout_layers = _get_clones(query_readout, num_layers)
            self.entity_readout_layers = _get_clones(entity_readout, num_layers)
        else:
            self.query_readout_layers = nn.ModuleList([query_readout for _ in range(num_layers)])
            self.entity_readout_layers = nn.ModuleList([entity_readout for _ in range(num_layers)])
        self.score_scale = self.hidden_dim ** -0.5
        self.new_query_head = nn.Linear(self.hidden_dim, 1)
        self.new_context_logit = nn.Parameter(torch.tensor(math.log(1.0 / 3.0)))
        self.conflict_logit = nn.Parameter(torch.tensor(math.log(1.0 / 3.0)))

        self.entity_state_residual_scale = 2.0
        self.entity_state_time_bias = nn.Embedding(8, 1)
        self.entity_state_reliability = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.entity_state_residual = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.log_entity_state_tau = nn.Parameter(torch.tensor(0.0))
        self.id_memory_dim = 128
        self.id_memory_gru = nn.GRUCell(self.hidden_dim, self.id_memory_dim)
        self.id_memory_proj = nn.Linear(self.id_memory_dim, self.hidden_dim, bias=False)
        self.id_gate = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "rel_pos_embeds" not in name:
                nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.entity_state_time_bias.weight)
        nn.init.zeros_(self.new_query_head.weight)
        nn.init.zeros_(self.new_query_head.bias)

    def forward(self, seq_info, use_decoder_checkpoint=False):
        history = seq_info["trajectory_features"]
        current = seq_info["unknown_features"]
        history_labels = seq_info["trajectory_id_labels"]
        history_times = seq_info["trajectory_times"]
        history_masks = seq_info["trajectory_masks"]
        current_times = seq_info["unknown_times"]
        current_masks = seq_info["unknown_masks"]
        current_labels = seq_info.get("unknown_id_labels")

        b, g, k, m, _ = current.shape
        frame_times = self._current_frame_times(current_times, current_masks)
        history_embeds = self.history_proj(history)
        bg = b * g

        queries = self.query_proj(current)
        queries = einops.rearrange(queries, "b g k m c -> (b g) (k m) c").contiguous()
        new_query_scores = self.new_query_head(queries).squeeze(-1)
        query_valid = einops.rearrange(
            ~current_masks, "b g k m -> (b g) k m",
        ).contiguous()
        self_attn_mask = einops.rearrange(current_masks, "b g k m -> (b g k) m").contiguous()
        all_padded = self_attn_mask.all(dim=-1, keepdim=True)
        self_attn_mask = torch.where(
            all_padded,
            torch.zeros_like(self_attn_mask),
            self_attn_mask,
        )

        label_valid = (history_labels >= 0) & (history_labels < self.num_id_vocabulary)
        mention_valid = einops.rearrange(
            (~history_masks) & label_valid, "b g t n -> (b g) (t n)",
        ).contiguous()
        mention_times = einops.rearrange(history_times, "b g t n -> (b g) (t n)")
        query_times = einops.repeat(frame_times, "b g k -> (b g) (k m)", m=m)
        rel_pos_idx = (
            query_times[:, :, None] - mention_times[:, None, :]
        ).clamp(min=0, max=self.rel_pe_length - 1)
        cross_attn_mask = mention_times[:, None, :] >= query_times[:, :, None]

        mention_tokens = einops.rearrange(
            history_embeds, "b g t n c -> (b g) (t n) c",
        ).contiguous()
        mention_labels = einops.rearrange(
            history_labels, "b g t n -> (b g) (t n)",
        ).contiguous()
        memory_prefixes = self._entity_memory_prefixes(
            history_tokens=mention_tokens,
            history_labels=mention_labels,
            history_times=mention_times,
            history_masks=~mention_valid,
            current_times=query_times,
            history_grid_shape=(history.shape[2], history.shape[3]),
        )
        history_new_token = self.new_token.view(1, 1, self.hidden_dim).expand(bg, 1, -1)
        mention_tokens = torch.cat([mention_tokens, history_new_token], dim=1)
        mention_valid = torch.cat([
            mention_valid,
            mention_valid.new_ones((bg, 1)),
        ], dim=1)
        mention_labels = torch.cat([
            mention_labels,
            mention_labels.new_full((bg, 1), self.num_id_vocabulary),
        ], dim=1)
        mention_times = torch.cat([
            mention_times,
            mention_times.new_zeros((bg, 1)),
        ], dim=1)
        rel_pos_idx = torch.cat([
            rel_pos_idx,
            rel_pos_idx.new_zeros((bg, k * m, 1)),
        ], dim=2)
        cross_attn_mask = torch.cat([
            cross_attn_mask,
            cross_attn_mask.new_zeros((bg, k * m, 1)),
        ], dim=2)
        cross_attn_mask = einops.repeat(
            cross_attn_mask, "bg q s -> (bg h) q s", h=self.n_heads,
        ).contiguous()
        cross_attn_mask = cross_attn_mask.to(dtype=queries.dtype).masked_fill(
            cross_attn_mask, float("-inf"),
        )
        mention_key_padding_mask = (~mention_valid).to(dtype=queries.dtype).masked_fill(
            ~mention_valid, float("-inf"),
        )

        all_logits = None
        all_labels = None
        all_masks = None
        logits = None
        for layer in range(self.num_layers):
            if use_decoder_checkpoint:
                queries = checkpoint(
                    self._forward_a_layer,
                    layer,
                    queries,
                    mention_tokens,
                    self_attn_mask,
                    mention_key_padding_mask,
                    cross_attn_mask,
                    rel_pos_idx,
                    k,
                    use_reentrant=False,
                )
            else:
                queries = self._forward_a_layer(
                    layer=layer,
                    queries=queries,
                    mention_tokens=mention_tokens,
                    self_attn_mask=self_attn_mask,
                    mention_key_padding_mask=mention_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pos_idx=rel_pos_idx,
                    num_frames=k,
                )
            logits = self._compute_logits(
                layer=layer,
                queries=queries,
                mention_tokens=mention_tokens,
                mention_labels=mention_labels,
                mention_valid=mention_valid,
                mention_times=mention_times,
                frame_times=einops.rearrange(frame_times, "b g k -> (b g) k"),
                memory_prefixes=memory_prefixes,
                new_query_scores=new_query_scores,
                query_valid=query_valid,
            )
            logits = einops.rearrange(
                logits, "(b g) k m v -> b g k m v", b=b, g=g,
            )
            labels = current_labels if self.training else None
            if all_logits is None:
                all_logits = logits
                all_labels = labels
                all_masks = current_masks
            else:
                all_logits = torch.cat([all_logits, logits], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0) if labels is not None else None
                all_masks = torch.cat([all_masks, current_masks], dim=0)

        if self.training and self.use_aux_loss:
            return all_logits, all_labels, all_masks
        return logits, current_labels if self.training else None, current_masks

    def _current_frame_times(self, current_times, current_masks):
        valid_times = current_times.masked_fill(current_masks, -1)
        return valid_times.amax(dim=-1).clamp(min=0)

    def _forward_a_layer(
            self,
            layer,
            queries,
            mention_tokens,
            self_attn_mask,
            mention_key_padding_mask,
            cross_attn_mask,
            rel_pos_idx,
            num_frames,
    ):
        if layer > 0:
            frame_queries = einops.rearrange(
                queries, "bg (k m) c -> (bg k) m c", k=num_frames,
            )
            self_out, _ = self.self_attn_layers[layer - 1](
                query=frame_queries,
                key=frame_queries,
                value=frame_queries,
                key_padding_mask=self_attn_mask,
            )
            frame_queries = self.self_attn_norm_layers[layer - 1](frame_queries + self_out)
            queries = einops.rearrange(
                frame_queries, "(bg k) m c -> bg (k m) c", k=num_frames,
            )

        rel_pos = self.rel_pos_embeds[layer][rel_pos_idx]
        attn_bias = einops.rearrange(rel_pos, "bg q s h -> (bg h) q s").contiguous()
        attn_bias = attn_bias + cross_attn_mask
        cross_out, _ = self.cross_attn_layers[layer](
            query=queries,
            key=mention_tokens,
            value=mention_tokens,
            key_padding_mask=mention_key_padding_mask,
            attn_mask=attn_bias,
        )
        queries = self.cross_attn_norm_layers[layer](queries + cross_out)
        queries = self.ffn_norm_layers[layer](queries + self.ffn_layers[layer](queries))
        return queries

    def _compute_logits(
            self,
            layer,
            queries,
            mention_tokens,
            mention_labels,
            mention_valid,
            mention_times,
            frame_times,
            memory_prefixes,
            new_query_scores,
            query_valid,
    ):
        query_words = self.query_readout_layers[layer](queries)
        mention_words = self.entity_readout_layers[layer](mention_tokens)
        logits = torch.matmul(query_words, mention_words.transpose(1, 2)) * self.score_scale
        logits = einops.rearrange(
            logits, "bg (k m) s -> bg k m s", k=frame_times.shape[1],
        )

        is_new = mention_labels == self.num_id_vocabulary
        causal_valid = mention_times[:, None, :] < frame_times[:, :, None]
        readout_valid = mention_valid[:, None, :] & (causal_valid | is_new[:, None, :])
        logits = logits.masked_fill(~readout_valid[:, :, None, :], float("-inf"))

        num_frames = frame_times.shape[1]
        num_queries = queries.shape[1]
        num_slots = num_queries // num_frames
        query_times = einops.repeat(frame_times, "bg k -> bg (k m)", m=num_slots)
        refined_entities = self._refine_entity_scores(
            mention_scores=einops.rearrange(
                logits[..., :-1], "bg k m s -> bg (k m) s",
            ),
            query_words=query_words,
            history_tokens=mention_tokens[:, :-1],
            mention_labels=mention_labels[:, :-1],
            mention_valid=mention_valid[:, :-1],
            history_times=mention_times[:, :-1],
            current_times=query_times,
            memory_prefixes=memory_prefixes,
        )
        refined_entities = einops.rearrange(
            refined_entities, "bg (k m) v -> bg k m v", k=num_frames,
        )
        refined_entities = self._suppress_same_frame_conflicts(
            refined_entities, query_valid,
        )
        new_scores = self._calibrate_new_scores(
            refined_entities,
            einops.rearrange(new_query_scores, "bg (k m) -> bg k m", k=num_frames),
        )
        return torch.cat([refined_entities, new_scores.unsqueeze(-1)], dim=-1)

    def _suppress_same_frame_conflicts(self, entity_scores, query_valid=None):
        valid = entity_scores > INVALID_LOGIT
        if query_valid is not None:
            valid = valid & query_valid.unsqueeze(-1)
        column_max = entity_scores.masked_fill(~valid, INVALID_LOGIT).amax(
            dim=2, keepdim=True,
        )
        upper_bound = torch.where(
            valid, column_max.expand_as(entity_scores), entity_scores,
        )
        gap = (upper_bound - entity_scores).clamp_min(0.0)
        penalty = self.conflict_logit.sigmoid() * gap
        suppressed = entity_scores - penalty.masked_fill(~valid, 0.0)
        return torch.where(valid, suppressed, entity_scores)

    def _calibrate_new_scores(self, entity_scores, new_query_scores):
        valid = entity_scores > INVALID_LOGIT
        n_valid = valid.sum(dim=-1, keepdim=True).to(entity_scores.dtype)
        context = torch.logsumexp(entity_scores, dim=-1, keepdim=True) - torch.log(
            n_valid.clamp(min=1.0),
        )
        context = torch.where(valid.any(dim=-1, keepdim=True), context, 0.0)
        return new_query_scores - self.new_context_logit.sigmoid() * context.squeeze(-1)

    def _refine_entity_scores(
            self,
            mention_scores,
            query_words,
            history_tokens,
            mention_labels,
            mention_valid,
            history_times,
            current_times,
            memory_prefixes,
    ):
        n_current = mention_scores.shape[1]
        valid_address = mention_labels.clamp(
            min=0, max=self.num_id_vocabulary - 1,
        )[:, None, :].expand(-1, n_current, -1)
        invalid = (~mention_valid[:, None, :]) | (
            history_times[:, None, :] >= current_times[:, :, None]
        )
        mention_scores = mention_scores.masked_fill(invalid, INVALID_LOGIT)
        features, base_max, count = self._entity_state_features(
            mention_scores=mention_scores,
            valid_address=valid_address,
            invalid=invalid,
            history_tokens=history_tokens,
            history_times=history_times,
            current_times=current_times,
        )
        residual = self.entity_state_residual_scale * torch.tanh(
            self.entity_state_residual(features).squeeze(-1)
        )
        pointer_scores = (base_max + residual).masked_fill(count <= 0, INVALID_LOGIT)

        id_queries = F.normalize(query_words, dim=-1)
        id_keys = F.normalize(self.id_memory_proj(memory_prefixes), dim=-1)
        id_scores = torch.einsum(
            "bqc,bqvc->bqv", id_queries, id_keys,
        ) * math.sqrt(self.hidden_dim)
        scale = math.sqrt(self.hidden_dim)
        gate_features = torch.stack([
            pointer_scores.clamp(min=-2 * scale, max=2 * scale) / (2 * scale),
            id_scores.clamp(min=-2 * scale, max=2 * scale) / (2 * scale),
            (id_scores - pointer_scores).clamp(min=-4 * scale, max=4 * scale) / (4 * scale),
            torch.log1p(count).clamp(max=math.log(31.0)) / math.log(31.0),
        ], dim=-1)
        gate = torch.tanh(self.id_gate(gate_features).squeeze(-1))
        output = pointer_scores + gate * (id_scores - pointer_scores)
        return output.masked_fill(count <= 0, INVALID_LOGIT)

    def _entity_state_features(
            self,
            mention_scores,
            valid_address,
            invalid,
            history_tokens,
            history_times,
            current_times,
    ):
        bg, n_current, _ = mention_scores.shape
        shape = (bg, n_current, self.num_id_vocabulary)
        valid = ~invalid
        valid_float = valid.to(mention_scores.dtype)

        base_max = mention_scores.new_full(shape, INVALID_LOGIT)
        base_max.scatter_reduce_(
            dim=-1,
            index=valid_address,
            src=mention_scores,
            reduce="amax",
            include_self=True,
        )

        ages = (current_times[:, :, None] - history_times[:, None, :]).clamp(min=1)
        boundaries = torch.tensor([1, 2, 3, 5, 8, 12, 20], device=ages.device)
        age_bucket = torch.bucketize(ages, boundaries)
        time_bias = self.entity_state_time_bias(age_bucket).squeeze(-1)
        reliability = torch.sigmoid(
            self.entity_state_reliability(history_tokens).squeeze(-1)
        ).clamp(min=1.0e-4, max=1.0)
        tau = self.log_entity_state_tau.exp().clamp(min=0.1, max=10.0)
        attention_logits = mention_scores / tau + time_bias + reliability[:, None, :].log()
        attention_logits = attention_logits.masked_fill(invalid, INVALID_LOGIT)

        attention_max = mention_scores.new_full(shape, INVALID_LOGIT)
        attention_max.scatter_reduce_(
            dim=-1,
            index=valid_address,
            src=attention_logits,
            reduce="amax",
            include_self=True,
        )
        shifted = attention_logits - attention_max.gather(dim=-1, index=valid_address)
        exp_weight = shifted.clamp(min=-30.0, max=0.0).exp() * valid_float
        denom = mention_scores.new_zeros(shape)
        denom.scatter_add_(dim=-1, index=valid_address, src=exp_weight)
        weights = exp_weight / denom.gather(dim=-1, index=valid_address).clamp(min=1.0e-8)

        count = mention_scores.new_zeros(shape)
        count.scatter_add_(dim=-1, index=valid_address, src=valid_float)
        entropy = mention_scores.new_zeros(shape)
        entropy.scatter_add_(
            dim=-1,
            index=valid_address,
            src=-weights * weights.clamp(min=1.0e-8).log(),
        )
        entropy_norm = entropy / torch.log(count.clamp(min=2.0))

        log_age = torch.log1p(ages.to(mention_scores.dtype)) / math.log(31.0)
        expected_age = mention_scores.new_zeros(shape)
        expected_age.scatter_add_(
            dim=-1, index=valid_address, src=weights * log_age,
        )
        recent_mass = mention_scores.new_zeros(shape)
        recent_mass.scatter_add_(
            dim=-1,
            index=valid_address,
            src=weights * (ages <= 3).to(weights.dtype),
        )
        reliability_mean = mention_scores.new_zeros(shape)
        reliability_mean.scatter_add_(
            dim=-1,
            index=valid_address,
            src=weights * reliability[:, None, :],
        )
        reliability_max = mention_scores.new_full(shape, INVALID_LOGIT)
        reliability_max.scatter_reduce_(
            dim=-1,
            index=valid_address,
            src=reliability[:, None, :].expand(-1, n_current, -1).masked_fill(
                invalid, INVALID_LOGIT,
            ),
            reduce="amax",
            include_self=True,
        )
        concentration = mention_scores.new_zeros(shape)
        concentration.scatter_add_(
            dim=-1, index=valid_address, src=weights.square(),
        )

        scale = math.sqrt(self.hidden_dim)
        features = torch.stack([
            base_max.clamp(min=-scale, max=scale) / scale,
            torch.log1p(count).clamp(max=math.log(31.0)) / math.log(31.0),
            entropy_norm.clamp(min=0.0, max=1.0),
            expected_age.clamp(min=0.0, max=1.0),
            recent_mass.clamp(min=0.0, max=1.0),
            reliability_mean.clamp(min=0.0, max=1.0),
            reliability_max.clamp(min=0.0, max=1.0),
            concentration.clamp(min=0.0, max=1.0),
        ], dim=-1)
        return features, base_max, count

    def _entity_memory_prefixes(
            self,
            history_tokens,
            history_labels,
            history_times,
            history_masks,
            current_times,
            history_grid_shape,
    ):
        bg, _, c = history_tokens.shape
        n_current = current_times.shape[1]
        v = self.num_id_vocabulary
        t_steps, n_slots = history_grid_shape
        device = history_tokens.device
        if t_steps == 0 or n_slots == 0 or n_current == 0:
            return history_tokens.new_zeros(
                (bg, n_current, v, self.id_memory_dim),
            )

        grid_labels = history_labels.view(bg, t_steps, n_slots).clamp(min=0, max=v - 1)
        grid_times = history_times.view(bg, t_steps, n_slots)
        grid_valid = ~history_masks.view(bg, t_steps, n_slots)
        grid_features = history_tokens.view(bg, t_steps, n_slots, c)

        b_idx = torch.arange(bg, device=device).view(bg, 1, 1).expand(bg, t_steps, n_slots)
        t_idx = torch.arange(t_steps, device=device).view(1, t_steps, 1).expand(
            bg, t_steps, n_slots,
        )
        destination = ((b_idx * v + grid_labels) * t_steps + t_idx).reshape(-1)
        source_valid = grid_valid.reshape(-1)
        dense = history_tokens.new_zeros(bg * v * t_steps, c)
        dense_valid = torch.zeros(bg * v * t_steps, dtype=torch.bool, device=device)
        dense[destination[source_valid]] = grid_features.reshape(-1, c)[source_valid]
        dense_valid[destination[source_valid]] = True
        dense = dense.view(bg * v, t_steps, c)
        dense_valid = dense_valid.view(bg * v, t_steps)

        hidden = history_tokens.new_zeros(bg * v, self.id_memory_dim)
        prefixes = []
        for step in range(t_steps):
            updated = self.id_memory_gru(dense[:, step], hidden)
            hidden = torch.where(dense_valid[:, step:step + 1], updated, hidden)
            prefixes.append(hidden.view(bg, v, self.id_memory_dim))
        prefixes = torch.stack(prefixes, dim=2)

        row_times = grid_times[:, :, 0]
        past = row_times[:, None, :] < current_times[:, :, None]
        has_past = past.any(dim=-1)
        steps = torch.arange(t_steps, device=device)
        last_row = (past.long() * steps).argmax(dim=-1)
        gather_index = last_row[:, None, :, None].expand(
            bg, v, n_current, self.id_memory_dim,
        )
        memory = prefixes.gather(dim=2, index=gather_index)
        memory = memory * has_past[:, None, :, None].to(memory.dtype)
        return memory.permute(0, 2, 1, 3)
