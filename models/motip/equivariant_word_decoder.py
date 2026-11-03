import math

import einops
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.ffn import FFN
from models.misc import _get_clones


INVALID_LOGIT = -1.0e4


class EquivariantWordDecoder(nn.Module):
    """Decode appearance+word tokens with causal, identity-equivariant readout."""

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
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = self.hidden_dim // head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length
        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head
        if self.hidden_dim % head_dim != 0:
            raise ValueError("feature_dim + id_dim must be divisible by head_dim")

        self.history_appearance_proj = nn.Linear(feature_dim, feature_dim)
        self.query_appearance_proj = nn.Linear(feature_dim, feature_dim)
        self.word_input_proj = nn.Linear(feature_dim, id_dim)
        self.entity_word_pool = nn.Linear(id_dim, 1, bias=False)
        self.new_word = nn.Parameter(torch.empty(id_dim))
        nn.init.normal_(self.new_word, std=0.02)

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

        query_readout = nn.Linear(id_dim, id_dim, bias=False)
        entity_readout = nn.Linear(id_dim, id_dim, bias=False)
        if use_aux_loss and not use_shared_aux_head:
            self.query_readout_layers = _get_clones(query_readout, num_layers)
            self.entity_readout_layers = _get_clones(entity_readout, num_layers)
        else:
            self.query_readout_layers = nn.ModuleList([query_readout for _ in range(num_layers)])
            self.entity_readout_layers = nn.ModuleList([entity_readout for _ in range(num_layers)])
        self.log_score_scale = nn.Parameter(torch.tensor(math.log(id_dim ** -0.5)))

        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "rel_pos_embeds" not in name:
                nn.init.xavier_uniform_(parameter)

    def forward(self, seq_info, use_decoder_checkpoint=False):
        history = seq_info["trajectory_features"]
        current = self.query_appearance_proj(seq_info["unknown_features"])
        history_labels = seq_info["trajectory_id_labels"]
        history_times = seq_info["trajectory_times"]
        history_masks = seq_info["trajectory_masks"]
        current_times = seq_info["unknown_times"]
        current_masks = seq_info["unknown_masks"]
        current_labels = seq_info.get("unknown_id_labels")

        b, g, k, m, _ = current.shape
        frame_times = self._current_frame_times(current_times, current_masks)
        history_appearance = self.history_appearance_proj(history)
        history_words, entity_words, entity_valid = self._causal_entity_words(
            history_appearance=history_appearance,
            history_labels=history_labels,
            history_times=history_times,
            history_masks=history_masks,
            frame_times=frame_times,
        )

        history_tokens = torch.cat([history_appearance, history_words], dim=-1)
        new_words = self.new_word.view(1, 1, 1, 1, self.id_dim).expand(
            b, g, k, m, self.id_dim,
        )
        query_tokens = torch.cat([current, new_words], dim=-1)
        queries = einops.rearrange(
            query_tokens, "b g k m c -> (b g) (k m) c",
        ).contiguous()
        mention_tokens = einops.rearrange(
            history_tokens, "b g t n c -> (b g) (t n) c",
        ).contiguous()

        self_attn_mask = einops.rearrange(
            current_masks, "b g k m -> (b g k) m",
        ).contiguous()
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
        flat_frame_times = einops.rearrange(frame_times, "b g k -> (b g) k")
        query_times = einops.repeat(flat_frame_times, "bg k -> bg (k m)", m=m)
        rel_pos_idx = (
            query_times[:, :, None] - mention_times[:, None, :]
        ).clamp(min=0, max=self.rel_pe_length - 1)
        cross_attn_mask = mention_times[:, None, :] >= query_times[:, :, None]
        cross_attn_mask = einops.repeat(
            cross_attn_mask, "bg q s -> (bg h) q s", h=self.n_heads,
        ).contiguous()
        cross_attn_mask = cross_attn_mask.to(dtype=queries.dtype).masked_fill(
            cross_attn_mask, float("-inf"),
        )
        mention_key_padding_mask = (~mention_valid).to(dtype=queries.dtype).masked_fill(
            ~mention_valid, float("-inf"),
        )

        entity_words = einops.rearrange(
            entity_words, "b g k v d -> (b g) k v d",
        ).contiguous()
        entity_valid = einops.rearrange(
            entity_valid, "b g k v -> (b g) k v",
        ).contiguous()

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
                entity_words=entity_words,
                entity_valid=entity_valid,
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

    def _causal_entity_words(
            self,
            history_appearance,
            history_labels,
            history_times,
            history_masks,
            frame_times,
    ):
        b, g, t_steps, n_slots, _ = history_appearance.shape
        num_frames = frame_times.shape[2]
        bg = b * g
        v = self.num_id_vocabulary
        d = self.id_dim
        device = history_appearance.device
        if t_steps == 0 or n_slots == 0:
            history_words = history_appearance.new_zeros((b, g, t_steps, n_slots, d))
            entity_words = history_appearance.new_zeros((b, g, num_frames, v, d))
            entity_valid = torch.zeros(
                (b, g, num_frames, v), dtype=torch.bool, device=device,
            )
            return history_words, entity_words, entity_valid

        appearance = einops.rearrange(
            history_appearance, "b g t n c -> (b g) t n c",
        ).contiguous()
        labels = einops.rearrange(history_labels, "b g t n -> (b g) t n")
        times = einops.rearrange(history_times, "b g t n -> (b g) t n")
        masks = einops.rearrange(history_masks, "b g t n -> (b g) t n")
        valid = (~masks) & (labels >= 0) & (labels < v)
        safe_labels = labels.clamp(min=0, max=v - 1)
        word_inputs = self.word_input_proj(appearance)

        b_idx = torch.arange(bg, device=device).view(bg, 1, 1).expand(bg, t_steps, n_slots)
        t_idx = torch.arange(t_steps, device=device).view(1, t_steps, 1).expand(
            bg, t_steps, n_slots,
        )
        destination = ((b_idx * v + safe_labels) * t_steps + t_idx).reshape(-1)
        source_valid = valid.reshape(-1)
        dense_inputs = history_appearance.new_zeros((bg * v * t_steps, d))
        dense_valid = torch.zeros(bg * v * t_steps, dtype=torch.bool, device=device)
        dense_inputs[destination[source_valid]] = word_inputs.reshape(-1, d)[source_valid]
        dense_valid[destination[source_valid]] = True
        dense_inputs = dense_inputs.view(bg * v, t_steps, d)
        dense_valid = dense_valid.view(bg * v, t_steps)

        pool_steps = torch.arange(t_steps, device=device)
        causal = pool_steps[:, None] <= pool_steps[None, :]
        pool_valid = causal[None, :, :] & dense_valid[:, :, None]
        pool_logits = self.entity_word_pool(dense_inputs).squeeze(-1)
        pool_logits = pool_logits[:, :, None].expand(-1, -1, t_steps).masked_fill(
            ~pool_valid, INVALID_LOGIT,
        )
        pool_weights = torch.softmax(pool_logits, dim=1)
        pool_weights = pool_weights * pool_valid.to(pool_weights.dtype)
        pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        word_states = torch.einsum("eij,eid->ejd", pool_weights, dense_inputs)
        seen_states = dense_valid.cumsum(dim=1) > 0

        states_by_time = word_states.view(bg, v, t_steps, d).permute(0, 2, 1, 3)
        history_index = safe_labels.unsqueeze(-1).expand(bg, t_steps, n_slots, d)
        history_words = states_by_time.gather(dim=2, index=history_index)
        history_words = history_words * valid.unsqueeze(-1).to(history_words.dtype)

        flat_frame_times = einops.rearrange(frame_times, "b g k -> (b g) k")
        row_times = times[:, :, 0]
        past = row_times[:, None, :] < flat_frame_times[:, :, None]
        has_past = past.any(dim=-1)
        steps = torch.arange(t_steps, device=device)
        last_step = (past.long() * steps).argmax(dim=-1)
        state_index = last_step[:, :, None, None].expand(bg, num_frames, v, d)
        entity_words = states_by_time.gather(dim=1, index=state_index)

        seen_by_time = seen_states.view(bg, v, t_steps).permute(0, 2, 1)
        seen_index = last_step[:, :, None].expand(bg, num_frames, v)
        entity_valid = seen_by_time.gather(dim=1, index=seen_index)
        entity_valid = entity_valid & has_past.unsqueeze(-1)
        entity_words = entity_words * entity_valid.unsqueeze(-1).to(entity_words.dtype)

        history_words = einops.rearrange(
            history_words, "(b g) t n d -> b g t n d", b=b, g=g,
        )
        entity_words = einops.rearrange(
            entity_words, "(b g) k v d -> b g k v d", b=b, g=g,
        )
        entity_valid = einops.rearrange(
            entity_valid, "(b g) k v -> b g k v", b=b, g=g,
        )
        return history_words, entity_words, entity_valid

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

    def _compute_logits(self, layer, queries, entity_words, entity_valid):
        num_frames = entity_words.shape[1]
        query_words = self.query_readout_layers[layer](queries[..., self.feature_dim:])
        query_words = einops.rearrange(
            query_words, "bg (k m) d -> bg k m d", k=num_frames,
        )
        entity_words = self.entity_readout_layers[layer](entity_words)
        entity_logits = torch.einsum("xkmd,xkvd->xkmv", query_words, entity_words)
        new_word = self.entity_readout_layers[layer](self.new_word)
        new_logits = torch.sum(query_words * new_word, dim=-1, keepdim=True)
        scale = self.log_score_scale.exp()
        entity_logits = (entity_logits * scale).masked_fill(
            ~entity_valid[:, :, None, :], INVALID_LOGIT,
        )
        return torch.cat([entity_logits, new_logits * scale], dim=-1)
