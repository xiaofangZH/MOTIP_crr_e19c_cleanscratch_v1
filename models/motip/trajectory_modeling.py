# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn

from models.ffn import FFN


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            use_causal_history_prototype: bool = False,
            use_symmetric_history_prototype: bool = False,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.use_causal_history_prototype = use_causal_history_prototype
        self.use_symmetric_history_prototype = use_symmetric_history_prototype

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        # H1: one deliberately small addition. At zero, this branch is exactly
        # identical to official MOTIP. During training it can learn how much of
        # each token should be replaced by its same-track causal prefix mean.
        if self.use_causal_history_prototype:
            self.causal_prototype_scale = nn.Parameter(torch.zeros(()))
        # E19d (1): symmetric counterpart.  The causal variant can only see the
        # prefix of a track, so its prototype is biased toward the first frames
        # (young appearance, worst occlusion).  The symmetric variant averages
        # the WHOLE valid track column instead.  Legitimate at inference: the
        # history block is entirely older than the current frame, so mixing
        # within it cannot look into the future.  Zero initialisation keeps the
        # first step identical to the official model, and the gradient through
        # this scalar is delta-scaled (not a zero product), so it wakes up
        # immediately.
        if self.use_symmetric_history_prototype:
            self.symmetric_prototype_scale = nn.Parameter(torch.zeros(()))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, seq_info):
        trajectory_features = self._transform_tokens(seq_info["trajectory_features"])
        if "unknown_features" in seq_info:
            seq_info["unknown_features"] = self._transform_tokens(
                seq_info["unknown_features"],
            )
        if self.use_symmetric_history_prototype:
            trajectory_features = self._mix_symmetric_history_prototype(
                trajectory_features,
                seq_info["trajectory_masks"],
            )
        if self.use_causal_history_prototype:
            trajectory_features = self._mix_causal_history_prototype(
                trajectory_features,
                seq_info["trajectory_masks"],
            )
        seq_info["trajectory_features"] = trajectory_features
        return seq_info

    def _transform_tokens(self, features):
        features = features + self.adapter(features)
        features = self.norm(features)
        features = features + self.ffn(features)
        return self.ffn_norm(features)

    def _mix_causal_history_prototype(self, features, masks):
        """Mix each valid token with the prefix mean of its own track column."""
        valid = (~masks).unsqueeze(-1).to(features.dtype)
        prefix_sum = (features * valid).cumsum(dim=2)
        prefix_count = valid.cumsum(dim=2).clamp_min(1.0)
        prefix_mean = prefix_sum / prefix_count
        delta = (prefix_mean - features) * valid
        return features + self.causal_prototype_scale * delta

    def _mix_symmetric_history_prototype(self, features, masks):
        """Mix each valid token with the masked mean of its own track column.

        `features` is [b, g, T, N, c] and `masks` is [b, g, T, N] (True =
        padded).  The mean runs over the whole temporal axis of the SAME track
        slot, so a token sees all valid observations of its own column rather
        than only the prefix.  Padded tokens are excluded from both the sum and
        the count, and are left untouched (delta multiplied by `valid`).
        """
        valid = (~masks).unsqueeze(-1).to(features.dtype)
        column_sum = (features * valid).sum(dim=2, keepdim=True)
        column_count = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        column_mean = column_sum / column_count
        delta = (column_mean - features) * valid
        return features + self.symmetric_prototype_scale * delta
