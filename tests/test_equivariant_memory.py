"""Unit tests for CRR-E5 equivariant identity memory.

CPU-only, no model checkpoint needed.  Verifies:
1. zero-init gate: equivariant_memory logits are bit-identical to the E1
   entity_state channel (and to P0 max) at initialisation -- safe bootstrap;
2. permutation equivariance: renaming identity slots only permutes output
   columns (paper contribution 1 evidence), with the identity channel active;
3. strict causality: future history cannot change earlier-query logits;
4. GRU gap masking: rows without an observation are exact no-ops;
5. Sinkhorn arbitration: row-normalised, permutation equivariant, off by
   default.
"""
import os
import sys

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.motip.entity_pointer import CausalEntityPointer  # noqa: E402


def sample(b=1, g=2, t=4, n=3, c=8, seed=3):
    torch.manual_seed(seed)
    history = torch.randn(b, g, t, n, c)
    current = torch.randn(b, g, t, n, c)
    labels = torch.tensor([0, 1, 2])
    ids = labels.view(1, 1, 1, n).expand(b, g, t, n).clone()
    times = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, n).clone()
    masks = torch.zeros((b, g, t, n), dtype=torch.bool)
    # genuine gaps: entity col 1 misses t=2; col 2 misses t=0,1; col 0 misses
    # t=3 in group 1 (occlusion-like holes, not prefix truncation only).
    masks[0, 0, 2, 1] = True
    masks[0, 0, 0, 2] = True
    masks[0, 0, 1, 2] = True
    masks[0, 1, 3, 0] = True
    return {
        "trajectory_features": history,
        "unknown_features": current,
        "trajectory_id_labels": ids,
        "unknown_id_labels": ids.clone(),
        "trajectory_times": times,
        "unknown_times": times.clone(),
        "trajectory_masks": masks,
        "unknown_masks": masks.clone(),
    }


def build_e5(seed=11, v=5, c=8, activate=True):
    torch.manual_seed(seed)
    p = CausalEntityPointer(
        feature_dim=c, num_id_vocabulary=v,
        aggregation_mode="equivariant_memory", id_memory_dim=6,
    )
    if activate:
        # Move the gate away from zero and energise both channels so the
        # equivariance test exercises the identity path, not the frozen zero.
        with torch.no_grad():
            p.id_gate[-1].weight.normal_(0, 1.0)
            p.id_gate[-1].bias.normal_(0, 0.5)
            p.id_memory_proj.weight.normal_(0, 1.0)
            p.entity_state_residual[-1].weight.normal_(0, 0.3)
            p.entity_state_reliability[-1].weight.normal_(0, 0.3)
            p.entity_state_time_bias.weight.normal_(0, 0.2)
    return p


def share_channels(src, dst):
    """Make `dst` share every E1/P0 module with `src` (for equivalence tests)."""
    dst.query_proj = src.query_proj
    dst.key_proj = src.key_proj
    dst.new_pointer = src.new_pointer
    dst.query_norm = src.query_norm
    dst.key_norm = src.key_norm
    dst.entity_state_time_bias = src.entity_state_time_bias
    dst.entity_state_reliability = src.entity_state_reliability
    dst.entity_state_residual = src.entity_state_residual
    dst.log_entity_state_tau = src.log_entity_state_tau


def test_zero_init_matches_e1_and_p0():
    seq = sample()
    p_e5 = build_e5(seed=7, activate=False)

    p_e1 = CausalEntityPointer(
        feature_dim=8, num_id_vocabulary=5, aggregation_mode="entity_state",
    )
    share_channels(p_e5, p_e1)
    logits_e5 = p_e5(seq)[0]
    logits_e1 = p_e1(seq)[0]
    assert torch.allclose(logits_e5, logits_e1, atol=1e-5), \
        "zero-init gate must make E5 bit-identical to the E1 appearance channel"

    p_max = CausalEntityPointer(feature_dim=8, num_id_vocabulary=5)
    p_max.query_proj = p_e5.query_proj
    p_max.key_proj = p_e5.key_proj
    p_max.new_pointer = p_e5.new_pointer
    p_max.query_norm = p_e5.query_norm
    p_max.key_norm = p_e5.key_norm
    logits_max = p_max(seq)[0]
    assert torch.allclose(logits_e5, logits_max, atol=1e-4), \
        "fresh E5 (zero E1 residual + zero gate) must equal the P0 max pointer"
    print("test_zero_init_matches_e1_and_p0: PASS")


def test_permutation_equivariance():
    p = build_e5(seed=13, activate=True)
    seq = sample()
    out = p(seq)[0]          # [b, g, t, n, V+1]

    pi = torch.tensor([2, 0, 4, 1, 3])   # old label -> new label
    seq2 = dict(seq)
    seq2["trajectory_id_labels"] = pi[seq["trajectory_id_labels"]]
    out2 = p(seq2)[0]

    v = p.num_id_vocabulary
    for e in range(v):
        assert torch.allclose(out2[..., pi[e]], out[..., e], atol=1e-5), \
            f"relabeling must permute column {e} -> {pi[e]}"
    # NEW column must be untouched by entity renaming.
    assert torch.allclose(out2[..., v], out[..., v], atol=1e-6), \
        "NEW column must be invariant under identity renaming"
    print("test_permutation_equivariance: PASS")


def test_strict_causality():
    p = build_e5(seed=17, activate=True)
    seq = sample()
    first = p(seq)[0]

    # Corrupting the last history row (t=3) cannot change any query at
    # time <= 3: all queries live at times 0..3 and row 3 is invalid for them.
    seq_last = dict(seq)
    seq_last["trajectory_features"] = seq["trajectory_features"].clone()
    seq_last["trajectory_features"][:, :, 3] += 1000.0
    second = p(seq_last)[0]
    assert torch.allclose(first, second, atol=1e-4), \
        "future history must not change any current logits"

    # Corrupting row t=2 may only affect queries at time 3.
    seq_mid = dict(seq)
    seq_mid["trajectory_features"] = seq["trajectory_features"].clone()
    seq_mid["trajectory_features"][:, :, 2] += 1000.0
    third = p(seq_mid)[0]
    assert torch.allclose(third[..., :3, :, :], first[..., :3, :, :], atol=1e-4), \
        "queries at times 0..2 must be unaffected by a t=2 history change"
    changed = not torch.allclose(third[..., 3, :, :], first[..., 3, :, :], atol=1e-4)
    assert changed, "queries at time 3 should see the t=2 history change"
    print("test_strict_causality: PASS")


def test_entity_memory_gap_is_noop():
    p = build_e5(seed=19, activate=True)
    b, g, t, n, c = 1, 1, 3, 1, 8
    history = torch.randn(b, g, t, n, c)
    labels = torch.zeros(b, g, t, n, dtype=torch.int64)
    times = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, n).clone()
    masks = torch.zeros(b, g, t, n, dtype=torch.bool)
    masks[0, 0, 1] = True     # gaps: no observation at t=1 and t=2
    masks[0, 0, 2] = True     # (only t=0 is observed)
    current_times = torch.full((b, g, 1, n), 3, dtype=torch.int64)

    h_q = p._entity_memory_prefixes(
        history_flat=history.reshape(b * g, -1, c),
        labels_flat=labels.reshape(b * g, -1),
        history_times_flat=times.reshape(b * g, -1),
        history_masks_flat=masks.reshape(b * g, -1),
        current_times_flat=current_times.reshape(b * g, -1),
        history_grid_shape=(t, n),
    )   # [1, 1, V, d]

    h_expected = p.id_memory_gru(
        history.reshape(-1, c)[0:1],
        torch.zeros(1, p.id_memory_dim),
    )
    assert torch.allclose(h_q[0, 0, 0], h_expected[0], atol=1e-6), \
        "gap rows must be exact no-ops in the GRU recurrence"
    print("test_entity_memory_gap_is_noop: PASS")


def test_sinkhorn_arbitration():
    from models.runtime_tracker import RuntimeTracker
    tracker = object.__new__(RuntimeTracker)
    tracker.sinkhorn_iters = 3

    torch.manual_seed(5)
    logits = torch.randn(4, 6)          # 4 queries, 5 entities + NEW
    out = tracker._sinkhorn_normalize(logits)

    rows = out.softmax(dim=-1).sum(dim=-1)
    assert torch.allclose(rows, torch.ones(4), atol=1e-5), \
        "Sinkhorn output rows must be normalised"

    pi = torch.tensor([3, 0, 4, 1, 2])
    order = torch.cat([pi, torch.tensor([5])])
    out2 = tracker._sinkhorn_normalize(logits[:, order])
    inv = torch.argsort(pi)     # inv[e] = column of out2 holding original e
    for e in range(5):
        assert torch.allclose(out2[:, inv[e]], out[:, e], atol=1e-5), \
            "Sinkhorn must commute with entity-column permutation"
    assert torch.allclose(out2[:, 5], out[:, 5], atol=1e-6), \
        "NEW column must be invariant"
    print("test_sinkhorn_arbitration: PASS")


def test_single_query_frame_shape():
    """Regression (remote E5 run crash 2026-08-28): a frame with exactly one
    active detection (n_current == 1) must not collapse the Nq axis of h_q.
    The old code appended .squeeze(2) after gather, which swapped the V and
    Nq axes whenever Nq == 1 and crashed torch.stack in the gate features.
    """
    p = build_e5(seed=23, activate=True)
    b, g, t, n, c = 1, 1, 4, 3, 8
    history = torch.randn(b, g, t, n, c)
    labels = torch.randint(0, 3, (b, g, t, n))
    times = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, n).clone()
    masks = torch.zeros((b, g, t, n), dtype=torch.bool)
    seq = {
        "trajectory_features": history,
        "unknown_features": torch.randn(b, g, 1, 1, c),      # ONE query
        "trajectory_id_labels": labels,
        "trajectory_times": times,
        "trajectory_masks": masks,
        "unknown_times": torch.full((b, g, 1, 1), t, dtype=torch.int64),
        "unknown_masks": torch.zeros((b, g, 1, 1), dtype=torch.bool),
    }
    logits = p(seq)[0]
    assert logits.shape == (b, g, 1, 1, p.num_id_vocabulary + 1), \
        f"single-query frame logits must keep all axes, got {tuple(logits.shape)}"
    print("test_single_query_frame_shape: PASS")


if __name__ == "__main__":
    test_zero_init_matches_e1_and_p0()
    test_permutation_equivariance()
    test_strict_causality()
    test_entity_memory_gap_is_noop()
    test_sinkhorn_arbitration()
    test_single_query_frame_shape()
    print("ALL E5 EQUIVARIANT MEMORY TESTS PASSED")
