import copy

import torch

from models.motip.entity_pointer import CausalEntityPointer


def make_seq(labels=None):
    torch.manual_seed(7)
    b, g, t, h, n, c = 1, 1, 4, 3, 2, 16
    history = torch.randn(b, g, t, h, c)
    current = torch.randn(b, g, t, n, c)
    if labels is None:
        labels = torch.tensor([0, 0, 1, 0, 1, 1, 2, 2, 0, 1, 2, 2]).reshape(b, g, t, h)
    times_h = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, h)
    times_q = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, n)
    return {
        "trajectory_features": history,
        "unknown_features": current,
        "trajectory_id_labels": labels,
        "unknown_id_labels": torch.zeros(b, g, t, n, dtype=torch.long),
        "trajectory_times": times_h,
        "unknown_times": times_q,
        "trajectory_masks": torch.zeros(b, g, t, h, dtype=torch.bool),
        "unknown_masks": torch.zeros(b, g, t, n, dtype=torch.bool),
    }


def make_pair():
    p0 = CausalEntityPointer(16, 5, aggregation_mode="max")
    e1 = CausalEntityPointer(16, 5, aggregation_mode="entity_state")
    e1.load_state_dict(p0.state_dict(), strict=False)
    return p0, e1


def test_zero_init_is_exact_p0_and_has_gradient():
    p0, e1 = make_pair()
    seq = make_seq()
    p0_logits = p0(seq)[0]
    e1_logits = e1(seq)[0]
    torch.testing.assert_close(e1_logits, p0_logits, rtol=0.0, atol=0.0)
    e1_logits[e1_logits > -100].sum().backward()
    assert e1.entity_state_residual[-1].weight.grad.abs().sum().item() > 0


def test_identity_renaming_only_permutes_entity_columns():
    _, e1 = make_pair()
    seq = make_seq()
    original = e1(seq)[0]
    permutation = torch.tensor([2, 0, 1, 3, 4])
    renamed = copy.deepcopy(seq)
    renamed["trajectory_id_labels"] = permutation[seq["trajectory_id_labels"]]
    renamed_logits = e1(renamed)[0]
    torch.testing.assert_close(
        renamed_logits[..., permutation], original[..., :5], rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        renamed_logits[..., -1], original[..., -1], rtol=0.0, atol=0.0,
    )


def test_future_history_is_strictly_masked():
    _, e1 = make_pair()
    seq = make_seq()
    before = e1(seq)[0]
    altered = copy.deepcopy(seq)
    future = altered["trajectory_times"] >= 2
    altered["trajectory_features"][future] += 1000.0
    after = e1(altered)[0]
    torch.testing.assert_close(after[:, :, :2], before[:, :, :2], rtol=0.0, atol=0.0)
