import torch

from models.motip.entity_pointer import CausalEntityPointer


def sample(labels=(2, 7)):
    torch.manual_seed(3)
    b, g, t, n, c = 1, 1, 3, 2, 8
    history = torch.randn(b, g, t, n, c)
    current = torch.randn(b, g, t, n, c)
    ids = torch.tensor(labels).view(1, 1, 1, n).expand(b, g, t, n).clone()
    times = torch.arange(t).view(1, 1, t, 1).expand(b, g, t, n).clone()
    masks = torch.zeros((b, g, t, n), dtype=torch.bool)
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


def test_future_history_cannot_change_past_logits():
    model = CausalEntityPointer(feature_dim=8, num_id_vocabulary=10)
    seq = sample()
    first = model(seq)[0]
    seq["trajectory_features"][:, :, 2] += 1000
    second = model(seq)[0]
    assert torch.equal(first[:, :, :2], second[:, :, :2])


def test_identity_renaming_only_permutes_pointer_addresses():
    model = CausalEntityPointer(feature_dim=8, num_id_vocabulary=10)
    seq_a = sample((2, 7))
    seq_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in seq_a.items()}
    seq_b["trajectory_id_labels"][seq_a["trajectory_id_labels"] == 2] = 5
    seq_b["trajectory_id_labels"][seq_a["trajectory_id_labels"] == 7] = 1
    out_a = model(seq_a)[0]
    out_b = model(seq_b)[0]
    assert torch.equal(out_a[..., 2], out_b[..., 5])
    assert torch.equal(out_a[..., 7], out_b[..., 1])
    assert torch.equal(out_a[..., 10], out_b[..., 10])


def test_absent_entities_are_unreachable():
    model = CausalEntityPointer(feature_dim=8, num_id_vocabulary=10)
    out = model(sample((2, 7)))[0]
    assert torch.all(out[..., 4] == -1.0e4)
