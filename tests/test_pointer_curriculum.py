import random

import torch

from data.transforms import TurnIntoTrajectoryAndUnknown
from train import apply_structured_pointer_history_corruption


def test_history_blackout_keeps_unknown_visible_and_creates_new_targets():
    random.seed(7)
    annotations = []
    for t in range(5):
        annotations.append({
            "id_labels": torch.tensor([[[0, 1]]]),
            "id_masks": torch.zeros((1, 1, 2), dtype=torch.bool),
            "ann_idxs": torch.tensor([[[0, 1]]]),
            "times": torch.tensor([[[t, t]]]),
        })
    transform = TurnIntoTrajectoryAndUnknown(
        num_id_vocabulary=50,
        aug_trajectory_occlusion_prob=0.0,
        aug_trajectory_switch_prob=0.0,
        pointer_history_blackout_prob=1.0,
        pointer_history_blackout_min=2,
        pointer_history_blackout_max=2,
        pointer_history_prefix_prob=1.0,
    )
    _, output, _ = transform(None, annotations, None)
    assert output[0]["trajectory_id_masks"].all()
    assert output[1]["trajectory_id_masks"].all()
    assert not output[0]["unknown_id_masks"].any()
    assert torch.all(output[0]["unknown_id_labels"] == 50)
    assert torch.all(output[1]["unknown_id_labels"] == 50)
    assert torch.all(output[2]["unknown_id_labels"] == 50)
    assert torch.all(output[3]["unknown_id_labels"] < 50)


def test_structured_corruption_is_persistent_and_does_not_touch_labels():
    torch.manual_seed(3)
    features = torch.zeros((1, 1, 5, 2, 3))
    features[..., 0, :] = 1.0
    features[..., 1, :] = 2.0
    seq = {
        "trajectory_features": features,
        "trajectory_boxes": torch.zeros((1, 1, 5, 2, 4)),
        "trajectory_masks": torch.zeros((1, 1, 5, 2), dtype=torch.bool),
        "trajectory_id_labels": torch.tensor([[[[0, 1]] * 5]]),
    }
    labels_before = seq["trajectory_id_labels"].clone()
    output = apply_structured_pointer_history_corruption(seq, probability=1.0)
    changed = output["trajectory_features"][0, 0, :, 0, 0] == 2.0
    assert changed.any()
    first_changed = int(torch.nonzero(changed)[0])
    assert changed[first_changed:].all()
    assert torch.equal(output["trajectory_id_labels"], labels_before)
