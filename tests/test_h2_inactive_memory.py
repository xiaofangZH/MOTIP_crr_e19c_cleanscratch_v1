from collections import OrderedDict
from types import SimpleNamespace

import torch

from models.runtime_tracker import RuntimeTracker


class FakeModel:
    def __init__(self, winning_label=7):
        self.id_decoder = SimpleNamespace(rel_pe_length=30)
        self.winning_label = winning_label
        self.seen_seq_info = None

    def __call__(self, *, seq_info, part):
        if part == "trajectory_modeling":
            self.seen_seq_info = seq_info
            return seq_info
        assert part == "id_decoder"
        count = seq_info["unknown_features"].shape[3]
        logits = torch.full((1, 1, 1, count, 51), -10.0)
        logits[..., self.winning_label] = 10.0
        logits[..., 50] = 0.0
        return logits, None, None


def make_tracker(enabled=True):
    tracker = RuntimeTracker.__new__(RuntimeTracker)
    tracker.model = FakeModel()
    tracker.dtype = torch.float32
    tracker.use_sigmoid = False
    tracker.num_id_vocabulary = 50
    tracker.use_h2_inactive_memory = enabled
    tracker.h2_max_inactive_age = 60
    tracker.h2_recovery_thresh = 0.2
    tracker.h2_inactive_memory = OrderedDict({
        7: {"feature": torch.ones(256), "box": torch.ones(4), "age": 35},
    })
    tracker.h2_stats = {
        "archived": 0, "queried_detections": 0, "recovered": 0,
        "expired": 0, "evicted_for_birth": 0,
    }
    return tracker


def test_h2_changes_only_newborn_and_recovers_once():
    tracker = make_tracker()
    predictions = torch.tensor([3, 50, 50])
    output = tracker._h2_recover_newborns(
        boxes=torch.zeros((3, 4)),
        output_embeds=torch.zeros((3, 256)),
        active_predictions=predictions,
    )
    assert output.tolist() == [3, 7, 50]
    assert tracker.h2_stats["recovered"] == 1
    assert 7 not in tracker.h2_inactive_memory


def test_h2_time_indices_are_strictly_past_and_bounded():
    tracker = make_tracker()
    tracker._h2_recover_newborns(
        boxes=torch.zeros((1, 4)),
        output_embeds=torch.zeros((1, 256)),
        active_predictions=torch.tensor([50]),
    )
    seq = tracker.model.seen_seq_info
    assert torch.all(seq["trajectory_times"] < seq["unknown_times"])
    assert int(seq["trajectory_times"].min()) >= 0
    assert int(seq["unknown_times"].max()) < 30


def test_archive_uses_last_real_observation_before_cut():
    tracker = make_tracker()
    tracker.h2_inactive_memory.clear()
    tracker.trajectory_masks = torch.tensor([[False], [True], [True], [True]])
    tracker.trajectory_features = torch.arange(4 * 256, dtype=torch.float32).view(4, 1, 256)
    tracker.trajectory_boxes = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
    tracker.trajectory_id_labels = torch.full((4, 1), 9, dtype=torch.int64)
    tracker._h2_archive_tracks_expiring_on_cut(cut_start=2, current_id_labels=torch.tensor([3]))
    assert list(tracker.h2_inactive_memory) == [9]
    assert torch.equal(tracker.h2_inactive_memory[9]["feature"], tracker.trajectory_features[0, 0])
    assert tracker.h2_inactive_memory[9]["age"] == 4


def test_track_recovered_at_timeout_is_not_archived():
    tracker = make_tracker()
    tracker.h2_inactive_memory.clear()
    tracker.trajectory_masks = torch.tensor([[False], [True], [True], [True]])
    tracker.trajectory_features = torch.zeros((4, 1, 256))
    tracker.trajectory_boxes = torch.zeros((4, 1, 4))
    tracker.trajectory_id_labels = torch.full((4, 1), 9, dtype=torch.int64)
    tracker._h2_archive_tracks_expiring_on_cut(cut_start=2, current_id_labels=torch.tensor([9]))
    assert not tracker.h2_inactive_memory


def test_stale_memory_label_cannot_duplicate_an_active_match():
    tracker = make_tracker()
    predictions = torch.tensor([7, 50])
    output = tracker._h2_recover_newborns(
        boxes=torch.zeros((2, 4)),
        output_embeds=torch.zeros((2, 256)),
        active_predictions=predictions,
    )
    assert output.tolist() == [7, 50]
    assert len(torch.unique(output)) == len(output)
    assert 7 not in tracker.h2_inactive_memory


def test_memory_expires_deterministically():
    tracker = make_tracker()
    tracker.h2_inactive_memory[7]["age"] = 60
    tracker._h2_age_memory()
    assert not tracker.h2_inactive_memory
    assert tracker.h2_stats["expired"] == 1
