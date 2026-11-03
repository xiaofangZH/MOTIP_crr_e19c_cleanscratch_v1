import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.motip.equivariant_set_decoder import EquivariantSetDecoder
from models.motip.trajectory_modeling import TrajectoryModeling


INVALID = -1.0e4


def make_decoder(num_layers=3, use_aux_loss=False, use_shared_aux_head=True):
    torch.manual_seed(11)
    return EquivariantSetDecoder(
        feature_dim=8,
        id_dim=8,
        ffn_dim_ratio=2,
        num_layers=num_layers,
        head_dim=4,
        num_id_vocabulary=10,
        rel_pe_length=6,
        use_aux_loss=use_aux_loss,
        use_shared_aux_head=use_shared_aux_head,
    )


def sample():
    torch.manual_seed(3)
    b, g, t, n, c = 1, 1, 3, 2, 8
    history = torch.randn(b, g, t, n, c)
    current = torch.randn(b, g, t, n, c)
    ids = torch.tensor((2, 7)).view(1, 1, 1, n).expand(b, g, t, n).clone()
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


class NewCalibrationTest(unittest.TestCase):
    def test_new_score_uses_entity_logmeanexp_as_negative_feedback(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_calibrate_new_scores"))
        entity_scores = torch.tensor([[
            [[2.0, 2.0, INVALID], [0.0, 0.0, INVALID]],
        ]])
        new_base = torch.tensor([[[3.0, 1.0]]])

        calibrated = model._calibrate_new_scores(entity_scores, new_base)

        torch.testing.assert_close(
            calibrated,
            torch.tensor([[[2.5, 1.0]]]),
            rtol=0,
            atol=1.0e-6,
        )

    def test_new_context_is_zero_when_no_entity_has_evidence(self):
        model = make_decoder()
        entity_scores = torch.full((1, 1, 2, 3), INVALID)
        new_base = torch.tensor([[[1.25, -0.5]]])

        calibrated = model._calibrate_new_scores(entity_scores, new_base)

        torch.testing.assert_close(calibrated, new_base, rtol=0, atol=0)

    def test_new_head_reads_the_pre_decoder_query_not_the_cross_attended_query(self):
        model = make_decoder(num_layers=1).eval()
        with torch.no_grad():
            model.new_context_logit.fill_(-100.0)
        seq = sample()

        with patch.object(
            model,
            "_forward_a_layer",
            side_effect=lambda **kwargs: kwargs["queries"],
        ):
            first = model(seq, use_decoder_checkpoint=False)[0]
        with patch.object(
            model,
            "_forward_a_layer",
            side_effect=lambda **kwargs: kwargs["queries"] + 100.0,
        ):
            second = model(seq, use_decoder_checkpoint=False)[0]

        self.assertFalse(torch.allclose(first[..., :-1], second[..., :-1]))
        torch.testing.assert_close(first[..., -1], second[..., -1], rtol=0, atol=0)


class SameFrameConflictTest(unittest.TestCase):
    def test_lower_queries_are_suppressed_while_column_max_is_unchanged(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_suppress_same_frame_conflicts"))
        scores = torch.tensor([[
            [[8.0, 0.0], [4.0, 0.0], [2.0, 0.0]],
        ]])

        suppressed = model._suppress_same_frame_conflicts(scores)

        expected = torch.tensor([[
            [[8.0, 0.0], [3.0, 0.0], [0.5, 0.0]],
        ]])
        torch.testing.assert_close(suppressed, expected, rtol=0, atol=1.0e-6)

    def test_each_frame_suppresses_only_its_own_queries(self):
        model = make_decoder()
        scores = torch.tensor([[
            [[8.0, 1.0], [4.0, 1.0], [2.0, 1.0]],
            [[0.5, 9.0], [0.4, 5.0], [0.3, 1.0]],
        ]])

        first = model._suppress_same_frame_conflicts(scores)
        changed = scores.clone()
        changed[:, 1] += 100.0
        second = model._suppress_same_frame_conflicts(changed)

        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first[:, 1], second[:, 1]))

    def test_padded_queries_neither_compete_nor_change(self):
        model = make_decoder()
        scores = torch.tensor([[
            [[8.0], [2.0], [4.0]],
        ]])
        query_valid = torch.tensor([[[True, False, True]]])

        suppressed = model._suppress_same_frame_conflicts(scores, query_valid)

        expected = torch.tensor([[
            [[8.0], [2.0], [3.0]],
        ]])
        torch.testing.assert_close(suppressed, expected, rtol=0, atol=1.0e-6)

    def test_conflict_suppression_is_equivariant_to_entity_renaming(self):
        model = make_decoder()
        torch.manual_seed(17)
        scores = torch.randn(1, 2, 3, 4)

        renamed = scores[..., [2, 0, 3, 1]]
        suppressed = model._suppress_same_frame_conflicts(scores)
        renamed_suppressed = model._suppress_same_frame_conflicts(renamed)

        torch.testing.assert_close(
            suppressed[..., [2, 0, 3, 1]],
            renamed_suppressed,
            rtol=0,
            atol=0,
        )

    def test_conflict_and_new_parameters_receive_entity_gradients(self):
        model = make_decoder(use_aux_loss=True, use_shared_aux_head=False).train()
        seq = sample()
        logits, labels, masks = model(seq)
        valid = ~masks
        F.cross_entropy(logits[valid], labels[valid]).backward()

        self.assertIsNotNone(model.new_query_head.weight.grad)
        self.assertGreater(model.new_query_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.conflict_logit.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.new_context_logit.grad.abs().sum().item(), 0.0)


class QueryHistorySymmetryTest(unittest.TestCase):
    def test_query_and_history_receive_the_same_tokenwise_transform(self):
        torch.manual_seed(29)
        model = TrajectoryModeling(
            detr_dim=8,
            ffn_dim_ratio=2,
            feature_dim=8,
        ).eval()
        features = torch.randn(1, 1, 3, 2, 8)
        seq = {
            "trajectory_features": features.clone(),
            "unknown_features": features.clone(),
            "trajectory_masks": torch.zeros((1, 1, 3, 2), dtype=torch.bool),
        }

        output = model(seq)

        self.assertEqual(output["unknown_features"].shape, features.shape)
        torch.testing.assert_close(
            output["trajectory_features"],
            output["unknown_features"],
            rtol=0,
            atol=0,
        )

    def test_query_transform_is_pointwise_and_cannot_mix_other_queries(self):
        torch.manual_seed(31)
        model = TrajectoryModeling(
            detr_dim=8,
            ffn_dim_ratio=2,
            feature_dim=8,
        ).eval()
        features = torch.randn(1, 1, 3, 2, 8)

        def make_seq():
            return {
                "trajectory_features": torch.zeros_like(features),
                "unknown_features": features.clone(),
                "trajectory_masks": torch.zeros((1, 1, 3, 2), dtype=torch.bool),
            }

        first = model(make_seq())["unknown_features"]
        self.assertFalse(torch.allclose(first, features))
        changed = make_seq()
        changed["unknown_features"][:, :, 2, 1] += 100.0
        second = model(changed)["unknown_features"]

        torch.testing.assert_close(first[:, :, :2], second[:, :, :2], rtol=0, atol=0)
        self.assertFalse(
            torch.allclose(first[:, :, 2, 1], second[:, :, 2, 1])
        )

    def test_query_path_receives_gradient_through_shared_transform(self):
        torch.manual_seed(37)
        model = TrajectoryModeling(
            detr_dim=8,
            ffn_dim_ratio=2,
            feature_dim=8,
        ).train()
        features = torch.randn(1, 1, 3, 2, 8)
        seq = {
            "trajectory_features": torch.zeros_like(features),
            "unknown_features": features,
            "trajectory_masks": torch.zeros((1, 1, 3, 2), dtype=torch.bool),
        }

        model(seq)["unknown_features"].sum().backward()

        self.assertIsNotNone(model.ffn.linear2.weight.grad)
        self.assertGreater(model.ffn.linear2.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
