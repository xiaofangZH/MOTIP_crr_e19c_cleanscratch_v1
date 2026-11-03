import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from models.motip.equivariant_set_decoder import EquivariantSetDecoder


def make_decoder(use_aux_loss=False, use_shared_aux_head=True):
    torch.manual_seed(11)
    return EquivariantSetDecoder(
        feature_dim=8,
        id_dim=8,
        ffn_dim_ratio=2,
        num_layers=3,
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


def entity_state_inputs(model):
    with torch.no_grad():
        for parameter in model.entity_state_reliability.parameters():
            parameter.zero_()
    return {
        "mention_scores": torch.tensor([[[0.2, 0.2]]]),
        "valid_address": torch.zeros((1, 1, 2), dtype=torch.long),
        "invalid": torch.zeros((1, 1, 2), dtype=torch.bool),
        "history_tokens": torch.zeros((1, 2, model.hidden_dim)),
        "history_times": torch.tensor([[1.0, 2.0]]),
        "current_times": torch.tensor([[4.0]]),
    }


class R3EntityRefinementTest(unittest.TestCase):
    def test_history_support_is_an_entity_score_feature(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_entity_state_features"))
        features, _, count = model._entity_state_features(**entity_state_inputs(model))
        self.assertEqual(count[0, 0, 0].item(), 2.0)
        torch.testing.assert_close(
            features[0, 0, 0, 1],
            torch.tensor(math.log(3.0) / math.log(31.0)),
        )

    def test_causal_age_is_an_entity_score_feature(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_entity_state_features"))
        features, _, _ = model._entity_state_features(**entity_state_inputs(model))
        expected = (math.log(4.0) + math.log(3.0)) / (2.0 * math.log(31.0))
        torch.testing.assert_close(features[0, 0, 0, 3], torch.tensor(expected))

    def test_reliability_is_an_entity_score_feature(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_entity_state_features"))
        features, _, _ = model._entity_state_features(**entity_state_inputs(model))
        torch.testing.assert_close(features[0, 0, 0, 5], torch.tensor(0.5))
        torch.testing.assert_close(features[0, 0, 0, 6], torch.tensor(0.5))

    def test_identity_memory_is_causal(self):
        model = make_decoder()
        self.assertTrue(hasattr(model, "_entity_memory_prefixes"))
        torch.manual_seed(23)
        history_tokens = torch.randn(1, 4, model.hidden_dim)
        kwargs = {
            "history_tokens": history_tokens,
            "history_labels": torch.tensor([[2, 7, 2, 7]]),
            "history_times": torch.tensor([[0, 0, 1, 1]]),
            "history_masks": torch.zeros((1, 4), dtype=torch.bool),
            "current_times": torch.tensor([[1.0, 2.0]]),
            "history_grid_shape": (2, 2),
        }
        first = model._entity_memory_prefixes(**kwargs)
        kwargs["history_tokens"] = history_tokens.clone()
        kwargs["history_tokens"][:, 2:] += 1000
        second = model._entity_memory_prefixes(**kwargs)
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first[:, 1], second[:, 1]))

    def test_identity_memory_receives_entity_loss_gradient(self):
        model = make_decoder().train()
        self.assertTrue(hasattr(model, "id_memory_gru"))
        seq = sample()
        target = seq["unknown_id_labels"].clone()
        target[:, :, 0] = model.num_id_vocabulary
        logits = model(seq)[0]
        F.cross_entropy(logits.reshape(-1, 11), target.reshape(-1)).backward()
        self.assertGreater(model.id_memory_gru.weight_ih.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.id_memory_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.id_gate[-1].weight.grad.abs().sum().item(), 0.0)

    def test_auxiliary_layers_all_receive_id_gradients(self):
        model = make_decoder(use_aux_loss=True, use_shared_aux_head=False).train()
        seq = sample()
        target = seq["unknown_id_labels"].clone()
        target[:, :, 0] = model.num_id_vocabulary
        logits, labels, masks = model(seq)
        self.assertEqual(logits.shape, (3, 1, 3, 2, 11))
        self.assertEqual(labels.shape, (3, 1, 3, 2))
        self.assertEqual(masks.shape, (3, 1, 3, 2))
        valid = ~masks[:, :, 1:, :]
        loss = F.cross_entropy(
            logits[:, :, 1:, :, :][valid],
            labels[:, :, 1:, :][valid],
        )
        loss.backward()
        for layer in range(model.num_layers):
            self.assertGreater(model.query_readout_layers[layer].weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(model.entity_readout_layers[layer].weight.grad.abs().sum().item(), 0.0)

    def test_r3_config_disables_switch_and_enables_auxiliary_id_loss(self):
        source_root = Path(__file__).resolve().parents[1]
        config_path = source_root / "configs" / "r1_equivariant_set_decoder_v1.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["AUG_TRAJECTORY_SWITCH_PROB"], 0.0)
        self.assertIs(config["USE_AUX_LOSS"], True)


if __name__ == "__main__":
    unittest.main()
