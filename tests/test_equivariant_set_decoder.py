import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.motip import build as build_motip
from models.motip.equivariant_set_decoder import EquivariantSetDecoder


def make_decoder():
    torch.manual_seed(11)
    return EquivariantSetDecoder(
        feature_dim=8,
        id_dim=8,
        ffn_dim_ratio=2,
        num_layers=3,
        head_dim=4,
        num_id_vocabulary=10,
        rel_pe_length=6,
        use_aux_loss=False,
        use_shared_aux_head=True,
    )


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


class EquivariantSetDecoderTest(unittest.TestCase):
    def test_output_shape_and_finite_scores(self):
        model = make_decoder().eval()
        seq = sample()
        logits, _, masks = model(seq, use_decoder_checkpoint=False)
        self.assertEqual(logits.shape, (1, 1, 3, 2, 11))
        self.assertIs(masks, seq["unknown_masks"])
        self.assertTrue(torch.isfinite(logits).all())

    def test_identity_renaming_only_permutes_entity_columns(self):
        model = make_decoder().eval()
        seq_a = sample((2, 7))
        seq_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in seq_a.items()}
        seq_b["trajectory_id_labels"][seq_a["trajectory_id_labels"] == 2] = 5
        seq_b["trajectory_id_labels"][seq_a["trajectory_id_labels"] == 7] = 1
        out_a = model(seq_a, use_decoder_checkpoint=False)[0]
        out_b = model(seq_b, use_decoder_checkpoint=False)[0]
        torch.testing.assert_close(out_a[..., 2], out_b[..., 5], rtol=0, atol=1e-5)
        torch.testing.assert_close(out_a[..., 7], out_b[..., 1], rtol=0, atol=1e-5)
        torch.testing.assert_close(out_a[..., 10], out_b[..., 10], rtol=0, atol=1e-5)

    def test_future_history_cannot_change_past_logits(self):
        model = make_decoder().eval()
        seq = sample()
        first = model(seq, use_decoder_checkpoint=False)[0]
        seq["trajectory_features"][:, :, 2] += 1000
        second = model(seq, use_decoder_checkpoint=False)[0]
        torch.testing.assert_close(first[:, :, :2], second[:, :, :2], rtol=0, atol=0)

    def test_absent_entities_are_masked_and_new_remains_reachable(self):
        model = make_decoder().eval()
        logits = model(sample((2, 7)), use_decoder_checkpoint=False)[0]
        self.assertTrue(torch.all(logits[..., 4] == -1.0e4))
        self.assertTrue(torch.isfinite(logits[..., 10]).all())
        self.assertTrue(torch.all(logits[:, :, 0, :, :10] == -1.0e4))

    def test_readout_selects_best_individual_mention_without_pre_pooling(self):
        model = EquivariantSetDecoder(
            feature_dim=8,
            id_dim=8,
            ffn_dim_ratio=2,
            num_layers=1,
            head_dim=4,
            num_id_vocabulary=10,
            rel_pe_length=6,
            use_aux_loss=False,
            use_shared_aux_head=True,
        ).eval()
        for projection in (model.query_proj, model.history_proj):
            with torch.no_grad():
                projection.weight.zero_()
                projection.weight[:8].copy_(torch.eye(8))
                projection.bias.zero_()
        for readouts in (model.query_readout_layers, model.entity_readout_layers):
            with torch.no_grad():
                readouts[0].weight.copy_(torch.eye(16))
        with torch.no_grad():
            model.new_token.zero_()
            model.entity_state_residual[-1].weight.zero_()
            model.entity_state_residual[-1].bias.zero_()
            model.id_gate[-1].weight.zero_()
            model.id_gate[-1].bias.zero_()
        model.score_scale = 1.0

        appearance = torch.randn(8)
        appearance = appearance / appearance.norm()
        seq = {
            "trajectory_features": torch.stack([appearance, -appearance]).view(1, 1, 2, 1, 8),
            "unknown_features": appearance.view(1, 1, 1, 1, 8),
            "trajectory_id_labels": torch.full((1, 1, 2, 1), 2, dtype=torch.long),
            "unknown_id_labels": torch.full((1, 1, 1, 1), 2, dtype=torch.long),
            "trajectory_times": torch.arange(2).view(1, 1, 2, 1),
            "unknown_times": torch.full((1, 1, 1, 1), 2, dtype=torch.long),
            "trajectory_masks": torch.zeros((1, 1, 2, 1), dtype=torch.bool),
            "unknown_masks": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
        }
        with patch.object(model, "_forward_a_layer", side_effect=lambda **kwargs: kwargs["queries"]):
            logits = model(seq, use_decoder_checkpoint=False)[0]

        # The clean mention is a perfect match; the corrupted mention is its
        # opposite. Pre-pooling them cannot preserve the best-match score.
        torch.testing.assert_close(logits[0, 0, 0, 0, 2], torch.tensor(1.0), rtol=0, atol=1e-5)

    def test_backward_reaches_all_decoder_components(self):
        model = make_decoder().train()
        seq = sample()
        target = seq["unknown_id_labels"].clone()
        target[:, :, 0] = model.num_id_vocabulary
        logits = model(seq, use_decoder_checkpoint=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, 11), target.reshape(-1))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for prefix in (
            "query_proj.",
            "history_proj.",
            "self_attn_layers.",
            "cross_attn_layers.",
            "new_token",
        ):
            self.assertTrue(any(
                name.startswith(prefix) and parameter.grad.abs().sum() > 0
                for name, parameter in model.named_parameters()
            ), prefix)

    def test_checkpointed_and_eager_decoders_match(self):
        model = make_decoder().train()
        seq = sample()
        eager = model(seq, use_decoder_checkpoint=False)[0]
        checkpointed = model(seq, use_decoder_checkpoint=True)[0]
        torch.testing.assert_close(eager, checkpointed, rtol=0, atol=1e-5)

    @staticmethod
    def _attention_records(model, seq, attention_type):
        records = []
        handles = []
        modules = model.cross_attn_layers if attention_type == "cross" else model.self_attn_layers

        def hook(module, args, kwargs, output):
            records.append({
                "key": kwargs.get("key"),
                "attn_mask": kwargs.get("attn_mask"),
            })

        for module in modules:
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        with torch.no_grad():
            model(seq, use_decoder_checkpoint=False)
        for handle in handles:
            handle.remove()
        return records

    @staticmethod
    def _blocked(mask):
        if mask.dtype == torch.bool:
            return mask
        return torch.isneginf(mask) | (mask < -1.0e3)

    def test_cross_attention_retains_individual_history_mentions(self):
        model = make_decoder().eval()
        seq = sample()
        records = self._attention_records(model, seq, "cross")
        expected_history = model.history_proj(seq["trajectory_features"].reshape(1, 6, 8))

        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record["key"].shape, (1, 7, 16))
            torch.testing.assert_close(record["key"][:, :6], expected_history)
            torch.testing.assert_close(
                record["key"][:, 6], model.new_token.view(1, 16).expand(1, 16),
            )
        for record in records[1:]:
            torch.testing.assert_close(records[0]["key"], record["key"], rtol=0, atol=0)

        corrupted = {k: v.clone() for k, v in seq.items()}
        corrupted["trajectory_features"][:, :, 1, 0] += 100
        corrupted_records = self._attention_records(model, corrupted, "cross")
        for clean, dirty in zip(records, corrupted_records):
            torch.testing.assert_close(clean["key"][:, 0], dirty["key"][:, 0], rtol=0, atol=0)
            self.assertFalse(torch.allclose(clean["key"][:, 2], dirty["key"][:, 2]))

    def test_cross_attention_mask_is_causal_for_each_mention(self):
        model = make_decoder().eval()
        records = self._attention_records(model, sample(), "cross")
        for record in records:
            mask = record["attn_mask"]
            self.assertIsNotNone(mask)
            self.assertEqual(mask.shape, (model.n_heads, 6, 7))
            blocked = self._blocked(mask)
            for query_time in range(3):
                for history_time in range(3):
                    actual = blocked[:, query_time * 2, history_time * 2]
                    if history_time >= query_time:
                        self.assertTrue(actual.all())
                    else:
                        self.assertFalse(actual.any())
            self.assertFalse(blocked[:, :, 6].any())

    def test_future_current_queries_cannot_change_past_logits(self):
        model = make_decoder().eval()
        seq = sample()
        first = model(seq, use_decoder_checkpoint=False)[0]
        seq["unknown_features"][:, :, 2] += 1000
        second = model(seq, use_decoder_checkpoint=False)[0]
        torch.testing.assert_close(first[:, :, :2], second[:, :, :2], rtol=0, atol=0)

    def test_build_selects_r1_without_enabling_legacy_pointer(self):
        source_root = Path(__file__).resolve().parents[1]
        config_path = source_root / "configs" / "r50_deformable_detr_motip_dancetrack.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["USE_EQUIVARIANT_SET_DECODER"] = True
        config["USE_CAUSAL_ENTITY_POINTER"] = False

        with patch(
            "models.motip.build_deformable_detr",
            return_value=(nn.Module(), object(), None),
        ):
            model, _ = build_motip(config)

        self.assertIsInstance(model.id_decoder, EquivariantSetDecoder)
        self.assertIsNone(model.entity_pointer)
        self.assertFalse(model.use_entity_pointer)

    def test_r1_config_only_changes_architecture_and_runtime(self):
        source_root = Path(__file__).resolve().parents[1]
        config_dir = source_root / "configs"
        with (config_dir / "r50_deformable_detr_motip_dancetrack.yaml").open(
            "r", encoding="utf-8",
        ) as stream:
            base = yaml.safe_load(stream)
        with (config_dir / "r1_equivariant_set_decoder_v1.yaml").open(
            "r", encoding="utf-8",
        ) as stream:
            r1 = yaml.safe_load(stream)

        allowed_overrides = {
            "SUPER_CONFIG_PATH",
            "USE_EQUIVARIANT_SET_DECODER",
            "USE_CAUSAL_ENTITY_POINTER",
            "USE_CAUSAL_HISTORY_PROTOTYPE",
            "USE_SYMMETRIC_HISTORY_PROTOTYPE",
            "USE_AUX_LOSS",
            "AUG_TRAJECTORY_SWITCH_PROB",
            "RESUME_MODEL",
            "RESUME_STRICT",
            "RESUME_OPTIMIZER",
            "RESUME_SCHEDULER",
            "RESUME_STATES",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "USE_WANDB",
        }
        for key, official_value in base.items():
            if key not in allowed_overrides:
                self.assertEqual(r1.get(key, official_value), official_value, key)

        self.assertIs(r1["USE_EQUIVARIANT_SET_DECODER"], True)
        self.assertIs(r1["USE_CAUSAL_ENTITY_POINTER"], False)
        self.assertIs(r1["USE_CAUSAL_HISTORY_PROTOTYPE"], False)
        self.assertIs(r1["USE_SYMMETRIC_HISTORY_PROTOTYPE"], False)
        self.assertIsNone(r1["RESUME_MODEL"])
        self.assertIs(r1["RESUME_OPTIMIZER"], False)
        self.assertIs(r1["RESUME_SCHEDULER"], False)
        self.assertIs(r1["RESUME_STATES"], False)
        self.assertEqual(r1["AUG_TRAJECTORY_SWITCH_PROB"], 0.0)
        self.assertIs(r1["USE_AUX_LOSS"], True)
        self.assertEqual(r1.get("DETR_NUM_CHECKPOINT_FRAMES", base["DETR_NUM_CHECKPOINT_FRAMES"]), 5)


if __name__ == "__main__":
    unittest.main()
