import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.motip import build as build_motip
from models.motip.equivariant_word_decoder import EquivariantWordDecoder


def make_decoder(num_layers=3, use_aux_loss=False, use_shared_aux_head=True):
    torch.manual_seed(11)
    return EquivariantWordDecoder(
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


class EquivariantWordDecoderTest(unittest.TestCase):
    def test_output_shape_and_finite_scores(self):
        model = make_decoder().eval()
        seq = sample()
        logits, labels, masks = model(seq, use_decoder_checkpoint=False)
        self.assertEqual(logits.shape, (1, 1, 3, 2, 11))
        self.assertIsNone(labels)
        self.assertIs(masks, seq["unknown_masks"])
        self.assertTrue(torch.isfinite(logits).all())

    def test_query_appearance_is_projected_before_tokenization(self):
        model = make_decoder(num_layers=1).eval()
        seq = sample()
        captured = {}

        def capture_queries(**kwargs):
            captured["queries"] = kwargs["queries"].clone()
            return kwargs["queries"]

        with patch.object(model, "_forward_a_layer", side_effect=capture_queries):
            model(seq, use_decoder_checkpoint=False)
        current = model.query_appearance_proj(seq["unknown_features"])
        new_words = model.new_word.view(1, 1, 1, 1, -1).expand_as(current)
        expected = torch.cat([current, new_words], dim=-1).reshape(1, 6, 16)
        torch.testing.assert_close(captured["queries"], expected, rtol=0, atol=0)

    def test_identity_renaming_only_permutes_entity_columns(self):
        model = make_decoder().eval()
        seq_a = sample((2, 7))
        seq_b = {k: v.clone() for k, v in seq_a.items()}
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

    def test_same_time_history_cannot_change_current_logits(self):
        model = make_decoder().eval()
        seq = sample()
        first = model(seq, use_decoder_checkpoint=False)[0]
        seq["trajectory_features"][:, :, 1] += 1000
        second = model(seq, use_decoder_checkpoint=False)[0]
        torch.testing.assert_close(first[:, :, 1], second[:, :, 1], rtol=0, atol=0)

    def test_future_current_queries_cannot_change_past_logits(self):
        model = make_decoder().eval()
        seq = sample()
        first = model(seq, use_decoder_checkpoint=False)[0]
        seq["unknown_features"][:, :, 2] += 1000
        second = model(seq, use_decoder_checkpoint=False)[0]
        torch.testing.assert_close(first[:, :, :2], second[:, :, :2], rtol=0, atol=0)

    def test_absent_entities_are_masked_and_new_remains_reachable(self):
        model = make_decoder().eval()
        logits = model(sample((2, 7)), use_decoder_checkpoint=False)[0]
        self.assertTrue(torch.all(logits[..., 4] == -1.0e4))
        self.assertTrue(torch.isfinite(logits[..., 10]).all())
        self.assertTrue(torch.all(logits[:, :, 0, :, :10] == -1.0e4))
        self.assertTrue(torch.all(logits[:, :, 1:, :, 2] > -1.0e4))
        self.assertTrue(torch.all(logits[:, :, 1:, :, 7] > -1.0e4))

    def test_entity_words_include_own_mention_but_candidates_are_strictly_past(self):
        model = make_decoder().eval()
        seq = sample()
        appearance = model.history_appearance_proj(seq["trajectory_features"])
        frame_times = seq["unknown_times"][..., 0]
        history_words, candidates, valid = model._causal_entity_words(
            history_appearance=appearance,
            history_labels=seq["trajectory_id_labels"],
            history_times=seq["trajectory_times"],
            history_masks=seq["trajectory_masks"],
            frame_times=frame_times,
        )
        changed_appearance = appearance.clone()
        changed_appearance[:, :, 1] += 100
        changed_history_words, changed_candidates, _ = model._causal_entity_words(
            history_appearance=changed_appearance,
            history_labels=seq["trajectory_id_labels"],
            history_times=seq["trajectory_times"],
            history_masks=seq["trajectory_masks"],
            frame_times=frame_times,
        )
        torch.testing.assert_close(
            candidates[:, :, 1], changed_candidates[:, :, 1], rtol=0, atol=0,
        )
        self.assertFalse(torch.allclose(history_words[:, :, 1], changed_history_words[:, :, 1]))
        self.assertFalse(torch.allclose(candidates[:, :, 2], changed_candidates[:, :, 2]))
        self.assertFalse(valid[:, :, 0].any())
        self.assertTrue(valid[:, :, 1:, 2].all())
        self.assertTrue(valid[:, :, 1:, 7].all())

    def test_decoder_contains_no_recurrent_module(self):
        model = make_decoder()
        self.assertFalse(any(
            isinstance(module, (nn.GRU, nn.GRUCell)) for module in model.modules()
        ))

    def test_entity_words_use_parallel_causal_content_pooling(self):
        model = make_decoder().eval()
        seq = sample()
        appearance = model.history_appearance_proj(seq["trajectory_features"])
        history_words, candidates, _ = model._causal_entity_words(
            history_appearance=appearance,
            history_labels=seq["trajectory_id_labels"],
            history_times=seq["trajectory_times"],
            history_masks=seq["trajectory_masks"],
            frame_times=torch.tensor([[[2]]]),
        )
        word_inputs = model.word_input_proj(appearance)[0, 0, :2, 0]
        weights = F.softmax(model.entity_word_pool(word_inputs).squeeze(-1), dim=0)
        expected = torch.sum(weights[:, None] * word_inputs, dim=0)
        torch.testing.assert_close(history_words[0, 0, 0, 0], word_inputs[0])
        torch.testing.assert_close(history_words[0, 0, 1, 0], expected)
        torch.testing.assert_close(candidates[0, 0, 0, 2], expected)

    def test_new_readout_uses_decoded_word_and_unnormalized_dot_product(self):
        model = make_decoder(num_layers=1).eval()
        seq = sample()
        decoded_word = torch.arange(1, 9, dtype=torch.float32) * 2
        with torch.no_grad():
            model.query_readout_layers[0].weight.copy_(torch.eye(8))
            model.entity_readout_layers[0].weight.copy_(torch.eye(8))
            model.log_score_scale.zero_()

        def replace_decoded_words(**kwargs):
            queries = kwargs["queries"].clone()
            queries[..., model.feature_dim:] = decoded_word
            return queries

        with patch.object(model, "_forward_a_layer", side_effect=replace_decoded_words):
            logits = model(seq, use_decoder_checkpoint=False)[0]
        expected = (decoded_word @ model.new_word).expand(6).reshape(1, 1, 3, 2)
        torch.testing.assert_close(logits[..., 10], expected, rtol=0, atol=1e-5)

    def test_entity_readout_uses_candidate_word_for_each_query_time(self):
        model = make_decoder(num_layers=1).eval()
        seq = sample()
        decoded_word = torch.arange(1, 9, dtype=torch.float32)
        with torch.no_grad():
            model.query_readout_layers[0].weight.copy_(torch.eye(8))
            model.entity_readout_layers[0].weight.copy_(torch.eye(8))
            model.log_score_scale.zero_()

        def replace_decoded_words(**kwargs):
            queries = kwargs["queries"].clone()
            queries[..., model.feature_dim:] = decoded_word
            return queries

        with patch.object(model, "_forward_a_layer", side_effect=replace_decoded_words):
            logits = model(seq, use_decoder_checkpoint=False)[0]
        appearance = model.history_appearance_proj(seq["trajectory_features"])
        _, candidates, _ = model._causal_entity_words(
            history_appearance=appearance,
            history_labels=seq["trajectory_id_labels"],
            history_times=seq["trajectory_times"],
            history_masks=seq["trajectory_masks"],
            frame_times=seq["unknown_times"][..., 0],
        )
        for frame in (1, 2):
            expected = decoded_word @ candidates[0, 0, frame, 2]
            actual = logits[0, 0, frame, :, 2]
            torch.testing.assert_close(actual, expected.expand(2), rtol=0, atol=1e-5)

    def test_auxiliary_outputs_cover_every_layer(self):
        model = make_decoder(use_aux_loss=True, use_shared_aux_head=False).train()
        seq = sample()
        logits, labels, masks = model(seq, use_decoder_checkpoint=False)
        self.assertEqual(logits.shape, (3, 1, 3, 2, 11))
        self.assertEqual(labels.shape, (3, 1, 3, 2))
        self.assertEqual(masks.shape, (3, 1, 3, 2))

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
            "history_appearance_proj.",
            "query_appearance_proj.",
            "word_input_proj.",
            "entity_word_pool.",
            "self_attn_layers.",
            "cross_attn_layers.",
            "query_readout_layers.",
            "entity_readout_layers.",
            "new_word",
            "log_score_scale",
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

    def test_build_selects_v4_without_enabling_legacy_pointer(self):
        source_root = Path(__file__).resolve().parents[1]
        config_path = source_root / "configs" / "r50_deformable_detr_motip_dancetrack.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["USE_EQUIVARIANT_WORD_DECODER"] = True
        config["USE_EQUIVARIANT_SET_DECODER"] = False
        config["USE_CAUSAL_ENTITY_POINTER"] = False
        with patch("models.motip.build_deformable_detr", return_value=(nn.Module(), object(), None)):
            model, _ = build_motip(config)
        self.assertIsInstance(model.id_decoder, EquivariantWordDecoder)
        self.assertIsNone(model.entity_pointer)
        self.assertFalse(model.use_entity_pointer)

    def test_v4_config_only_changes_architecture_and_runtime(self):
        source_root = Path(__file__).resolve().parents[1]
        config_dir = source_root / "configs"
        with (config_dir / "r50_deformable_detr_motip_dancetrack.yaml").open(
            "r", encoding="utf-8",
        ) as stream:
            base = yaml.safe_load(stream)
        with (config_dir / "r3_causal_word_decoder_v4.yaml").open(
            "r", encoding="utf-8",
        ) as stream:
            v4 = yaml.safe_load(stream)
        allowed_overrides = {
            "SUPER_CONFIG_PATH",
            "USE_EQUIVARIANT_WORD_DECODER",
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
                self.assertEqual(v4.get(key, official_value), official_value, key)
        self.assertIs(v4["USE_EQUIVARIANT_WORD_DECODER"], True)
        self.assertIs(v4["USE_EQUIVARIANT_SET_DECODER"], False)
        self.assertIs(v4["USE_CAUSAL_ENTITY_POINTER"], False)
        self.assertIs(v4["USE_CAUSAL_HISTORY_PROTOTYPE"], False)
        self.assertIs(v4["USE_SYMMETRIC_HISTORY_PROTOTYPE"], False)
        self.assertIsNone(v4["RESUME_MODEL"])
        self.assertIs(v4["RESUME_OPTIMIZER"], False)
        self.assertIs(v4["RESUME_SCHEDULER"], False)
        self.assertIs(v4["RESUME_STATES"], False)
        self.assertEqual(v4["AUG_TRAJECTORY_SWITCH_PROB"], 0.0)
        self.assertIs(v4["USE_AUX_LOSS"], True)


if __name__ == "__main__":
    unittest.main()
