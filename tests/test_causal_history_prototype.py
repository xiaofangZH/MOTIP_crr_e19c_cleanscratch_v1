import torch

from models.motip.trajectory_modeling import TrajectoryModeling


def build_enabled():
    return TrajectoryModeling(
        detr_dim=4,
        ffn_dim_ratio=2,
        feature_dim=4,
        use_causal_history_prototype=True,
    )


def test_zero_init_is_exact_identity_for_new_branch():
    module = build_enabled()
    features = torch.randn(2, 1, 5, 3, 4)
    masks = torch.rand(2, 1, 5, 3) < 0.2
    actual = module._mix_causal_history_prototype(features, masks)
    assert torch.equal(actual, features)


def test_prefix_prototype_is_strictly_causal():
    module = build_enabled()
    module.causal_prototype_scale.data.fill_(1.0)
    masks = torch.zeros(1, 1, 5, 1, dtype=torch.bool)
    first = torch.randn(1, 1, 5, 1, 4)
    second = first.clone()
    second[:, :, 4] += 1000.0
    out_first = module._mix_causal_history_prototype(first, masks)
    out_second = module._mix_causal_history_prototype(second, masks)
    assert torch.equal(out_first[:, :, :4], out_second[:, :, :4])


def test_masked_tokens_do_not_enter_prefix_mean():
    module = build_enabled()
    module.causal_prototype_scale.data.fill_(1.0)
    features = torch.tensor([[[[[2.0]], [[100.0]], [[4.0]]]]])
    masks = torch.tensor([[[[False], [True], [False]]]])
    actual = module._mix_causal_history_prototype(features, masks)
    assert actual[0, 0, 2, 0, 0].item() == 3.0
    assert actual[0, 0, 1, 0, 0].item() == 100.0


def test_scale_receives_finite_gradient():
    module = build_enabled()
    features = torch.randn(1, 1, 4, 2, 4, requires_grad=True)
    masks = torch.zeros(1, 1, 4, 2, dtype=torch.bool)
    module._mix_causal_history_prototype(features, masks).square().mean().backward()
    assert module.causal_prototype_scale.grad is not None
    assert torch.isfinite(module.causal_prototype_scale.grad)
