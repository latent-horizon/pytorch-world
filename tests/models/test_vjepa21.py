import torch
import torch.nn.functional as F

from torch_world.masks import MultiBlock3DMask, apply_masks
from torch_world.models.jepa._vjepa21 import VJEPA21, VJEPA21Encoder, VJEPA21Predictor


def make_model(images=False, **kwargs):
    encoder = VJEPA21Encoder(
        img_size=16, patch_size=4, embed_dim=24, depth=4, num_heads=2, num_frames=4, use_rope=True, images=images
    )
    predictor = VJEPA21Predictor(
        encoder.grid, 24, 12, depth=1, num_heads=2, num_mask_tokens=2, use_rope=True, images=images
    )
    return VJEPA21(encoder, predictor, **kwargs)


def masks(B, grid=(2, 4, 4)):
    g = torch.Generator().manual_seed(0)
    short = MultiBlock3DMask(grid, spatial_scale=(0.15, 0.15), num_blocks=8)(B, g)
    long = MultiBlock3DMask(grid, spatial_scale=(0.7, 0.7), num_blocks=2)(B, g)
    return short[0] + long[0], short[1] + long[1]


def test_encoder_levels():
    encoder = make_model().encoder
    x = torch.randn(2, 3, 4, 16, 16)
    assert encoder.levels == [0, 1, 2, 3]
    assert encoder(x).shape == (2, 32, 24)
    levels = encoder(x, all_levels=True)
    assert levels.shape == (2, 32, 96)
    assert torch.allclose(levels[..., -24:], encoder(x))


def test_predictor_shapes():
    model = make_model()
    (m_enc, *_), (m_pred, *_) = masks(2)
    z, c = model.predictor(model.encoder(torch.randn(2, 3, 4, 16, 16), [m_enc], all_levels=True), [m_enc], [m_pred])
    assert z.shape == (2, m_pred.size(1), 96) and c.shape == (2, m_enc.size(1), 96)


def test_loss_components_and_gradients():
    model = make_model(context_coeff=0.5)
    out = model(torch.randn(2, 3, 4, 16, 16), *masks(2))
    assert all(torch.isfinite(v) for v in out.values())
    assert torch.allclose(out["loss"], out["pred_loss"] + 0.5 * out["context_loss"])
    out["loss"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)
    assert all(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_context_coeff_override_and_distance_weights():
    model = make_model()
    x, (masks_enc, masks_pred) = torch.randn(2, 3, 4, 16, 16), masks(2)
    with torch.no_grad():
        out = model(x, masks_enc, masks_pred, context_coeff=0.0)
        unweighted = make_model(distance_weighting=False)
        unweighted.load_state_dict(model.state_dict())
        plain = unweighted(x, masks_enc, masks_pred)
    assert torch.allclose(out["loss"], out["pred_loss"])
    assert out["context_loss"] < plain["context_loss"]
    w = model.distance_weights(masks_enc[0], masks_pred[0])
    assert w.shape == masks_enc[0].shape and (w <= 1).all() and (w > 0).all()


def test_targets_are_layer_normed_levels():
    model = make_model(context_coeff=0.0)
    x, (masks_enc, masks_pred) = torch.randn(2, 3, 4, 16, 16), masks(2)
    with torch.no_grad():
        h = model.target_encoder(x, all_levels=True).unflatten(-1, (4, 24))
        h = F.layer_norm(h, (24,)).flatten(-2)
        expected = sum(
            (model.predictor(model.encoder(x, [e], all_levels=True), [e], [p], i)[0] - apply_masks(h, [p])).abs().mean()
            for i, (e, p) in enumerate(zip(masks_enc, masks_pred))
        )
        assert torch.allclose(model(x, masks_enc, masks_pred)["pred_loss"], expected / 2)


def test_images_use_their_own_tokenizer():
    model = make_model(images=True)
    images = torch.randn(2, 3, 16, 16)
    assert model.encoder(images).shape == (2, 16, 24)
    out = model(images, *masks(2, grid=(1, 4, 4)))
    out["loss"].backward()
    assert model.encoder.image_embed.proj.weight.grad is not None
    assert model.encoder.patch_embed.proj.weight.grad is None
    assert model.encoder.modality_embed.grad[1].abs().sum() > 0


def test_ema_target_update():
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.randn(2, 3, 4, 16, 16), *masks(2))["loss"].backward()
    opt.step()
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target(0.5)
    for b, t, o in zip(before, model.target_encoder.parameters(), model.encoder.parameters()):
        assert torch.allclose(t, 0.5 * b + 0.5 * o)
