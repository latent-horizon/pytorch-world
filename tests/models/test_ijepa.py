import torch
import torch.nn.functional as F

from torch_world.masks import MultiBlockMask, apply_masks
from torch_world.models import IJEPA
from torch_world.nn import MaskedPredictor, VisionTransformer


def make_model():
    encoder = VisionTransformer(img_size=32, patch_size=4, embed_dim=32, depth=2, num_heads=4)
    return IJEPA(encoder, MaskedPredictor(encoder.grid, 32, predictor_dim=16, depth=1, num_heads=2))


def batch(B=4):
    masks_enc, masks_pred = MultiBlockMask((8, 8), min_keep=2)(B, generator=torch.Generator().manual_seed(0))
    return torch.randn(B, 3, 32, 32), masks_enc, masks_pred


def test_loss_is_smooth_l1_to_layer_normed_target_patches():
    model = make_model()
    x, masks_enc, masks_pred = batch()
    with torch.no_grad():
        h = F.layer_norm(model.target_encoder(x), (32,))
        target = torch.cat([apply_masks(h, [m]) for m in masks_pred])
        z = model.predictor(model.encoder(x, masks_enc), masks_enc, masks_pred)
        assert torch.allclose(model(x, masks_enc, masks_pred)["loss"], F.smooth_l1_loss(z, target))


def test_target_encoder_is_frozen_copy():
    model = make_model()
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    for a, b in zip(model.encoder.parameters(), model.target_encoder.parameters()):
        assert torch.equal(a, b)
    model(*batch())["loss"].backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert model.encoder.patch_embed.proj.weight.grad is not None


def test_update_target_is_ema():
    model = make_model()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target(0.9)
    for t0, t, o in zip(before, model.target_encoder.parameters(), model.encoder.parameters()):
        assert torch.allclose(t, 0.9 * t0 + 0.1 * o)


def test_training_reduces_loss():
    model = make_model()
    x, masks_enc, masks_pred = batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        loss = model(x, masks_enc, masks_pred)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target(0.996)
        losses.append(loss.item())
    assert losses[-1] < losses[0]
