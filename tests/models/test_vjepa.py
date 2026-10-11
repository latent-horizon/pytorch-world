import torch
import torch.nn.functional as F

from torch_world.masks import MultiBlock3DMask, apply_masks
from torch_world.models import VJEPA
from torch_world.nn import MaskedPredictor, VisionTransformer


def make_model(use_rope=False, **kwargs):
    encoder = VisionTransformer(
        img_size=16, patch_size=4, embed_dim=24, depth=1, num_heads=2, num_frames=4, use_rope=use_rope
    )
    predictor = MaskedPredictor(
        encoder.grid, 24, 12, depth=1, num_heads=2, num_mask_tokens=2, zero_init_mask_tokens=True, use_rope=use_rope
    )
    return VJEPA(encoder, predictor, **kwargs)


def masks(B):
    g = torch.Generator().manual_seed(0)
    short = MultiBlock3DMask((2, 4, 4), spatial_scale=(0.15, 0.15), num_blocks=8)(B, g)
    long = MultiBlock3DMask((2, 4, 4), spatial_scale=(0.7, 0.7), num_blocks=2)(B, g)
    return short[0] + long[0], short[1] + long[1]


def test_loss_exp_two_is_half_mse():
    model = make_model(loss_exp=2.0)
    x = torch.randn(2, 3, 4, 16, 16)
    masks_enc, masks_pred = masks(2)
    with torch.no_grad():
        h = F.layer_norm(model.target_encoder(x), (24,))
        expected = (
            sum(
                F.mse_loss(model.predictor(model.encoder(x, [e]), [e], [p], mask_index=i), apply_masks(h, [p])) / 2
                for i, (e, p) in enumerate(zip(masks_enc, masks_pred))
            )
            / 2
        )
        out = model(x, masks_enc, masks_pred)
    assert torch.allclose(out["loss"], expected) and torch.allclose(out["jepa_loss"], expected)


def test_each_strategy_uses_its_mask_token():
    model = make_model()
    out = model(torch.randn(2, 3, 4, 16, 16), *masks(2))
    out["loss"].backward()
    assert (model.predictor.mask_tokens.grad.flatten(1).abs().sum(1) > 0).all()


def test_rope_variant_and_variance_regularizer():
    model = make_model(use_rope=True, reg_coeff=1.0)
    out = model(torch.randn(2, 3, 4, 16, 16), *masks(2))
    assert torch.allclose(out["loss"], out["jepa_loss"] + out["reg_loss"])
    assert out["reg_loss"] > 0
