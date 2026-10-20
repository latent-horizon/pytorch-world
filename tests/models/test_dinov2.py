import torch
import torch.nn.functional as F

from torch_world.masks import BlockwiseMask
from torch_world.models import DINOHead, DINOv2, DINOv2Encoder
from torch_world.models.dino import sinkhorn_knopp


def make_model(**kwargs):
    encoder = DINOv2Encoder(img_size=16, patch_size=4, embed_dim=24, depth=2, num_heads=2)
    return DINOv2(encoder, DINOHead(24, num_prototypes=32, hidden_dim=48, bottleneck_dim=16), **kwargs)


def make_batch(B=6, V=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    global_crops, local_crops = torch.randn(2, B, 3, 16, 16, generator=g), torch.randn(V, B, 3, 8, 8, generator=g)
    mask = BlockwiseMask((4, 4), probability=0.5)(2 * B, generator=g).view(2, B, 16)
    return global_crops, local_crops, mask


def reference_sinkhorn(scores, temperature, iterations=3):
    """dinov2/loss/dino_clstoken_loss.py, on a single device."""
    Q = torch.exp(scores / temperature).t()
    B, K = Q.shape[1], Q.shape[0]
    Q /= Q.sum()
    for _ in range(iterations):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    return (Q * B).t()


def reference_losses(model, global_crops, local_crops, mask, teacher_temp):
    """The DINO and iBOT terms as dinov2/train/ssl_meta_arch.py combines them."""
    G, B, V = 2, global_crops.size(1), len(local_crops)
    with torch.no_grad():
        tokens = model.teacher_encoder(global_crops.flatten(0, 1))
        cls = torch.cat(tokens[:, 0].chunk(2)[::-1])  # reversed, so each crop is matched to the other
        teacher_cls = reference_sinkhorn(model.teacher_head(cls), teacher_temp).view(G, B, -1)
        teacher_patches = reference_sinkhorn(model.teacher_head(tokens[:, 1:][mask.flatten(0, 1)]), teacher_temp)
    tokens = model.encoder(global_crops.flatten(0, 1), mask.flatten(0, 1))
    student_local = model.head(model.encoder(local_crops.flatten(0, 1))[:, 0])
    student_global = model.head(tokens[:, 0])
    terms = G * V + (G - 1) * G

    def dino_loss(students, teachers):
        total = 0
        for s in students:
            log_p = F.log_softmax(s / 0.1, dim=-1)
            for t in teachers:
                total -= (t * log_p).sum(-1).mean()
        return total

    dino = dino_loss(student_local.chunk(V), teacher_cls) / terms
    dino += dino_loss([student_global], [teacher_cls.flatten(0, 1)]) * 2 / terms
    masks = mask.flatten(0, 1)
    weight = (1 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]
    student_patches = model.head(tokens[:, 1:][masks])
    ibot = (teacher_patches * F.log_softmax(student_patches / 0.1, dim=-1)).sum(-1)
    ibot = -(ibot * weight).sum() / masks.shape[0] * 2 * (1 / G)
    return dino, ibot


def test_losses_match_reference():
    model = make_model()
    batch = make_batch()
    out = model(*batch, teacher_temp=0.05)
    dino, ibot = reference_losses(model, *batch, teacher_temp=0.05)
    assert torch.allclose(out["dino_loss"], dino, atol=1e-5)
    assert torch.allclose(out["ibot_loss"], ibot, atol=1e-5)
    assert torch.allclose(out["loss"], out["dino_loss"] + out["ibot_loss"] + 0.1 * out["koleo_loss"])


def test_sinkhorn_matches_reference_and_balances_prototypes():
    scores = torch.randn(64, 16)
    assert torch.allclose(sinkhorn_knopp(scores, 0.1), reference_sinkhorn(scores, 0.1), atol=1e-5)
    q = sinkhorn_knopp(scores, 0.1, iterations=100)
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-5)
    assert torch.allclose(q.sum(0), torch.full((16,), 64 / 16), rtol=1e-3)


def test_encoder_resizes_positions_and_masks_patches():
    encoder = DINOv2Encoder(img_size=16, patch_size=4, embed_dim=24, depth=2, num_heads=2).eval()
    x = torch.randn(2, 3, 16, 16)
    assert encoder(x).shape == (2, 1 + 16, 24)
    assert encoder(torch.randn(2, 3, 8, 12)).shape == (2, 1 + 6, 24)
    unmasked = torch.zeros(2, 16, dtype=torch.bool)
    assert torch.equal(encoder(x, unmasked), encoder(x))
    masked = unmasked.clone()
    masked[:, :8] = True  # the top two rows of patches, pixel rows 0 to 7
    y = x.clone()
    y[..., :8, :] = torch.randn_like(y[..., :8, :])
    assert torch.allclose(encoder(x, masked), encoder(y, masked), atol=1e-6)
    assert not torch.allclose(encoder(x), encoder(y))


def test_teacher_gets_no_gradients_and_follows_ema():
    model = make_model()
    model(*make_batch())["loss"].backward()
    assert all(p.grad is None for p in [*model.teacher_encoder.parameters(), *model.teacher_head.parameters()])
    assert model.encoder.mask_token.grad.abs().sum() > 0
    assert model.head.prototypes.parametrizations.weight.original1.grad.abs().sum() > 0
    model.update_teacher(0.0)
    for t, s in zip(model.teacher_encoder.parameters(), model.encoder.parameters()):
        assert torch.equal(t, s)
