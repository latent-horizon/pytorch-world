import torch

from torch_world.nn import MLP, AdaLNBlock, Attention, MaskedPredictor, Transformer, VisionTransformer


def test_attention_bool_mask_blocks_tokens():
    attn = Attention(16, num_heads=2)
    x = torch.randn(1, 4, 16)
    mask = torch.tensor([True, True, False, False]).expand(4, 4)
    y = attn(x, attn_mask=mask)
    x2 = x.clone()
    x2[:, 2:] = torch.randn(1, 2, 16)
    assert torch.allclose(attn(x2, attn_mask=mask)[:, :2], y[:, :2], atol=1e-6)


def test_attention_custom_head_dim():
    attn = Attention(10, num_heads=4, head_dim=8)
    assert attn.qkv.out_features == 96 and attn(torch.randn(2, 3, 10)).shape == (2, 3, 10)


def test_mlp_layout():
    mlp = MLP([4, 8, 8, 2], norm=torch.nn.BatchNorm1d)
    kinds = [type(m).__name__ for m in mlp]
    assert kinds == ["Linear", "BatchNorm1d", "GELU", "Linear", "BatchNorm1d", "GELU", "Linear"]


def test_adaln_block_starts_as_identity():
    block = AdaLNBlock(16, 2)
    x, c = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    assert torch.equal(block(x, c), x)


def test_transformer_causal():
    model = Transformer(16, 2, 2).eval()
    x = torch.randn(1, 6, 16)
    y = model(x, is_causal=True)
    x[:, 4:] = 0
    assert torch.allclose(model(x, is_causal=True)[:, :4], y[:, :4], atol=1e-6)


def test_vit_image_video_and_masks():
    vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=3)
    assert vit.grid == (4, 4) and vit(torch.randn(2, 3, 32, 32)).shape == (2, 16, 48)
    masks = [torch.tensor([[0, 1, 2], [5, 6, 7]])]
    assert vit(torch.randn(2, 3, 32, 32), masks).shape == (2, 3, 48)
    video = VisionTransformer(
        img_size=32, patch_size=8, embed_dim=48, depth=1, num_heads=3, num_frames=4, tubelet_size=2
    )
    assert video.grid == (2, 4, 4) and video(torch.randn(2, 3, 4, 32, 32)).shape == (2, 32, 48)


def test_vit_masked_tokens_match_full_pass_without_attention_mixing():
    vit = VisionTransformer(img_size=16, patch_size=8, embed_dim=24, depth=0, num_heads=2)
    x = torch.randn(1, 3, 16, 16)
    assert torch.allclose(vit(x, [torch.tensor([[1, 3]])]), vit(x)[:, [1, 3]], atol=1e-6)


def test_vit_rope_and_cls():
    vit = VisionTransformer(
        img_size=16, patch_size=8, embed_dim=24, depth=1, num_heads=2, use_rope=True, cls_token=True
    )
    assert vit(torch.randn(2, 3, 16, 16)).shape == (2, 5, 24)
    assert vit(torch.randn(2, 3, 16, 16), [torch.tensor([[0, 1], [2, 3]])]).shape == (2, 3, 24)


def test_from_size():
    vit = VisionTransformer.from_size("tiny", img_size=32, patch_size=16)
    assert vit.embed_dim == 192 and len(vit.transformer.blocks) == 12


def test_masked_predictor_shapes_and_order():
    pred = MaskedPredictor((4, 4), embed_dim=32, predictor_dim=16, depth=1, num_heads=2, num_mask_tokens=2)
    B = 3
    masks_x = [torch.randint(0, 16, (B, 5)), torch.randint(0, 16, (B, 5))]
    masks_y = [torch.randint(0, 16, (B, 2)) for _ in range(4)]
    out = pred(torch.randn(2 * B, 5, 32), masks_x, masks_y, mask_index=1)
    assert out.shape == (4 * 2 * B, 2, 32)
    rope = MaskedPredictor((2, 4, 4), embed_dim=32, predictor_dim=24, depth=1, num_heads=2, use_rope=True)
    assert rope(torch.randn(B, 5, 32), masks_x[:1], masks_y[:1]).shape == (B, 2, 32)
