import torch
from torch import nn

from torch_world import evaluation
from torch_world.envs import TwoRooms
from torch_world.models import VLAJEPA
from torch_world.nn import WordTokenizer
from torch_world.planning import RecedingHorizon


def vlajepa(**kwargs):
    vision = nn.Sequential(nn.Flatten(2), nn.Linear(8 * 8, 16))  # (N, 3, 8, 8) -> 3 tokens of 16
    target = nn.Sequential(nn.Unflatten(1, (3, 1)), nn.Flatten(2), nn.Linear(64, 12))  # 3 tokens of 12
    model = VLAJEPA(
        vision, target, vocab_size=10, action_dim=2, vision_dim=16, target_dim=12, num_patches=3, frames=4,
        chunk_size=3, latent_tokens=2, action_tokens=4, dim=16, depth=2, num_heads=2, predictor_dim=16,
        predictor_depth=2, predictor_heads=2, expert_dim=16, expert_depth=2, expert_heads=2, flow_repeats=2, **kwargs,
    )  # fmt: skip
    for block in model.expert.blocks:  # adaLN-zero gates start closed: open them so the condition matters
        nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def batch(B=3):
    ids = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])[:B]
    return torch.randn(B, 4, 3, 8, 8), ids, ids != 0, torch.rand(B, 3, 2) * 2 - 1


def test_loss_components_and_gradients():
    model = vlajepa()
    pixels, ids, mask, actions = batch()
    out = model(pixels, ids, mask, actions)
    assert all(torch.isfinite(v) for v in out.values())
    assert torch.allclose(out["loss"], out["action_loss"] + 0.1 * out["wm_loss"])
    out["loss"].backward()
    for p in (model.vision[1].weight, model.word_embed.weight, model.latent_query, model.action_query):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert model.predictor_proj.weight.grad.abs().sum() > 0 and model.action_out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())  # frozen target encoder


def test_video_without_actions_trains_world_model_only():
    model = vlajepa()
    pixels, ids, mask, _ = batch()
    out = model(pixels, ids, mask)
    assert set(out) == {"loss", "wm_loss"} and torch.equal(out["loss"], out["wm_loss"])
    out["loss"].backward()
    assert model.action_out.weight.grad is None and model.latent_query.grad.abs().sum() > 0


def test_precomputed_latents_match_encoding():
    model = vlajepa()
    pixels, ids, mask, actions = batch()
    z = model.encode(pixels)
    assert z.shape == (3, 4, 3, 12)
    torch.manual_seed(0)
    a = model(pixels, ids, mask, actions)["wm_loss"]
    torch.manual_seed(0)
    b = model(pixels[:, :1], ids, mask, actions, latents=z)["wm_loss"]
    assert torch.allclose(a, b)


def test_future_frames_never_reach_the_policy():
    model = vlajepa().eval()
    pixels, ids, mask, _ = batch()
    other = pixels.clone()
    other[:, 1:] = torch.randn_like(other[:, 1:])
    g = torch.Generator().manual_seed(0)
    a = model.sample_actions(pixels[:, 0], ids, mask, generator=g)
    b = model.sample_actions(other[:, 0], ids, mask, generator=torch.Generator().manual_seed(0))
    assert a.shape == (3, 3, 2) and torch.equal(a, b)
    c = model.sample_actions(pixels[:, 0], ids.flip(0), mask.flip(0), generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(a, c)  # the instruction matters


def test_vlm_is_causal_and_ignores_padding():
    model = vlajepa().eval()
    image = torch.randn(1, 3, 8, 8)
    ids, mask = torch.tensor([[1, 2, 0]]), torch.tensor([[True, True, False]])
    latent, cond = model.context(image, ids, mask)
    assert latent.shape == (1, 3, 2, 16) and cond.shape == (1, 4, 16)
    _, cond2 = model.context(image, torch.tensor([[1, 2, 7]]), mask)  # change a padding id
    assert torch.allclose(cond, cond2, atol=1e-6)
    # embodied-action tokens come last, so the latent-action tokens cannot see them
    with torch.no_grad():
        model.action_query.add_(1.0)
    latent3, cond3 = model.context(image, ids, mask)
    assert torch.allclose(latent, latent3, atol=1e-6) and not torch.allclose(cond, cond3)


def test_predictor_is_frame_causal_and_rolls_out():
    model = vlajepa().eval()
    z, latent = torch.randn(2, 3, 3, 12), torch.randn(2, 3, 2, 16)
    z2, latent2 = z.clone(), latent.clone()
    z2[:, 2], latent2[:, 2] = torch.randn_like(z2[:, 2]), torch.randn_like(latent2[:, 2])
    with torch.no_grad():
        p, p2 = model.predict(z, latent), model.predict(z2, latent2)
        future = model.predict_future(z, latent)
        actions, imagined = model.imagine(torch.randn(2, 3, 8, 8), torch.tensor([[1, 2], [3, 0]]))
    assert p.shape == (2, 3, 3, 12) and torch.allclose(p[:, :2], p2[:, :2], atol=1e-5)
    assert not torch.allclose(p[:, 2], p2[:, 2])
    assert future.shape == (2, 3, 3, 12) and torch.allclose(future[:, 0], p[:, 0], atol=1e-5)
    assert actions.shape == (2, 3, 2) and imagined.shape == (2, 4, 3, 12)


def test_policy_runs_closed_loop_in_two_rooms():
    texts = ["go left", "go right"]
    tokenizer = WordTokenizer(texts)
    vision = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(2), nn.Linear(16, 16))
    target = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(1), nn.Linear(12, 12))
    model = VLAJEPA(
        vision, target, len(tokenizer), action_dim=2, vision_dim=16, target_dim=12, frames=3, chunk_size=4, dim=16,
        depth=1, num_heads=2, predictor_dim=16, predictor_depth=1, predictor_heads=2, expert_dim=16, expert_depth=1,
        expert_heads=2,
    ).eval()  # fmt: skip
    env = TwoRooms(batch_size=2, size=16)
    start, goal = env.tasks(generator=torch.Generator().manual_seed(0))
    ids, mask = tokenizer(texts)
    policy = RecedingHorizon(lambda obs: model.sample_actions(obs, ids, mask), action_dim=2, execute=2)
    result = evaluation.goal_reaching(env, {"VLA-JEPA": policy}, start, goal, steps=5)
    assert result.paths["VLA-JEPA"].shape == (2, 6, 2) and policy.calls == 3
