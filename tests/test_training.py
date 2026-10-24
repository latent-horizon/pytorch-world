import torch
from torch.utils.data import DataLoader

from torch_world import evaluation
from torch_world.datasets import TrajectoryDataset
from torch_world.envs import TwoRooms
from torch_world.losses import SIGReg
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM
from torch_world.utils import fit


def lewm(action_dim=2):
    encoder = CLSEncoder(img_size=16, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    return LeWM(
        encoder, action_dim, embed_dim=32, depth=1, num_heads=2, head_dim=16, mlp_dim=64, proj_dim=64, sigreg=SIGReg(32)
    )


def test_fit_logs_losses_and_leaves_eval_mode():
    data = TwoRooms(batch_size=16, size=16).collect(8, generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TrajectoryDataset(data, 4), batch_size=32, shuffle=True, drop_last=True)
    model = lewm()
    log = fit(model, loader, epochs=2, clip=1.0)
    assert set(log) == {"loss", "pred", "sigreg"} and len(log["loss"]) == 2 * len(loader)
    assert not model.training


def test_goal_policy_with_and_without_chunks():
    env = TwoRooms(batch_size=4, size=16)
    start, goal = env.tasks(torch.Generator().manual_seed(0))
    planner = CEM(horizon=2, action_dim=2, num_samples=8, num_elites=2, iterations=1, bounds=(-1, 1))
    policy = evaluation.goal_policy(lewm().eval(), planner, env.render(goal))
    assert policy(env.reset(start)).shape == (4, 2)

    chunked = CEM(horizon=2, action_dim=6, num_samples=8, num_elites=2, iterations=1, bounds=(-1, 1))
    policy = evaluation.goal_policy(lewm(6).eval(), chunked, env.render(goal), action_dim=2, execute=1)
    actions = [policy(env.reset(start)) for _ in range(4)]
    assert all(a.shape == (4, 2) for a in actions) and policy.calls == 2
