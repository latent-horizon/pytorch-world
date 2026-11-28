import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_world.benchmarks.safety_robustness import calibrate, forecast_metrics
from torch_world.datasets import TrajectoryDataset
from torch_world.envs import ObstacleArm
from torch_world.models import PLDM, LeWM
from torch_world.safety import FailureMargin
from torch_world.utils import fit


def arm_clearance(env):
    kind = env.mj.mjtGeom
    (obstacle,) = (i for i, t in enumerate(env.model.geom_type) if t == kind.mjGEOM_BOX)
    links = [i for i, t in enumerate(env.model.geom_type) if t == kind.mjGEOM_CAPSULE]
    distances = []
    for data in env.data:
        env.mj.mj_kinematics(env.model, data)
        distances.append(min(env.mj.mj_geomDistance(env.model, data, obstacle, link, 1.0, None) for link in links))
    return torch.tensor(distances)


def arm_replay(n, seed, horizon, candidates):
    gen = torch.Generator().manual_seed(seed)
    with ObstacleArm(batch_size=n * candidates, observation="state") as env:
        start = torch.rand(n, 4, generator=gen)
        start[:, 0] = -0.3 * start[:, 0]
        start[:, 1] = start[:, 1] - 0.5
        start[:, 2:] = 0
        env.reset(start.repeat_interleave(candidates, 0))
        states, distances = [env.state.clone()], [arm_clearance(env)]
        actions = torch.rand(n * candidates, horizon, 2, generator=gen) * 2 - 1
        actions[:, 1:] = (0.8 * actions[:, :1] + 0.2 * actions[:, 1:]).clamp(-1, 1)
        for command in actions.unbind(1):
            states.append(env.step(command))
            distances.append(arm_clearance(env))
    states = torch.stack(states, 1)
    return {
        "states": states,
        "actions": torch.cat([actions, torch.zeros_like(actions[:, :1])], 1),
        "clearance": torch.stack(distances, 1),
    }


@torch.no_grad()
def arm_scores(model, margin, data, candidates):
    states, actions = data["states"], data["actions"][:, :-1]
    z = model.encode(states[:, :1])
    pred = model.rollout(z, actions)[:, 1:]
    observed = model.encode(states[:, 1:])
    truth = data["clearance"][:, 1:] < 0.01
    return {
        (source, h): (
            -margin.score(latents[:, :h]).amin(1).reshape(-1, candidates),
            truth[:, :h].any(1).reshape(-1, candidates),
        )
        for source, latents in (("imagined", pred), ("observed", observed))
        for h in (4, 8)
    }


def run_arm_audit(seed=7):
    torch.manual_seed(seed)
    train = arm_replay(128, seed, 24, 1)
    calibration = arm_replay(48, 101, 8, 8)
    test = arm_replay(64, 203, 8, 8)
    loader = DataLoader(TrajectoryDataset(train, window=4), batch_size=64, shuffle=True, drop_last=True)

    def encoder():
        return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 32))

    models = {
        "PLDM": PLDM(encoder(), 2, latent_dim=32),
        "LeWM": LeWM(encoder(), 2, embed_dim=32, depth=2, num_heads=2, head_dim=16, mlp_dim=64, proj_dim=64),
    }
    rows = []
    for name, model in models.items():
        fit(model, loader, epochs=5, lr=0.001, inputs=lambda b: (b["states"], b["actions"]))
        model.eval().requires_grad_(False)
        margin = FailureMargin(32, hidden=(64, 64))
        labels = train["clearance"].flatten() < 0.01
        frames = TensorDataset(model.encode(train["states"]).flatten(0, 1), labels)
        fit(margin, DataLoader(frames, batch_size=64, shuffle=True), epochs=8, lr=0.001, inputs=lambda b: b)
        calibration_scores = arm_scores(model, margin, calibration, 8)
        for key, (scores, events) in arm_scores(model, margin, test, 8).items():
            edges, rates = calibrate(*calibration_scores[key])
            row = forecast_metrics(rates[torch.bucketize(scores, edges)], events, 0.1)
            row.update({"model": name, "source": key[0], "horizon": key[1], "seed": seed})
            rows.append(row)
    return rows
