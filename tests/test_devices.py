import pytest
import torch

from torch_world.envs import TwoRooms
from torch_world.losses import SIGReg
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM

DEVICES = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")),
]


def lewm(device):
    encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    model = LeWM(encoder, action_dim=2, embed_dim=32, depth=1, num_heads=2, head_dim=16, mlp_dim=64, proj_dim=64)
    model.sigreg = SIGReg(32)
    return model.to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_train_and_plan_on_device(device):
    env = TwoRooms(batch_size=8, size=32, device=device)
    data = env.collect(4, generator=torch.Generator().manual_seed(0))
    assert data["pixels"].device.type == device
    model = lewm(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        loss = model(data["pixels"], data["actions"])["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        z, goal = model.encode(data["pixels"][:, :1]), model.encode(data["pixels"][:, -1:])[:, 0]
    plan = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))(model, z, goal)
    assert plan.shape == (8, 3, 2) and plan.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bf16_autocast_forward():
    model = lewm("cuda")
    pixels, actions = torch.rand(4, 4, 3, 32, 32, device="cuda"), torch.rand(4, 4, 2, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(pixels, actions)
    out["loss"].backward()
    assert torch.isfinite(out["loss"].float())
