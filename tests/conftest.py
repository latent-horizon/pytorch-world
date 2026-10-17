import pytest
import torch


@pytest.fixture(scope="session")
def without_shadows():
    """Skip shadows and multisampling, which dominate software rendering in CI, where visuals are not under test."""

    def disable(env):
        env.render_shadows = False
        env.model.vis.quality.offsamples = 0
        return env

    return disable


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
