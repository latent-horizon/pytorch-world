import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
