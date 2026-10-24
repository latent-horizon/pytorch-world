import json

import pytest
import torch

from torch_world import hub
from torch_world.models import LeWM

pytest.importorskip("safetensors")

SPEC = {
    "class": "LeWM",
    "kwargs": {
        "encoder": {
            "class": "CLSEncoder",
            "kwargs": {"img_size": 16, "patch_size": 8, "embed_dim": 32, "depth": 1, "num_heads": 2},
        },
        "action_dim": 2,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "head_dim": 16,
        "mlp_dim": 64,
        "proj_dim": 64,
    },
}


def test_save_and_load_round_trip(tmp_path):
    model = hub.build(SPEC).eval()
    assert isinstance(model, LeWM)
    card = hub.model_card(
        "lewm-test", "A test.", results=[{"task": "reach", "success": "50%"}], details={"data": "none"}
    )
    hub.save_pretrained(model, tmp_path, SPEC, card=card, frameskip=1)
    loaded = hub.from_pretrained(tmp_path)
    pixels = torch.rand(2, 1, 3, 16, 16)
    assert torch.equal(model.encode(pixels), loaded.encode(pixels))
    assert loaded.config["frameskip"] == 1 and not loaded.training
    assert json.loads((tmp_path / "config.json").read_text())["spec"] == SPEC
    readme = (tmp_path / "README.md").read_text()
    assert readme.startswith("---\nlicense: apache-2.0") and "| reach | 50% |" in readme


def test_build_refuses_names_outside_the_library():
    with pytest.raises(ValueError):
        hub.build({"class": "system", "kwargs": {}})
    with pytest.raises(ValueError):
        hub.build({"class": "CLSEncoder", "constructor": "__init_subclass__"})


def test_external_submodules_are_not_stored(tmp_path):
    model = hub.build(SPEC)
    hub.save_pretrained(model, tmp_path, SPEC, external=("encoder",))
    assert json.loads((tmp_path / "config.json").read_text())["external"] == ["encoder"]
    loaded = hub.from_pretrained(tmp_path)
    assert torch.equal(loaded.predictor.blocks[0].attn.qkv.weight, model.predictor.blocks[0].attn.qkv.weight)
