"""Opt-in bridge to a pinned, separately obtained LeWAM reference checkout.

Only the named upstream definitions are loaded, avoiding unrelated environment
and training dependencies. The upstream encoder, transformer, loss and SIGReg
are executed unchanged. No reference source is bundled or downloaded here.
"""

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

REFERENCE_COMMIT = "f57659fb26858082975794d37a8f3d380ddc1fa7"


def load_reference(root):
    root = Path(root)
    scope = {"torch": torch, "nn": nn, "F": F, "math": math, "DictConfig": SimpleNamespace}
    hashes = {}
    selections = {
        "lewam/models/module.py": {"sinusoid", "MLP", "SIGReg"},
        "lewam/models/encoders.py": {"Encoder", "batchnorm_to_groupnorm", "ResNetEncoder"},
        "lewam/models/lewam.py": {"MoTBlock", "LeWAM"},
        "scripts/train_lewam.py": {"reshape_view_major", "sigreg_loss", "run_batch", "lr_scale"},
    }
    for filename, names in selections.items():
        path = root / filename
        raw = path.read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw)
        tree.body = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
        if {x.name for x in tree.body} != names:
            raise ValueError(f"Reference definitions missing from {path}")
        exec(compile(tree, str(path), "exec"), scope)  # noqa: S102 - explicitly supplied reference source
        if filename.endswith("encoders.py"):
            scope["build_encoder"] = lambda backbone, output, hidden, **kw: scope["ResNetEncoder"](output, hidden)
    scope["_IMG_MEAN"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    scope["_IMG_STD"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return scope, hashes


class ReferenceLeWAM(nn.Module):
    """Physical-action API around the actual upstream single-view model."""

    def __init__(self, config, reference, action_mean, action_std):
        super().__init__()
        self.functions, self.source_hashes = load_reference(reference)
        self.config = config
        cfg = {
            "fs": config.frameskip,
            "num_actions_pred": config.chunk_size,
            "num_states_pred": 1,
            "action_raw_dim": 2,
            "history_len": config.history,
            "goal_type": "sampled",
            "n_flow_steps": config.flow_steps,
            "state_head": "mse",
            "action_head": "flow",
            "z_dim": 192,
            "d_model": 192,
            "n_heads": 4,
            "depth": 4,
            "dropout": 0.1,
            "encoder_backbone": "resnet18dp",
            "encoder_size": "tiny",
            "proj_hidden": 384,
            "img_size": config.size,
            "sigreg_grouped": True,
        }
        self.model = self.functions["LeWAM"](cfg)
        self.sigreg = self.functions["SIGReg"]()
        self.register_buffer("action_mean", action_mean.clone())
        self.register_buffer("action_std", action_std.clone())
        self.goal_type, self.num_states, self.state_dim = "sampled", 1, 0
        self.frameskip, self.chunk_size, self.action_dim = config.frameskip, config.chunk_size, 2
        self.train_config = SimpleNamespace(
            train=SimpleNamespace(encoder_checkpoint_chunks=0, p_drop_goal=0.0, w_act=1.0, w_dyn=1.0, w_reg=0.04),
            model=SimpleNamespace(goal_type="sampled", num_states_pred=1, sigreg_grouped=True),
            data=SimpleNamespace(frameskip=config.frameskip),
        )

    def encode(self, pixels):
        return self.model.encode(pixels.flatten(0, 1)).unflatten(0, pixels.shape[:2])

    def normalize_actions(self, actions):
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions, *, clamp=False):
        raw = actions * self.action_std + self.action_mean
        return raw.clamp(-1, 1) if clamp else raw

    def sample_actions(self, z, goal, *, history_pad=None, remaining=None, generator=None, clamp=True):
        pad = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device) if history_pad is None else history_pad
        actions = self.model.predict_actions(z, pad, goal, remaining, generator=generator)
        return self.denormalize_actions(actions, clamp=clamp)

    def predict_future(self, z, actions, goal=None, *, history_pad=None):
        pad = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device) if history_pad is None else history_pad
        return self.model.predict_states(z, pad, self.normalize_actions(actions))

    def loss(self, batch):
        mean = batch["history_pixels"].new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = batch["history_pixels"].new_tensor([0.229, 0.224, 0.225])[:, None, None]

        def raw(key):
            return ((batch[key] * std + mean) * 255).round().clamp(0, 255).byte()

        actions = self.normalize_actions(batch["actions"]).masked_fill(~batch["action_valid"][..., None], 0)
        packed = (
            raw("history_pixels"),
            batch["history_pad"],
            actions,
            batch["action_valid"],
            raw("future_pixels"),
            batch["state_valid"],
            raw("goal_pixels"),
            batch["remaining"],
        )
        loss, terms, _ = self.functions["run_batch"](
            packed, self.model, None, self.sigreg, self.train_config, str(actions.device), self.training
        )
        return {"loss": loss, "action_loss": terms["act"], "pred_loss": terms["state"], "sigreg_loss": terms["reg"]}


def initialize_local_from_reference(local, reference):
    """Copy every active neural parameter; normalization stays identical."""
    upstream = reference.model
    local.encoder.trunk.load_state_dict(upstream.encoder.trunk.state_dict())
    local.encoder.kp_conv.load_state_dict(upstream.encoder.kp_conv.state_dict())
    local.projector.load_state_dict(upstream.encoder.projector.net.state_dict())
    for ours, theirs in {
        "history_projection": "hist_proj",
        "goal_projection": "goal_proj",
        "action_projection": "action_proj",
        "time_projection": "tau_proj",
        "horizon_projection": "h_proj",
        "state_output": "state_out",
        "action_output": "action_out",
    }.items():
        getattr(local, ours).load_state_dict(getattr(upstream, theirs).state_dict())
    with torch.no_grad():
        for ours, theirs in {
            "history_position": "hist_pos_embed",
            "action_position": "action_pos_embed",
            "state_queries": "next_state_query",
            "state_position": "next_state_pos",
        }.items():
            getattr(local, ours).copy_(getattr(upstream, theirs))
        local.token_type.weight.copy_(torch.cat((upstream.state_type.weight, upstream.action_type.weight)))
    for ours, theirs in zip(local.blocks, upstream.blocks):
        ours.modulation.load_state_dict(theirs.ada_action.state_dict())
        for name in ("qkv", "out", "mlp"):
            for i, stream in enumerate(("state", "action")):
                getattr(ours, name)[i].load_state_dict(getattr(theirs, f"{name}_{stream}").state_dict())
