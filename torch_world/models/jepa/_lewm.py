import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP, Transformer


class LeWM(WorldModel):
    """LeWorldModel: an end-to-end JEPA world model trained from pixels with two terms, next-embedding MSE and SIGReg.

    Each frame is encoded to one embedding (the encoder's CLS token) and projected. A causal transformer, conditioned
    on action embeddings through adaLN-zero, predicts the next embedding from the last ``history`` frames. There is no
    EMA target and no stop-gradient: ``loss = mse + lamb * sigreg``.

    Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels", 2026.
    Paper: https://arxiv.org/abs/2603.19312
    Code: https://github.com/lucas-maes/le-wm

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)`` (ViT-tiny CLS in the paper).
        action_dim: dimension of one action, times the frameskip when actions are stacked.
        state_dim: optional measured-state coordinates appended to the visual latent.
            Callers normalize these externally. States are predicted, not held fixed, during planning.
        rollout_steps: optional autoregressive training horizon. Values above one add
            rollout MSE to the original teacher-forced objective and require longer frame sequences.
        state_weight: relative weight of state MSE versus visual MSE, in both training and planning.
        rollout_weight: coefficient of the additional autoregressive loss.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        history: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        proj_dim: int = 2048,
        dropout: float = 0.1,
        lamb: float = 0.09,
        sigreg: SIGReg | None = None,
        state_dim: int = 0,
        rollout_steps: int = 1,
        state_weight: float = 1.0,
        rollout_weight: float = 1.0,
    ):
        super().__init__()
        if state_dim < 0 or rollout_steps < 1 or state_weight < 0 or rollout_weight < 0:
            raise ValueError("Invalid state dimensions, rollout length or loss weights")
        self.history = history
        self.state_dim, self.embed_dim = state_dim, embed_dim
        self.rollout_steps = rollout_steps
        self.state_weight, self.rollout_weight = state_weight, rollout_weight
        latent_dim = embed_dim + state_dim
        self.lamb = lamb
        self.encoder = encoder
        self.projector = MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * latent_dim, latent_dim], act=nn.SiLU)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, history, latent_dim))
        self.predictor = Transformer(
            latent_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.pred_proj = MLP([latent_dim, proj_dim, latent_dim], norm=nn.BatchNorm1d)
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor, states: Tensor | None = None) -> Tensor:
        """Encode pixels, optionally appending externally normalized measured state.

        State coordinates remain explicit and are predicted during rollout;
        future measured states are never supplied to the predictor.
        """
        z = self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])
        if self.state_dim:
            if states is None or states.shape != (*pixels.shape[:2], self.state_dim):
                raise ValueError("Measured states must match batch, time and state_dim")
            z = torch.cat((z, states), dim=-1)
        return z

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        T = z.size(1)
        h = self.predictor(z + self.pos_embed[:, :T], self.action_encoder(actions), is_causal=True)
        return self.pred_proj(h.flatten(0, 1)).unflatten(0, (z.size(0), T))

    def _distance(self, a: Tensor, b: Tensor) -> Tensor:
        error = (a - b).square()
        visual = error[..., : self.embed_dim].mean(-1)
        if self.state_dim:
            visual = visual + self.state_weight * error[..., self.embed_dim :].mean(-1)
        return visual

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        return self._distance(z[:, -1], goal)

    def forward(self, pixels: Tensor, actions: Tensor, states: Tensor | None = None) -> dict[str, Tensor]:
        """Use ``history + rollout_steps`` frames; the final padded action is unused.

        Defaults retain the original one-step objective. Optional rollout loss
        backpropagates through imagined futures, with no teacher-forced future
        state or visual inputs. SIGReg applies only to learned visual features.
        """
        length = self.history + self.rollout_steps
        if pixels.size(1) < length or actions.size(1) < length - 1:
            raise ValueError("Not enough frames/actions for the configured training rollout")
        z = self.encode(pixels[:, :length], None if states is None else states[:, :length])
        pred = self.predict(z[:, : self.history], actions[:, : self.history])
        mse = self._distance(pred, z[:, 1 : self.history + 1]).mean()
        sigreg = self.sigreg(z[..., : self.embed_dim].transpose(0, 1))
        out = {"loss": mse + self.lamb * sigreg, "pred_loss": mse, "sigreg_loss": sigreg}
        if self.rollout_steps > 1:
            imagined = self.rollout(z[:, : self.history], actions[:, : length - 1])[:, self.history :]
            rollout_loss = self._distance(imagined, z[:, self.history :]).mean()
            out["rollout_loss"] = rollout_loss
            out["loss"] = out["loss"] + self.rollout_weight * rollout_loss
        if self.state_dim:
            out["state_loss"] = (
                (pred[..., self.embed_dim :] - z[:, 1 : self.history + 1, self.embed_dim :]).square().mean()
            )
        return out
