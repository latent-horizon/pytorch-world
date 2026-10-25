import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from torch_world.models.base import WorldModel
from torch_world.models.tdmpc._tdmpc2 import symexp, symlog
from torch_world.nn import MLP
from torch_world.utils.ema import ema_update


def two_hot(x: Tensor, bins: Tensor) -> Tensor:
    """Scalars ``(...)`` to two-hot targets ``(..., K)`` over ``bins`` ``(K,)``, evenly spaced in symlog space."""
    lo, hi, k = bins[0], bins[-1], bins.numel()
    pos = (symlog(x).clamp(lo, hi) - lo) / (hi - lo) * (k - 1)
    idx = pos.floor().clamp(max=k - 2)
    offset = (pos - idx)[..., None]
    idx = idx.long()[..., None]
    out = x.new_zeros(*x.shape, k).scatter(-1, idx, 1 - offset)
    return out.scatter_add(-1, idx + 1, offset)


def two_hot_inv(logits: Tensor, bins: Tensor) -> Tensor:
    """Logits ``(..., K)`` to the scalars ``(...)``: ``symexp`` of the expected bin."""
    return symexp((logits.softmax(-1) * bins).sum(-1))


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of ``(N, C, H, W)`` feature maps."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def conv_encoder(in_channels: int = 3, depth: int = 32, img_size: int = 64) -> nn.Sequential:
    """DreamerV3's image encoder: stride-2 convolutions (kernel 4) of doubling depth, each with LayerNorm and SiLU,
    down to 4x4, then flattened. Maps ``(N, C, img_size, img_size)`` in ``[0, 1]`` to ``(N, 16 * depth * 2^(L-1))``
    with ``L = log2(img_size / 4)`` stages."""
    layers, c = [_Shift(-0.5)], in_channels
    for i in range(int(math.log2(img_size // 4))):
        layers += [nn.Conv2d(c, depth * 2**i, 4, 2, 1), ChannelNorm(depth * 2**i), nn.SiLU()]
        c = depth * 2**i
    return nn.Sequential(*layers, nn.Flatten())


def conv_decoder(feat_dim: int, out_channels: int = 3, depth: int = 32, img_size: int = 64) -> nn.Sequential:
    """DreamerV3's image decoder: a linear layer to ``4x4`` maps, then transposed convolutions that invert
    ``conv_encoder``. Maps features ``(N, feat_dim)`` to images ``(N, C, img_size, img_size)`` centred on 0.5."""
    stages = int(math.log2(img_size // 4))
    c = depth * 2 ** (stages - 1)
    layers = [nn.Linear(feat_dim, 16 * c), nn.Unflatten(1, (c, 4, 4))]
    for i in reversed(range(stages - 1)):
        layers += [nn.ConvTranspose2d(c, depth * 2**i, 4, 2, 1), ChannelNorm(depth * 2**i), nn.SiLU()]
        c = depth * 2**i
    return nn.Sequential(*layers, nn.ConvTranspose2d(c, out_channels, 4, 2, 1), _Shift(0.5))


class _Shift(nn.Module):
    def __init__(self, shift: float):
        super().__init__()
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        return (x.float() / 255 if x.dtype == torch.uint8 else x) + self.shift


class GRUCell(nn.Module):
    """DreamerV3's GRU: one linear layer over ``[x, h]`` with LayerNorm, and an update gate biased by ``-1``."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        reset, cand, update = self.norm(self.linear(torch.cat([x, h], -1))).chunk(3, -1)
        cand = (reset.sigmoid() * cand).tanh()
        update = (update - 1).sigmoid()
        return update * cand + (1 - update) * h


def _mlp(dims: list[int]) -> MLP:
    return MLP(dims, norm=nn.LayerNorm, act=nn.SiLU)


class DreamerV3(WorldModel):
    """DreamerV3: a recurrent state-space model (RSSM) learned by reconstruction, and an actor-critic learned in its
    imagination.

    The model state ``s_t = (h_t, z_t)`` joins a deterministic GRU state ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`` and a
    stochastic state of ``stoch`` categoricals with ``classes`` classes each, drawn with straight-through gradients
    from the posterior ``q(z_t | h_t, x_t)`` when the image is seen and from the prior ``p(z_t | h_t)`` when it is
    imagined; both mix 1% uniform into their probabilities. From ``s_t`` a CNN decoder reconstructs the image, a reward
    head regresses the reward with two-hot symlog bins and a continue head predicts whether the episode goes on. KL
    balancing with free bits trains the prior toward the posterior (``dyn``) and, more weakly, the reverse (``rep``):

        L = E[ ||x̂_t - x_t||² - ln p(r_t | s_t) - ln p(c_t | s_t) + β_dyn max(1, KL[sg(q) || p])
               + β_rep max(1, KL[q || sg(p)]) ]

    With rewards, an actor and a critic learn on trajectories imagined from every posterior state for ``horizon``
    steps. The critic regresses two-hot λ-returns ``R_t = r_{t+1} + γ c_{t+1} ((1 - λ) v_{t+1} + λ R_{t+1})`` and the
    outputs of its own EMA; the actor maximizes ``(R_t - v_t) / max(1, S)``, with ``S`` the EMA of the 5th to 95th
    percentile range of the returns, plus ``η`` times its entropy. Without rewards only the world model learns.

    The model state, flattened to ``(..., deter_dim + stoch * classes)``, is the latent of ``encode``, ``predict``
    and ``rollout``, so ``CEM`` can plan on it toward an encoded goal image; ``act`` gives the actor's actions.
    ``forward`` sums the world-model, critic and actor losses: their gradients reach disjoint parameters, as with the
    reference's three optimizers. Train with ``param_groups``, ``clip_grad_norm`` and ``update_target``.

    Hafner et al., "Mastering Diverse Domains through World Models", 2023.
    Paper: https://arxiv.org/abs/2301.04104
    Code: https://github.com/danijar/dreamerv3

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)``; ``conv_encoder`` by default.
        decoder: module mapping model states ``(N, feat_dim)`` to images; ``conv_decoder`` by default.
        cnn_depth: channels of the first convolution of the default encoder and decoder (the paper's multiplier).
        free_nats: free bits of both KL terms, summed over the categoricals.
        unimix: share of the uniform distribution mixed into every categorical.
        actor_grad: ``"dynamics"`` backpropagates returns through the imagined dynamics (the paper's first version
            for continuous actions), ``"reinforce"`` uses the score function (its 2024 revision).
        sample: draw the stochastic state in ``encode``, ``predict`` and ``rollout``; otherwise take its mode.
        cost_on: the part of the model state that ``cost`` compares with the goal: ``"stoch"``, ``"deter"`` or
            ``"full"`` (the default, which planned best on Two-Rooms).
    """

    history = 1

    def __init__(
        self,
        action_dim: int,
        in_channels: int = 3,
        img_size: int = 64,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        embed_dim: int | None = None,
        cnn_depth: int = 16,
        deter_dim: int = 256,
        stoch: int = 32,
        classes: int = 16,
        hidden_dim: int = 256,
        mlp_layers: int = 2,
        num_bins: int = 255,
        unimix: float = 0.01,
        free_nats: float = 1.0,
        dyn_coef: float = 0.5,
        rep_coef: float = 0.1,
        horizon: int = 15,
        discount: float = 1 - 1 / 333,
        lam: float = 0.95,
        entropy_coef: float = 3e-4,
        slow_reg: float = 1.0,
        retnorm_decay: float = 0.99,
        min_std: float = 0.1,
        max_std: float = 1.0,
        actor_grad: str = "dynamics",
        sample: bool = False,
        cost_on: str = "full",
    ):
        super().__init__()
        if cost_on not in ("stoch", "deter", "full"):
            raise ValueError(f"cost_on must be 'stoch', 'deter' or 'full', not {cost_on!r}")
        if actor_grad not in ("dynamics", "reinforce"):
            raise ValueError(f"actor_grad must be 'dynamics' or 'reinforce', not {actor_grad!r}")
        self.action_dim, self.deter_dim, self.stoch, self.classes = action_dim, deter_dim, stoch, classes
        self.unimix, self.free_nats, self.dyn_coef, self.rep_coef = unimix, free_nats, dyn_coef, rep_coef
        self.horizon, self.discount, self.lam, self.entropy_coef = horizon, discount, lam, entropy_coef
        self.slow_reg, self.retnorm_decay, self.min_std, self.max_std = slow_reg, retnorm_decay, min_std, max_std
        self.actor_grad, self.sample, self.cost_on = actor_grad, sample, cost_on
        self.feat_dim = feat = deter_dim + stoch * classes
        if encoder is None:
            encoder = conv_encoder(in_channels, cnn_depth, img_size)
            embed_dim = 16 * cnn_depth * 2 ** (int(math.log2(img_size // 4)) - 1)
        if embed_dim is None:
            raise ValueError("pass embed_dim with a custom encoder")
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else conv_decoder(feat, in_channels, cnn_depth, img_size)
        self.img_in = nn.Sequential(
            nn.Linear(stoch * classes + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.gru = GRUCell(hidden_dim, deter_dim)
        self.prior = _mlp([deter_dim, hidden_dim, stoch * classes])
        self.posterior = _mlp([deter_dim + embed_dim, hidden_dim, stoch * classes])
        self.init_deter = nn.Parameter(torch.zeros(deter_dim))
        heads = [hidden_dim] * mlp_layers
        self.reward = _mlp([feat, *heads, num_bins])
        self.cont = _mlp([feat, *heads, 1])
        self.actor = _mlp([feat, *heads, 2 * action_dim])
        self.critic = _mlp([feat, *heads, num_bins])
        for head in (self.reward, self.critic):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.register_buffer("bins", torch.linspace(-20.0, 20.0, num_bins))
        self.register_buffer("ret_low", torch.zeros(()))
        self.register_buffer("ret_high", torch.zeros(()))

    # ----- RSSM -----

    def _dist(self, logits: Tensor) -> Tensor:
        """Unimix log-probabilities ``(..., stoch, classes)`` from logits ``(..., stoch * classes)``."""
        probs = logits.unflatten(-1, (self.stoch, self.classes)).softmax(-1)
        return ((1 - self.unimix) * probs + self.unimix / self.classes).log()

    def _draw(self, logp: Tensor, sample: bool) -> Tensor:
        """One-hot states ``(..., stoch * classes)``: a straight-through sample, or the mode."""
        probs = logp.exp()
        idx = torch.multinomial(probs.flatten(0, -2), 1).view(probs.shape[:-1]) if sample else probs.argmax(-1)
        onehot = F.one_hot(idx, self.classes).to(probs)
        return (onehot + probs - probs.detach()).flatten(-2)

    def initial(self, batch_size: int) -> Tensor:
        """The learned initial model state ``(batch_size, feat_dim)``: ``tanh`` of a learned ``h``, and the mode of
        its prior."""
        h = self.init_deter.tanh().expand(batch_size, -1)
        return torch.cat([h, self._draw(self._dist(self.prior(h)), False)], -1)

    def img_step(self, s: Tensor, action: Tensor, sample: bool = True, frozen: bool = False) -> tuple[Tensor, Tensor]:
        """Prior step from model states ``(..., feat_dim)`` and actions ``(..., action_dim)``; returns the next state
        and the prior log-probabilities ``(..., stoch, classes)``. ``frozen`` keeps gradients out of the weights."""
        call = _frozen if frozen else _plain
        action = action / action.abs().clamp(min=1).detach()  # the reference's action clip at 1
        x = call(self.img_in, torch.cat([s[..., self.deter_dim :], action], -1))
        h = call(self.gru, x, s[..., : self.deter_dim])
        logp = self._dist(call(self.prior, h))
        return torch.cat([h, self._draw(logp, sample)], -1), logp

    def obs_step(self, s: Tensor, action: Tensor, embed: Tensor, sample: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        """Posterior step: the next state ``(..., feat_dim)`` given the next image's embedding, with the posterior and
        prior log-probabilities."""
        prior, prior_logp = self.img_step(s, action, sample)
        h = prior[..., : self.deter_dim]
        logp = self._dist(self.posterior(torch.cat([h, embed], -1)))
        return torch.cat([h, self._draw(logp, sample)], -1), logp, prior_logp

    def observe(
        self, obs: Tensor, actions: Tensor | None = None, state: Tensor | None = None, sample: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Filter images ``(B, T, C, H, W)`` with the ``T - 1`` actions between them ``(B, T - 1, action_dim)``
        (zeros when omitted). The first step starts from ``state`` ``(B, feat_dim)``, the initial state by default,
        with a zero action, as in the reference. Returns posterior states ``(B, T, feat_dim)`` and the posterior and
        prior log-probabilities ``(B, T, stoch, classes)``."""
        B, T = obs.shape[:2]
        embed = self.encoder(obs.flatten(0, 1)).unflatten(0, (B, T))
        zero = embed.new_zeros(B, 1, self.action_dim)
        actions = zero.expand(-1, T - 1, -1) if actions is None else actions
        actions = torch.cat([zero, actions[:, : T - 1].to(embed)], 1)
        s = self.initial(B) if state is None else state
        posts, post_logps, prior_logps = [], [], []
        for t in range(T):
            s, post_logp, prior_logp = self.obs_step(s, actions[:, t], embed[:, t], sample)
            posts.append(s)
            post_logps.append(post_logp)
            prior_logps.append(prior_logp)
        return torch.stack(posts, 1), torch.stack(post_logps, 1), torch.stack(prior_logps, 1)

    def encode(self, obs: Tensor, actions: Tensor | None = None) -> Tensor:
        """Images ``(B, T, C, H, W)`` to filtered model states ``(B, T, feat_dim)``; see ``observe``."""
        return self.observe(obs, actions, sample=self.sample)[0]

    def step(self, state: Tensor, action: Tensor, obs: Tensor) -> Tensor:
        """Track the model state online: from ``state`` ``(B, feat_dim)``, ``action`` and the next image
        ``(B, C, H, W)`` to the next posterior state."""
        return self.obs_step(state, action, self.encoder(obs), self.sample)[0]

    def predict(self, z: Tensor, actions: Tensor, sample: bool | None = None) -> Tensor:
        """Prior next states from model states ``(..., feat_dim)`` and actions ``(..., action_dim)``."""
        return self.img_step(z, actions, self.sample if sample is None else sample)[0]

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        """Squared distance between the final model state of each rollout ``(B, T, feat_dim)`` and the goal state
        ``(B, feat_dim)``, on the part named by ``cost_on``: the stochastic state, the deterministic one or both."""
        part = {"stoch": slice(self.deter_dim, None), "deter": slice(0, self.deter_dim), "full": slice(None)}
        part = part[self.cost_on]
        return (z[:, -1, part] - goal[:, part]).square().mean(-1)

    def decode(self, z: Tensor) -> Tensor:
        """Model states ``(..., feat_dim)`` to images ``(..., C, H, W)``."""
        return self.decoder(z.flatten(0, -2)).unflatten(0, z.shape[:-1])

    # ----- actor-critic -----

    def policy(self, s: Tensor, frozen: bool = False) -> torch.distributions.Normal:
        """The actor's Gaussian over actions at model states ``(..., feat_dim)``: mean ``tanh(μ)`` and std in
        ``[min_std, max_std]``, as in the reference's continuous actor."""
        mean, std = (_frozen if frozen else _plain)(self.actor, s).chunk(2, -1)
        std = (self.max_std - self.min_std) * (std + 2).sigmoid() + self.min_std
        return torch.distributions.Normal(mean.tanh(), std)

    def act(self, s: Tensor, sample: bool = False) -> Tensor:
        """Actions ``(..., action_dim)`` in ``[-1, 1]`` at model states: the actor's mean, or a clipped sample."""
        dist = self.policy(s)
        return (dist.sample() if sample else dist.mean).clamp(-1, 1)

    def value(self, s: Tensor, slow: bool = False, frozen: bool = False) -> Tensor:
        """The critic's (or its EMA's) value ``(...)`` of model states ``(..., feat_dim)``."""
        net = self.slow_critic if slow else self.critic
        return two_hot_inv((_frozen if frozen else _plain)(net, s), self.bins)

    def imagine(self, s: Tensor, horizon: int | None = None, frozen: bool = False) -> tuple[Tensor, Tensor]:
        """Roll the actor out in imagination from states ``(N, feat_dim)``; returns states ``(N, H + 1, feat_dim)``
        and the sampled actions ``(N, H + 1, action_dim)`` (the last one is never executed)."""
        states, actions = [s], []
        for _ in range(horizon or self.horizon):
            actions.append(self.policy(states[-1].detach() if self.actor_grad == "reinforce" else states[-1]).rsample())
            states.append(self.img_step(states[-1], actions[-1], sample=True, frozen=frozen)[0])
        actions.append(self.policy(states[-1].detach()).rsample())
        return torch.stack(states, 1), torch.stack(actions, 1)

    def lambda_returns(self, rewards: Tensor, conts: Tensor, values: Tensor) -> Tensor:
        """λ-returns ``(N, H)`` from rewards and continue flags ``(N, H)`` of states ``1..H`` and values
        ``(N, H + 1)`` of states ``0..H``: ``R_t = r_{t+1} + γ c_{t+1} ((1 - λ) v_{t+1} + λ R_{t+1})``, ``R_H = v_H``."""
        out, ret = [], values[:, -1]
        for t in reversed(range(rewards.size(1))):
            ret = rewards[:, t] + self.discount * conts[:, t] * ((1 - self.lam) * values[:, t + 1] + self.lam * ret)
            out.append(ret)
        return torch.stack(out[::-1], 1)

    def actor_critic_loss(self, start: Tensor, cont: Tensor | None = None) -> dict[str, Tensor]:
        """Critic and actor losses on imagined trajectories from start states ``(N, feat_dim)`` (detached), whose
        continue flags ``cont`` ``(N,)`` default to 1. Neither loss updates the world model."""
        start = start.detach()
        states, actions = self.imagine(start, frozen=True)
        conts = (_frozen(self.cont, states[:, 1:]).squeeze(-1) > 0).to(start)  # the mode, as in the reference
        rewards = two_hot_inv(_frozen(self.reward, states[:, 1:]), self.bins)
        values = self.value(states, frozen=True)
        rets = self.lambda_returns(rewards, conts, values)
        first = torch.ones_like(conts[:, 0]) if cont is None else cont.to(conts)
        weight = torch.cumprod(torch.cat([first[:, None], self.discount * conts[:, :-1]], 1), 1).detach()

        if self.training:
            low, high = torch.quantile(rets.detach().flatten().float(), rets.new_tensor([0.05, 0.95]))
            self.ret_low.lerp_(low, 1 - self.retnorm_decay)
            self.ret_high.lerp_(high, 1 - self.retnorm_decay)
        scale = (self.ret_high - self.ret_low).clamp(min=1.0)
        adv = (rets - values[:, :-1].detach()) / scale
        policy = self.policy(states[:, :-1].detach())
        entropy = policy.entropy().sum(-1)
        if self.actor_grad == "dynamics":
            actor = -adv
        else:
            actor = -policy.log_prob(actions[:, :-1].detach()).sum(-1) * adv.detach()
        actor = (weight * (actor - self.entropy_coef * entropy)).mean()

        traj = states[:, :-1].detach()
        logits = self.critic(traj).log_softmax(-1)
        with torch.no_grad():
            slow = self.value(traj, slow=True)
        critic = -(two_hot(rets.detach(), self.bins) * logits).sum(-1)
        critic = critic - self.slow_reg * (two_hot(slow, self.bins) * logits).sum(-1)
        critic = (weight * critic).mean()
        return {"critic_loss": critic, "actor_loss": actor, "entropy": entropy.mean().detach()}

    # ----- training -----

    def forward(
        self, obs: Tensor, actions: Tensor, rewards: Tensor | None = None, terminated: Tensor | None = None
    ) -> dict[str, Tensor]:
        """``obs`` ``(B, T + 1, C, H, W)`` in ``[0, 1]``, ``actions`` ``(B, T, action_dim)`` and optionally ``rewards``
        and ``terminated`` ``(B, T)``: ``actions[:, t]`` is applied at ``obs[:, t]``, earns ``rewards[:, t]`` and leads
        to ``obs[:, t + 1]``, which ``terminated[:, t]`` marks as terminal. Without rewards only the world model learns.
        """
        posts, post_logp, prior_logp = self.observe(obs, actions, sample=True)
        target = obs.float() / 255 if obs.dtype == torch.uint8 else obs
        image = (self.decode(posts) - target).square().flatten(2).sum(-1).mean()
        dyn = _kl(post_logp.detach(), prior_logp)
        rep = _kl(post_logp, prior_logp.detach())
        out = {
            "image_loss": image,
            "dyn_loss": dyn.clamp(min=self.free_nats).mean(),
            "rep_loss": rep.clamp(min=self.free_nats).mean(),
            "kl": dyn.mean().detach(),
        }
        loss = image + self.dyn_coef * out["dyn_loss"] + self.rep_coef * out["rep_loss"]
        if rewards is None:
            return {"loss": loss, **out}

        terminated = torch.zeros_like(rewards) if terminated is None else terminated.to(rewards.dtype)
        out["reward_loss"] = -(two_hot(rewards, self.bins) * self.reward(posts[:, 1:]).log_softmax(-1)).sum(-1).mean()
        out["continue_loss"] = F.binary_cross_entropy_with_logits(self.cont(posts[:, 1:]).squeeze(-1), 1 - terminated)
        loss = loss + out["reward_loss"] + out["continue_loss"]
        cont = torch.cat([torch.ones_like(terminated[:, :1]), 1 - terminated], 1)
        out.update(self.actor_critic_loss(posts.flatten(0, 1), cont.flatten()))
        return {"loss": loss + out["critic_loss"] + out["actor_loss"], **out}

    def param_groups(self, lr: float = 1e-4, actor_lr: float = 3e-5, critic_lr: float = 3e-5) -> list[dict]:
        """Adam groups with the paper's rates: the world model at ``1e-4`` (``eps=1e-8``), the actor and the critic at
        ``3e-5`` (``eps=1e-5``)."""
        actor, critic = list(self.actor.parameters()), list(self.critic.parameters())
        skip = {id(p) for p in actor + critic}
        world = [p for p in self.parameters() if p.requires_grad and id(p) not in skip]
        return [
            {"params": world, "lr": lr, "eps": 1e-8},
            {"params": actor, "lr": actor_lr, "eps": 1e-5},
            {"params": critic, "lr": critic_lr, "eps": 1e-5},
        ]

    def clip_grad_norm(self, world: float = 1000.0, actor: float = 100.0, critic: float = 100.0) -> list[Tensor]:
        """Clip the world model's, the actor's and the critic's gradients separately (the paper's norms); returns
        the three norms."""
        return [nn.utils.clip_grad_norm_(g["params"], m) for g, m in zip(self.param_groups(), (world, actor, critic))]

    def update_target(self, momentum: float = 0.98):
        """EMA update of the slow critic that regularizes the critic."""
        ema_update(self.slow_critic, self.critic, momentum)


def _kl(p_logp: Tensor, q_logp: Tensor) -> Tensor:
    """KL between factorized categoricals ``(..., stoch, classes)``, summed over the categoricals: ``(...)``."""
    return (p_logp.exp() * (p_logp - q_logp)).sum((-2, -1))


def _plain(module: nn.Module, *args):
    return module(*args)


def _frozen(module: nn.Module, *args):
    """``module(*args)`` with gradients reaching the inputs but not the weights."""
    return functional_call(module, {k: v.detach() for k, v in module.named_parameters()}, args)
