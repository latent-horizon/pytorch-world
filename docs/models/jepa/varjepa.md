# Variational JEPA

> Huang, *VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models*, 2026. [Paper](https://arxiv.org/abs/2601.14354) · [Code](https://github.com/YongchaoHuang/VJEPA) · [Notebook](../../examples/action/varjepa.ipynb)

Variational JEPA makes the JEPA predictor probabilistic: it outputs a Gaussian $p_\phi(z_{t+1} \mid z_{\le t}, a_{\le t}) = \mathcal{N}(\mu, \operatorname{diag}\sigma^2)$ over the next embedding instead of a point. Targets come from $q(z \mid x)$, a diagonal Gaussian whose mean is the embedding of an EMA target encoder and whose log-variance is a learned head on it. The loss is the negative log-likelihood of a reparameterized target sample, plus a KL toward a standard normal prior. Here the encoder and the action-conditioned causal transformer follow [LeWM](lewm.md), and SIGReg on the online embeddings guards against collapse.

$$
\mathcal{L} = \mathbb{E}_{z \sim q(\cdot \mid x_{t+1})}\Big[\tfrac12 \log \sigma^2 + \tfrac{(z - \mu)^2}{2\sigma^2}\Big]
+ \beta\,\operatorname{KL}\big(q(z \mid x_{t+1}) \,\Vert\, \mathcal{N}(0, I)\big) + \lambda\,\operatorname{SIGReg}(z)
$$

## Usage

```python
from torch_world.models import VariationalJEPA
from torch_world.nn import CLSEncoder

model = VariationalJEPA(CLSEncoder(img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4), action_dim=10,
                        embed_dim=128, history=3, depth=4, num_heads=4, head_dim=32, mlp_dim=512, proj_dim=512)

out = model(pixels, actions)              # (B, history + 1, C, H, W), (B, history + 1, A)
# out: {"loss", "nll_loss", "kl_loss", "sigreg_loss", "variance"}
model.update_target(0.99)                 # after each optimizer step

z = model.encode(pixels)                  # (B, T, embed_dim)
mean, logvar = model.distribution(z, actions)
futures = model.sample(z[:, :1], plan, 16)  # (B, 16, T + 1, embed_dim)
```

`predict` returns the predictive mean, so `rollout` and [`CEM`](../../api/planning.md#torch_world.planning.CEM) plan on it by default. The paper shows this MAP plan is optimal for quadratic costs when the noise does not depend on the action. `CEM(...)(model, z, goal, sample=True)` plans on sampled rollouts instead, one per candidate, as in the paper's sampling-based latent MPC. Averaging the cost over several samples, or penalizing its variance, are the risk-sensitive variants the paper mentions.

The paper states the objective for general encoders. Its pixel-control experiments add VICReg against collapse; this implementation uses SIGReg for the same role. Log-variances are clamped to `logvar_range`, as in the reference. The projector and head use BatchNorm: call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning.

## Reference

Arguments and methods: [`VariationalJEPA`](../../api/models.md#torch_world.models.VariationalJEPA) in the API reference.
