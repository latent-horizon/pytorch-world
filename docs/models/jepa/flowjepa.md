# Flow-JEPA

> Huo, Luo and Song, *Flow-JEPA: Robust Latent Dynamics for JEPA World Models via Flow Matching*, 2026. [Paper](https://arxiv.org/abs/2608.29029) · [Code](https://github.com/HuoYanchen/Flow-JEPA) · [Notebook](../../examples/action/flowjepa.ipynb)

Flow-JEPA keeps LeWM's encoder and SIGReg, and replaces its one-step predictor with a conditional flow-matching model of the whole future. From the current embedding $z_t$ and $h$ actions, a transformer predicts the velocity that transports Gaussian source latents $Z_0$ to the future embeddings $Z = [z_{t+1}, \dots, z_{t+h}]$, all at once. The flow time conditions every block through adaLN-zero, and future step $i$ sees only the actions up to $a_{t+i-1}$. Each draw of $Z_0$ gives a different imagined future.

$$
Z_\tau = (1 - \tau) Z_0 + \tau Z, \qquad
\mathcal{L} = \big\lVert v_\theta(Z_\tau, \tau, z_t, a_{t:t+h}) - (Z - Z_0) \big\rVert^2 + \lambda\,\operatorname{SIGReg}(z)
$$

## Usage

```python
from torch_world.models import FlowJEPA
from torch_world.nn import CLSEncoder

model = FlowJEPA(CLSEncoder(img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4), action_dim=10,
                 embed_dim=128, num_preds=5, depth=4, num_heads=4, head_dim=32, mlp_dim=512, proj_dim=512)

out = model(pixels, actions)              # (B, num_preds + 1, C, H, W), (B, num_preds + 1, A)
# out: {"loss", "flow_loss", "sigreg_loss"}

z = model.encode(pixels[:, :1])           # (B, 1, embed_dim)
future = model.rollout(z, plan)           # one sampled future, (B, T + 1, embed_dim)
futures = model.sample(z, plan, 16)       # (B, 16, T + 1, embed_dim)
```

`Z_0` is standard noise by default; `source="current", source_std=0.5` centers it on $z_t$, the paper's other setting. Sampling integrates the flow in `steps` Euler steps (8 in the paper). `predict` and `rollout` draw a fresh sample on every call (pass `generator` or `noise` to fix it), so [`CEM`](../../api/planning.md#torch_world.planning.CEM) scores each candidate on one sampled future, as the reference does. The predictor generates the whole trajectory from the last context embedding (`history = 1`); plan over `num_preds` steps, the horizon it was trained on. The paper uses a frameskip of 5 with `num_preds = 5`, and `lamb = 0.09` on Push-T, `0.1` elsewhere.

Self-attention among future tokens is bidirectional, as in the paper's Two-Room and Cube runs; `causal=True` gives the Reacher and Push-T setting. The reference cross-attends from future tokens to the context and actions. Here they share one masked self-attention, with the same visibility. The projector uses BatchNorm: call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning.

## Reference

Arguments and methods: [`FlowJEPA`](../../api/models.md#torch_world.models.FlowJEPA) in the API reference.
