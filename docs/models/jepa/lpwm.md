# LpWM

> Kuang et al., *LpWM: A Case for Sparse Representations in World Models*, 2026. [Paper](https://arxiv.org/abs/2608.22764) · [Code](https://github.com/YilunKuang/lpworldmodel) · [Notebook](../../examples/action/lpwm.ipynb)

LpWM is a LeWM-style world model whose latents are non-negative and sparse. Each frame is encoded to one embedding (the encoder's CLS token), projected and passed through a RepReLU link $h$ (ReLU forward, GELU gradient backward), so most units are exactly zero. A causal transformer, conditioned on action embeddings through adaLN-zero, predicts the next code from the last `history` frames, through the same link. In place of SIGReg, RDMReg matches the codes of each timestep to a rectified Laplace with a sliced 2-Wasserstein distance, which prevents collapse and sets the share of active units:

$$
\mathcal{L} = \big\lVert \hat z_{t+1} - z_{t+1} \big\rVert^2 + \lambda\,\mathbb{E}_{c \sim \mathcal{U}(\mathbb{S}^{d-1})}\, W_2^2\big(c^\top z,\ c^\top y\big), \qquad y \sim \textstyle\prod_i \operatorname{ReLU}\big(\mathcal{GN}_p(\mu, \sigma)\big)
$$

With `p = 1` and `mu = 0` half of the target's units are zero. The paper finds that which units fire encodes the discrete regime of the dynamics, and their magnitudes the continuous state within it. Planning is LeWM's: CEM toward the code of a goal image.

## Usage

```python
from torch_world.models import LpWM
from torch_world.nn import CLSEncoder

model = LpWM(CLSEncoder(img_size=64, patch_size=8, embed_dim=192), action_dim=2, history=1, depth=6, num_heads=8,
             head_dim=32, mlp_dim=768, proj_dim=768, lamb=10.0)

out = model(pixels, actions)        # (B, history + 1, C, H, W), (B, history + 1, A)
# out: {"loss", "pred_loss", "rdmreg_loss"}

z = model.encode(pixels)            # (B, T, embed_dim), non-negative with exact zeros
active = z > 0                      # which units fire
```

`encoder` maps images `(N, C, H, W)` to embeddings `(N, embed_dim)`; a ViT-tiny CLS token in the paper, which [`CLSEncoder`](../../api/nn.md#torch_world.nn.CLSEncoder) provides. `RDMReg(num_slices, p, mu, rectify)` sets the target: `p = 1` is the paper's sparse default, `mu < 0` makes codes sparser, and `p = 2, rectify=False` recovers LeWM's isotropic Gaussian. The paper uses `history = 1` on its Wall environment and `3` elsewhere, and `lamb = 10` with 1024 slices.

The projectors use BatchNorm: call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning.

## Reference

Arguments and methods: [`LpWM`](../../api/models.md#torch_world.models.LpWM) in the API reference.
