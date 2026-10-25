# AdaJEPA

> Wang, Bounou, LeCun and Ren, *AdaJEPA: An Adaptive Latent World Model*, 2026. [Paper](https://arxiv.org/abs/2606.32026) · [Code](https://github.com/agentic-learning-ai-lab/adajepa) · [Notebook](../../examples/action/adajepa.ipynb)

Latent world models are usually frozen once trained. When the test environment shifts, through a new look, new dynamics or a new layout, the predictions go wrong and so do the plans. AdaJEPA keeps training the world model inside the planning loop. It plans, runs the first action, adds the transition it observed, $(o_t, a_t, o_{t+1})$, to a small buffer of recent transitions, and takes one gradient step on the same latent prediction loss before planning again. The target is the encoded next observation under a stop-gradient, so no labels, rewards or demonstrations are needed. Only a few layers change: the last predictor block with its final norm, and the encoder's head. Each episode starts again from the trained weights.

$$
\mathcal{L}_{\rm ada}(\mathcal{B}) = \frac{1}{|\mathcal{B}|} \sum_{(o_i, a_i, o_{i+1}) \in \mathcal{B}}
\big\lVert f_\theta\big(E_\phi(o_i), E_\psi(a_i)\big) - \operatorname{sg}\big(E_\phi(o_{i+1})\big) \big\rVert^2,
\qquad \Omega \leftarrow \Omega - \eta\, \nabla_\Omega \mathcal{L}_{\rm ada}
$$

The world model is trained offline on the same loss, with the temporal straightening term of the paper's base model (Wang et al., *Temporal Straightening for Latent Planning*, 2026), $1 - \cos(z_{t+1} - z_t,\, z_{t+2} - z_{t+1})$, and VCReg variance and covariance terms:

$$
\mathcal{L} = \mathcal{L}_{\rm pred} + \lambda_c\,\mathcal{L}_{\rm curv} + \lambda_s\,\mathcal{L}_{\rm std} + \lambda_v\,\mathcal{L}_{\rm cov}
$$

## Usage

```python
from torch_world import evaluation
from torch_world.models import AdaJEPA, PLDMEncoder
from torch_world.planning import CEM

model = AdaJEPA(PLDMEncoder(img_size=64, latent_dim=128), action_dim=2, embed_dim=128, history=3)

out = model(pixels, actions)              # (B, history + 1, C, H, W), (B, history + 1, A)
# out: {"loss", "pred_loss", "curvature_loss", "std_loss", "cov_loss"}

# test-time adaptation on observed transitions: (1, T + 1, C, H, W) frames, (1, T, A) actions
losses = model.adapt(frames, actions, steps=1, lr=5e-4, encoder_lr=1e-5)

# closed loop: adapt on the last 5 transitions of each episode, then plan
planner = CEM(horizon=10, action_dim=2, num_samples=128, num_elites=16, iterations=5, bounds=(-1, 1), objective="path")
policy = model.policy(planner, goal_images, buffer=5, steps=1, lr=5e-4, encoder_lr=1e-5)
run = evaluation.goal_reaching(env, {"AdaJEPA": policy}, start, goal, steps=60)
```

`adapted_parameters()` returns the two groups updated at test time: the last predictor block with the final norm, and the encoder's last child module with parameters (the final linear layer of `PLDMEncoder`). `adapt` builds a fresh Adam optimizer on every call, as the reference code does, with separate learning rates for the predictor and the encoder head (`encoder_lr=0` freezes the encoder); the paper's defaults are the training learning rates, $5\times10^{-4}$ and $10^{-5}$, one step per replan and a buffer of the 5 most recent transitions. `policy` returns an `AdaptivePolicy`: every episode of the batch keeps its own copy of the adapted parameters and its own buffer, the goal image is re-encoded with the adapted encoder at every plan, and the model's weights are restored after each call. Episodes are planned one at a time, so it is slower than `evaluation.goal_policy`; `steps=0` runs the same loop frozen.

## Differences from the paper

- **Base model.** The paper's main models are a ResNet encoder with a ViT predictor from Temporal Straightening (also DINO-WM). Here the encoder is any image encoder (the notebook uses the small IMPALA CNN of [PLDM](pldm.md)) and the predictor is a small causal transformer over the last `history` latents, each concatenated with its action embedding (the paper's concatenation; the action features of the output are dropped). One frame is one latent, with no proprioception.
- **Anti-collapse.** The reference trains with a stop-gradient target, the curvature term and an encoder learning rate 50 times below the predictor's. `fit` uses one learning rate, so VCReg variance and covariance terms are on by default (`std_coeff=1`, `cov_coeff=0.04`; the reference has them as an option), and latents go through a LayerNorm without parameters, as in PLDM. Without it the latent scale diverged in our runs. `curvature_coeff=0.1` is a choice: the paper does not give the weights of its checkpoints.
- **Planner.** The paper plans by gradient descent (and CEM) and executes an action chunk of a frameskip-5 model. Here the model sees every frame, CEM plans and the first action runs, so the model adapts after every environment step.
- **Mode.** The reference switches the adapted modules to train mode while adapting. Here the model stays in eval mode, so dropout and BatchNorm statistics do not change at test time.

## Reference

Arguments and methods: [`AdaJEPA`](../../api/models.md#torch_world.models.AdaJEPA) in the API reference.
