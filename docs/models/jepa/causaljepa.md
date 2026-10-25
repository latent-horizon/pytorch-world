# Causal-JEPA

> Nam, Le Lidec, Maes, LeCun and Balestriero, *Causal-JEPA: Learning World Models through Object-Level Latent Masking*, 2026. [Paper](https://arxiv.org/abs/2602.11389) · [Code](https://github.com/galilai-group/cjepa) · [Notebook](../../examples/action/causaljepa.ipynb)

Causal-JEPA (C-JEPA) is a world model over objects. A frozen object-centric encoder turns every frame into $N$ slots $S_t = \{s^1_t, \dots, s^N_t\}$. Each timestep becomes $N$ object tokens plus an action token (and a proprioception token when there is one). A bidirectional transformer reads the last $T_h$ steps and predicts the next ones. There is no positional encoding along the object axis, so the predictor treats objects as a set.

In training, the history of a few random objects is hidden after the first step. Their tokens become queries $\tilde z^i_\tau = \phi(z^i_{t_0}) + m + e_\tau$: a projection of the object's first-step slot (its identity anchor), a learned mask token and the time embedding. The same queries stand for every token of the future. To recover a hidden object, the predictor has to use the other objects and the actions. The paper calls this a latent intervention: it makes learning the interactions necessary, not optional.

$$
\mathcal{L} = \sum_{\tau=t_0+1}^{t} \sum_{i \in \mathcal{M}} \big\lVert \hat z^i_\tau - z^i_\tau \big\rVert^2
+ \sum_{\tau=t+1}^{t+T_p} \sum_{i=1}^{N} \big\lVert \hat z^i_\tau - z^i_\tau \big\rVert^2
$$

At test time the history is fully visible, and only the future is predicted. Planning runs CEM on the distance between the final predicted slots and the goal's slots, after matching the slots one to one (the Hungarian matching of the reference).

## Usage

```python
from torch_world.models import CausalJEPA
from torch_world.models import SlotEncoder
from torch_world.utils import fit

slots = SlotEncoder(img_size=64, num_slots=5, slot_dim=64)   # track="match": a slot follows its object
fit(slots, frames, epochs=10, lr=4e-4, clip=1.0, inputs=lambda b: (b[0], True))   # pixel reconstruction

model = CausalJEPA(slots, action_dim=10, slot_dim=64, history=3, max_masked=2,
                   depth=6, num_heads=8, head_dim=32, mlp_dim=512)          # freezes the slot encoder

out = model(obs, actions)                 # pixels (B, history + 1, C, H, W) or slots (B, history + 1, S, D)
# out: {"loss", "history_loss", "future_loss"}  (+ "proprio_loss" with proprio_dim)

z = model.encode(pixels)                  # (B, T, S, D)
future = model.rollout(z[:, :1], plan)    # (B, T + 1, S, D)
filled = model.predict(z, actions, masked=hidden)  # objects hidden after the first step, inferred: (B, T + 1, S, D)
```

The encoder is frozen, so its slots can be extracted once and passed to `forward` in place of the pixels, as the paper trains. `max_masked` sets how many objects are hidden: each sample hides a number drawn uniformly from $0$ to `max_masked`, as in the paper (0 to 2 on Push-T, 0 to 4 on CLEVRER). `max_masked=0` gives the paper's OC-JEPA ablation, the same model with no object masking. `predict(z, actions, masked=...)` hides chosen objects after the first step and returns what the model infers for them. Slot matching in `cost` tries every permutation of the slots, which is exact and fast up to 8 slots. Pass `match=False` to compare slots in order.

The defaults are the paper's: 6 layers, 16 heads of 64, MLP width 2048, slot dimension 128, history 3, one predicted step, 0 to 2 hidden objects. The paper trains with Adam at $5 \times 10^{-4}$ for 30 epochs, at batch size 256.

### Differences from the paper

- **Object encoder.** The paper uses VideoSAUR on frozen DINOv2 features, trained separately. The library has no VideoSAUR, so it provides [`SlotEncoder`](#reference), a small slot-attention autoencoder (Locatello et al., 2020) with a spatial-broadcast decoder. It learns its initial slots as queries, with the BO-QSA straight-through gradient. The learned queries split a frame by region, not by object, so a slot's index would change as objects move. To keep each object in one slot through a video, `track="match"` (the default) reorders the slots of every frame to best match the previous frame's, the matching the reference applies to predicted frames. `track="propagate"` instead starts each frame from the previous frame's slots, as VideoSAUR and SAVi do. In a tuning run on the notebook's scene, matching kept slot identity better than propagating (94% against 90% of objects in the same slot through an episode). Propagated slots also drifted away from those of a single frame, which is what the planner encodes. Any frozen module that maps videos to slots `(B, T, S, D)` can stand in for it.
- **Which objects are hidden.** The reference draws the hidden slots from a generator reseeded on every call, so it hides the same slot indices at every step. Here each sample draws its own hidden objects, following the paper's text ("randomly masks between zero and two slots").
- **Losses.** The two terms are averaged over their tokens and summed, as in the reference code. The paper writes them as sums. Masked slots count only after the first step, where they are actually hidden; the reference also counts their visible first step.
- **Action and proprioception tokens.** The paper embeds them with a 1D convolution with kernel size 1; here that is a linear layer. The reference appends them as extra nodes per step, and so does this implementation.
- **Transformer.** Standard pre-norm blocks. The reference applies attention without a pre-norm.
- **Rollout.** Each step predicts one frame from the last `history` frames, with the real action of every step. With `num_preds > 1` the reference appends all predicted frames at once, though their action tokens were hidden. The reference also re-matches the slots of each predicted frame to the previous one. Here the predictor keeps slot order by construction (every output token is anchored to its input slot), so only the goal cost matches slots.

## Reference

Arguments and methods: [`CausalJEPA`](../../api/models.md#torch_world.models.CausalJEPA) in the API reference.
