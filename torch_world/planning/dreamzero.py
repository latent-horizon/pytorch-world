"""Closed-loop DreamZero control on real observations."""

from collections.abc import Callable

import torch
from torch import Tensor


class DreamZeroPolicy:
    """Run a ``DreamZero`` chunk by chunk, writing real observations back into its context.

    Every ``chunk_actions // raw_frames_per_chunk`` environment steps the image observation ``(B, 3, H, W)`` is
    recorded. When the action queue is empty, the recorded frames of the last ``num_chunks - 1`` chunks plus the frame
    before them are encoded with the model's autoencoder, and the next chunk is sampled from those latents and the
    actions executed over them. The model's predicted video is discarded, as DreamZero replaces predicted frames in its
    KV cache with the real ones: the model rebuilds its context key-value cache from the real latents for every chunk.

    ``plan(model, z, goal, past_actions, generator)`` chooses the chunk; by default it is the learned policy,
    ``model.sample_actions``. A ``WAMPlanner`` can rank and refine proposals toward ``goal`` instead. Each call returns
    one environment action ``(B, A)``. Call ``reset`` between episodes.
    """

    def __init__(
        self,
        model,
        goal: Tensor | None = None,
        plan: Callable | None = None,
        generator: torch.Generator | None = None,
    ):
        if model.state_dim:
            raise ValueError("DreamZeroPolicy runs image-only models; build states into a custom loop")
        if model.chunk_size % model.raw_frames_per_chunk:
            raise ValueError("chunk_actions must be a multiple of the raw frames each chunk covers")
        self.model, self.goal, self.plan, self.generator = model, goal, plan, generator
        self.stride = model.chunk_size // model.raw_frames_per_chunk
        self.reset()

    def reset(self):
        self.frames: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.queue: list[Tensor] = []
        self.steps = 0

    @torch.no_grad()
    def __call__(self, observation: Tensor) -> Tensor:
        model = self.model
        if self.steps % self.stride == 0:
            self.frames.append(observation)
        if not self.queue:
            context = model.num_chunks - 1
            frames = self.frames[-(1 + context * model.raw_frames_per_chunk) :]
            chunks = (len(frames) - 1) // model.raw_frames_per_chunk
            z = model.encode(torch.stack(frames, dim=1))
            past = self.actions[len(self.actions) - chunks * model.chunk_size :] if chunks else []
            past = torch.stack(past, dim=1) if past else z.new_zeros(z.size(0), 0, model.action_dim)
            if self.plan is None:
                chunk = model.sample_actions(z, self.goal, past_actions=past, generator=self.generator)
            else:
                chunk = self.plan(model, z, self.goal, past, self.generator)
            self.queue = list(chunk.unbind(1))
        action = self.queue.pop(0)
        self.actions.append(action)
        self.steps += 1
        return action
