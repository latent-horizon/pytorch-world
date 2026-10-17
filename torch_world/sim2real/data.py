"""Episode-bounded simulation datasets with explicit transition alignment."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from torch_world.sim2real.config import PIECES
from torch_world.sim2real.simulation import SO101Chess


@dataclass
class Episode:
    """T actions connect T+1 observations; no padded action is a real transition."""

    pixels: torch.Tensor
    proprio: torch.Tensor
    actions: torch.Tensor
    requested_actions: torch.Tensor
    metadata: dict
    action_valid: torch.Tensor | None = None


class EpisodeWindows(Dataset):
    """Delay-aligned transitions, including when frozen visual features are cached."""

    def __init__(self, episodes, history, features=None):
        if history < 1:
            raise ValueError("history must be positive")
        self.episodes, self.history = episodes, history
        self.features = features
        self.offsets = []
        self.index = []
        for e, episode in enumerate(episodes):
            delay = episode.metadata.get("randomization", {}).get("observation_delay", 0)
            if not isinstance(delay, int) or delay < 0:
                raise ValueError("Observation delay must be a nonnegative integer")
            if len(episode.pixels) != len(episode.actions) + 1 or len(episode.proprio) != len(episode.pixels):
                raise ValueError("Episode must contain T actions and T+1 observations")
            self.offsets.append(delay)
            self.index.extend((e, t) for t in range(len(episode.actions) - delay - history + 1))
        if not self.index:
            raise ValueError("No episode has enough transitions for this model history")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        e, t = self.index[index]
        episode, h = self.episodes[e], self.history
        start = t + self.offsets[e]
        # Existing WorldModel losses expect T frames and T actions, ignoring
        # the last action. Pad only here; the stored episode has no fake step.
        actions = torch.cat((episode.actions[t : t + h], torch.zeros_like(episode.actions[:1])))
        visual = (
            {"features": self.features[e][start : start + h + 1]}
            if self.features is not None
            else {
                "pixels": episode.pixels[start : start + h + 1].float()
                / (255 if episode.pixels.dtype == torch.uint8 else 1)
            }
        )
        return {
            **visual,
            "states": episode.proprio[start : start + h + 1],
            "actions": actions,
        }


def collect_episodes(
    config,
    task,
    randomization,
    *,
    count=24,
    seed=7,
    steps=None,
    noise=0.025,
    calibration=None,
    progress=None,
    include_commanded=False,
):
    """Balanced piece/color schedule, with separate simulator instances per episode.

    Privileged expert actions supply simulation-only play data. Failed grasps
    are retained and labeled; collecting an episode never implies task success.
    """
    if count < 1 or noise < 0 or (steps is not None and steps < 1):
        raise ValueError("Invalid collection count, noise or length")
    episodes = []
    for index in range(count):
        piece, color = PIECES[index % 6], ("white", "black")[(index // 6) % 2]
        with SO101Chess(
            config, task, piece=piece, color=color, randomization=randomization, calibration=calibration
        ) as env:
            obs = env.reset(seed=seed + index)
            rng = np.random.default_rng(seed + index)
            pixels, states, actions, requested, timestamps = (
                [(obs.rgb * 255).round().to(torch.uint8)],
                [obs.state(include_commanded)],
                [],
                [],
                [obs.timestamp],
            )
            failure = None
            for _ in range(min(steps or task.max_steps, task.max_steps)):
                action = (env.expert() + torch.tensor(rng.normal(0, noise, 4), dtype=torch.float32)).clamp(-1, 1)
                try:
                    obs = env.step(action)
                except ValueError as error:
                    failure = str(error)
                    break
                pixels.append((obs.rgb * 255).round().to(torch.uint8))
                states.append(obs.state(include_commanded))
                timestamps.append(obs.timestamp)
                actions.append(torch.tensor(env.last_command.applied, dtype=torch.float32))
                requested.append(action)
                if env.done:
                    break
            if not actions:
                raise RuntimeError(f"No valid transitions collected for {piece}: {failure}")
            episodes.append(
                Episode(
                    torch.stack(pixels),
                    torch.stack(states),
                    torch.stack(actions),
                    torch.stack(requested),
                    {
                        "seed": seed + index,
                        "piece": piece,
                        "color": color,
                        "source": env.source,
                        "target": env.target_square,
                        "randomization": env.episode_parameters,
                        "timestamps": timestamps,
                        "outcome": env.info(),
                        "failure": failure,
                    },
                )
            )
            if progress is not None:
                progress({"stage": "collection", "episode": index + 1, "total": count, **episodes[-1].metadata})
    return episodes


@torch.no_grad()
def cache_visual_features(adapter, episodes, *, batch_size=16, progress=None):
    """Encode each frame once, on the chosen device; retain immutable CPU features."""
    if adapter.kind != "dinowm":
        raise ValueError("Feature caching currently supports DINO-WM only")
    adapter.eval()
    result = []
    for i, episode in enumerate(episodes):
        result.append(
            torch.cat(
                [
                    adapter.model.encoder(
                        frames.to(adapter.device).float() / (255 if frames.dtype == torch.uint8 else 1)
                    ).cpu()
                    for frames in episode.pixels.split(batch_size)
                ]
            )
        )
        if progress is not None:
            progress({"stage": "feature_cache", "episode": i + 1, "total": len(episodes)})
    return result


class LeWAMWindows(Dataset):
    """Aligned chunks with terminal image goals and successful-demo action masks.

    The simulator returns observation i with a constant per-episode delay d.
    Dropping the first d observations aligns frame i with applied action i;
    the last d actions then have no observed successor and are excluded.
    Recorded applied actions already account for actuator delay. This correction
    is specific to SO101Chess's queue contract, not arbitrary real recordings.
    History at episode start repeats the first frame, without crossing episodes.
    """

    def __init__(self, episodes, history, chunk_size):
        if min(history, chunk_size) < 1:
            raise ValueError("history and chunk_size must be positive")
        self.episodes, self.history, self.chunk_size = episodes, history, chunk_size
        self.offsets = []
        self.index = []
        for e, episode in enumerate(episodes):
            delay = episode.metadata.get("randomization", {}).get("observation_delay", 0)
            if not isinstance(delay, int) or delay < 0:
                raise ValueError("Observation delay must be a nonnegative integer")
            if len(episode.pixels) != len(episode.actions) + 1 or len(episode.proprio) != len(episode.pixels):
                raise ValueError("Episode must contain T actions and T+1 observations")
            if episode.action_valid is not None and (
                episode.action_valid.shape != (len(episode.actions),)
                or not torch.isfinite(episode.action_valid).all()
                or not ((episode.action_valid >= 0) & (episode.action_valid <= 1)).all()
            ):
                raise ValueError("action_valid must contain one finite weight in [0,1] per action")
            self.offsets.append(delay)
            self.index.extend((e, t) for t in range(len(episode.actions) - delay - chunk_size + 1))
        if not self.index:
            raise ValueError("No aligned episode has a complete LeWAM action chunk")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        e, t = self.index[index]
        episode, h, k, delay = self.episodes[e], self.history, self.chunk_size, self.offsets[e]
        frames = torch.arange(t - h + 1, t + k + 1).clamp_min(0) + delay
        action_ids = torch.arange(t - h + 1, t + k)
        actions = episode.actions[action_ids.clamp_min(0)].clone()
        actions[action_ids < 0] = 0
        pixels = episode.pixels[frames]
        scale = 255 if pixels.dtype == torch.uint8 else 1
        success = bool(episode.metadata.get("outcome", {}).get("success", False))
        return {
            "pixels": pixels.float() / scale,
            "states": episode.proprio[frames],
            "actions": actions,
            "goal_pixels": episode.pixels[-1].float() / scale,
            "goal_states": episode.proprio[-1],
            "action_valid": torch.full((k,), float(success))
            * (episode.action_valid[t : t + k] if episode.action_valid is not None else 1),
        }
