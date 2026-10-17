"""Shared GT references and successful-example displays for every benchmark."""

import torch

from torch_world.envs import get_spec
from torch_world.envs.registry import environment_scope, is_environment


def capture_frame(env, observation=None, *, target=False):
    """Snapshot a presentation frame; store RGB as uint8, preserving state vectors."""
    renderer = getattr(env, "render_goal_frame" if target else "render_frame", None)
    frame = renderer() if renderer else observation
    frame = frame[0].detach().cpu().clone()
    if frame.ndim == 3 and frame.dtype != torch.uint8:
        frame = frame.clamp(0, 1).mul(255).round().to(torch.uint8)
    return frame


def episode_indices(length, stride=2):
    """Subsample an episode without dropping its terminal observation."""
    if length < 1 or stride < 1:
        raise ValueError("Episode length and stride must be positive")
    return list(range(0, length - 1, stride)) + [length - 1]


def episode_gif(frames, target, *, title, success=None, fps=8, stride=1, target_title="GT", frame_steps=None):
    """Return the library GIF object with a persistent, explicitly labeled GT tile."""
    from torch_world import plots

    if frames.ndim != 4 or target.ndim != 3:
        raise ValueError("GIFs require RGB episode frames and an RGB GT reference")
    label = title if success is None else f"{'✓ success' if success else '✗ failure'} · {title}"
    indices = episode_indices(len(frames), stride)
    frame_steps = list(range(len(frames))) if frame_steps is None else frame_steps
    if len(frame_steps) != len(frames):
        raise ValueError("frame_steps must match the episode length")
    labels = [frame_steps[i] for i in indices]
    frames = frames[indices]
    videos = torch.stack((target.expand_as(frames), frames))
    return plots.episodes(videos, titles=[target_title, label], ncols=2, size=3.5, fps=fps, hold=3, frame_steps=labels)


def show_env(environment, *, seed=7, env_kwargs=None) -> None:
    """Preview a named task or registry ID with its initial observation and GT."""
    from IPython.display import Markdown, display

    from torch_world import plots

    spec = get_spec(environment)
    kwargs = dict(env_kwargs or {})
    if not is_environment(environment):
        kwargs["batch_size"] = 1
    with environment_scope(environment, **kwargs) as env:
        spec = get_spec(env)
        obs = env.reset(seed=seed)
        initial = capture_frame(env, obs)
        target = capture_frame(env, target=True)
        display(
            Markdown(
                f"**{spec.id} · {spec.backend}** — {spec.description}. "
                f"{env.action_dim} normalized action components in [-1, 1]."
            )
        )
        visual = (
            plots.images(
                torch.stack((initial, target)), titles=["Initial", getattr(env, "gt_label", "GT")], ncols=2, size=3.5
            )
            if initial.ndim == 3
            else state_figure({"Initial": initial[None]}, target)
        )
        display(visual)


@torch.no_grad()
def show_success(env_id, *, policy=None, seed=7, steps=None, env_kwargs=None, path=None):
    """Display a verified successful episode GIF beside GT; optionally save it.

    Reset the world, replay its reference policy (or the supplied controller), and
    check the real transition success flag. Failed or unavailable demonstrations
    raise ValueError; a static target is never substituted for an episode.
    A supplied policy receives batched observations and returns normalized actions.
    """
    from pathlib import Path

    from IPython.display import Markdown, display

    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")
    spec = get_spec(env_id)
    kwargs = dict(env_kwargs or {})
    if not is_environment(env_id):
        kwargs["batch_size"] = 1
    with environment_scope(env_id, **kwargs) as env:
        spec = get_spec(env)
        obs = env.reset(seed=seed)
        policy = env.reference_policy() if policy is None else policy
        target = capture_frame(env, target=True)
        frames = [capture_frame(env, obs)]
        if frames[0].ndim != 3 or target.ndim != 3:
            raise ValueError("Success GIFs require RGB render_frame() and render_goal_frame() implementations")
        # Demonstrations may need more time than the experiment evaluation budget.
        # Restore the caller's limit even when policy execution or rendering fails.
        reference_steps = getattr(policy, "max_steps", env.max_episode_steps)
        budget = steps if steps is not None else reference_steps
        stride = max(1, (budget + 179) // 180)
        frame_steps = [0]
        original_limit = env.max_episode_steps
        succeeded = False
        try:
            env.max_episode_steps = max(original_limit, budget)
            for step in range(1, budget + 1):
                obs = env.step(policy(obs))
                succeeded = env.transition.success is not None and bool(env.transition.success[0])
                finished = succeeded or bool(env.transition.done[0]) or step == budget
                if step % stride == 0 or finished:
                    frames.append(capture_frame(env, obs))
                    frame_steps.append(step)
                if finished:
                    break
        finally:
            env.max_episode_steps = original_limit
        if not succeeded:
            raise ValueError(f"{spec.id}: the supplied policy did not produce a verified success")
        visual = episode_gif(
            torch.stack(frames),
            target,
            title="Reference",
            success=True,
            target_title=getattr(env, "gt_label", "GT"),
            frame_steps=frame_steps,
        )
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            visual.save(destination)
        display(
            Markdown(
                f"**{spec.id}** — verified reference episode · {step} simulator steps; separate from model evaluation."
            )
        )
        display(visual)
        return visual


def state_figure(frames, target):
    """State trajectories against GT; unspecified reference coordinates are omitted."""
    from torch_world import plots

    indices = target.isfinite().nonzero().flatten()[:6]
    with plots.style() as plt:
        fig, axes = plt.subplots(
            max(len(frames), 1), 1, figsize=(7.8, 2.5 * max(len(frames), 1)), squeeze=False, layout="constrained"
        )
        for ax, (name, video) in zip(axes[:, 0], frames.items() or [("GT", None)]):
            for j, index in enumerate(indices.tolist()):
                color = plots.SERIES[j % len(plots.SERIES)]
                if video is not None:
                    ax.plot(video[:, index].numpy(), color=color, label=f"state[{index}]")
                ax.axhline(float(target[index]), color=color, ls="--", label=f"GT[{index}]")
            ax.set(title=name, xlabel="step", ylabel="state / GT")
            ax.legend(ncols=3, fontsize=7)
        plt.close(fig)
    return fig
