"""Run in Isaac Lab 3.x: python tests/isaaclab_smoke.py FrankaObstacleReach-v0."""

import argparse

import torch

from torch_world.envs import make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--camera", choices=("task", "overview"), default="task")
    args = parser.parse_args()
    with make(
        args.task,
        batch_size=2,
        device="cuda:0",
        max_episode_steps=2,
        observation="rgb",
        quality=args.quality,
        camera=args.camera,
    ) as env:
        obs = env.reset(seed=7)
        state, initial = env.state.clone(), env.render_frame()
        obj = env.env.scene["object"] if env.task != "reach" else None
        object_state = obj.data.root_state_w.clone() if obj is not None else None
        gt = env.render_goal_frame()
        assert initial.shape == gt.shape == (2, 3, env.render_size, env.render_size)
        assert initial.dtype == torch.uint8 and initial.float().std() > 5
        torch.testing.assert_close(env.state, state)
        torch.testing.assert_close(env.render_frame(), initial)
        if obj is not None:
            torch.testing.assert_close(obj.data.root_state_w.clone(), object_state)
            assert not torch.equal(initial, gt)
        for _ in range(2):
            obs = env.step(torch.zeros(2, env.action_dim, device="cuda:0"))
            assert torch.isfinite(obs).all()
        assert env.transition.done.all()
        final = env.render_frame()
        env.step(torch.zeros(2, env.action_dim, device="cuda:0"))
        torch.testing.assert_close(env.render_frame(), final)
        assert obs.shape == (2, 3, env.size, env.size)
        print("PASS", args.task, args.quality, args.camera, tuple(obs.shape), tuple(final.shape))


if __name__ == "__main__":
    main()
