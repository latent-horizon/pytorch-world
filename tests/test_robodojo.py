"""RoboDojo through XPolicyLab: the policy server model and the episode loader share one robot and image layout."""

import numpy as np
import pytest
import torch

from torch_world.datasets.robodojo import robot_layout
from torch_world.evaluation import RoboDojoPolicy

KEYS = ["left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"]


def observation(env_idx, value=0.0):
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :32] = 255  # left half white: a 48 x 64 frame squeezed to a square stays half white
    state = {key: np.full(width, value + i, np.float32) for i, (key, width) in enumerate(robot_layout())}
    return {"env_idx": env_idx, "vision": {"cam_head": {"color": image}}, "state": state, "instruction": "stack"}


class Chunks:
    def __init__(self):
        self.seen, self.resets = None, 0

    def __call__(self, observations):
        self.seen = observations
        actions = torch.linspace(-1, 1, 14).expand(len(observations), 2, -1).clone()
        actions[:, 1] = 1
        return actions

    def reset(self):
        self.resets += 1


def test_policy_serves_batched_chunks_in_robot_units():
    policy = Chunks()
    low, high = torch.zeros(14), torch.full((14,), 2.0)
    model = RoboDojoPolicy(policy, size=8, action_bounds=(low, high))
    model.update_obs_batch([observation(3), observation(5)])
    actions = model.get_action_batch([5, 3])
    assert (
        policy.seen.shape == (2, 3, 8, 8)
        and policy.seen[:, :, :, :4].eq(1).all()
        and policy.seen[:, :, :, 4:].eq(0).all()
    )
    assert len(actions) == 2 and len(actions[0]) == 2 and list(actions[0][0]) == KEYS
    assert actions[0][0]["left_arm_joint_state"].shape == (6,) and actions[0][0]["right_ee_joint_state"].shape == (1,)
    assert actions[0][0]["left_arm_joint_state"][0] == 0 and actions[0][1]["right_ee_joint_state"][0] == 2
    assert len(model.get_action_batch()) == 2  # defaults to the environments last observed
    model.update_obs(observation(0))
    assert len(model.get_action()) == 2  # one environment: a single chunk
    model.reset()
    assert policy.resets == 2  # once on construction, once per episode
    with pytest.raises(RuntimeError, match="before update_obs"):
        model.get_action()


def test_state_observations_follow_the_packed_layout():
    model = RoboDojoPolicy(lambda obs: torch.zeros(len(obs), 14), observation="state")
    model.update_obs(observation(0, value=10))
    assert model.observe(observation(0, value=10)).tolist() == [10] * 6 + [11] + [12] * 6 + [13]
    assert len(model.get_action()) == 1  # (B, A) actions are one-step chunks
    ee = RoboDojoPolicy(lambda obs: torch.zeros(len(obs), 16), observation="state", action_type="ee")
    assert [k for k, _ in ee.layout] == ["left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"]


def test_policy_rejects_mismatched_actions_and_layouts():
    model = RoboDojoPolicy(lambda obs: torch.zeros(len(obs), 13), size=8)
    model.update_obs(observation(0))
    with pytest.raises(ValueError, match="finite actions"):
        model.get_action()
    with pytest.raises(ValueError, match="two arms"):
        RoboDojoPolicy(lambda obs: obs, robot={"arm_dim": [6], "ee_dim": [1]})
    with pytest.raises(ValueError, match="action_bounds"):
        RoboDojoPolicy(lambda obs: obs, action_bounds=(torch.zeros(3), torch.ones(3)))


def test_load_robodojo_matches_the_policy_inputs(tmp_path):
    h5py = pytest.importorskip("h5py")
    from torch.utils.data import ConcatDataset

    from torch_world.datasets import TrajectoryDataset, load_robodojo

    T, layout = 5, robot_layout(plural=True)
    with h5py.File(tmp_path / "episode.hdf5", "w") as file:
        file["vision/cam_head/colors"] = np.stack([observation(0)["vision"]["cam_head"]["color"]] * T)
        for i, (key, width) in enumerate(layout):
            file[f"state/{key}"] = np.full((T, width), i, np.float32)
            file[f"action/{key}"] = np.full((T, width), -i, np.float32)
        file["instruction"] = "stack the blocks"
    episode = load_robodojo(tmp_path / "episode.hdf5", size=8)
    assert episode["observations"].shape == (1, T, 3, 8, 8) and episode["observations"].dtype == torch.uint8
    assert episode["states"].shape == episode["actions"].shape == (1, T, 14) and episode["actions"][0, 0, 6] == -1
    policy_input = RoboDojoPolicy(lambda obs: obs, size=8).observe(observation(0))
    assert torch.equal(episode["observations"][0, 0].float() / 255, policy_input)
    assert len(ConcatDataset([TrajectoryDataset(episode, window=2)] * 2)) == 8


def test_relative_actions_are_offsets_from_the_current_state():
    model = RoboDojoPolicy(lambda obs: torch.ones(len(obs), 3, 14), size=8, relative=True)
    model.update_obs_batch([observation(0, value=10), observation(1, value=20)])
    chunks = model.get_action_batch()
    assert chunks[0][2]["left_arm_joint_state"].tolist() == [11] * 6 and chunks[1][0]["right_ee_joint_state"][0] == 24
    with pytest.raises(ValueError, match="joint offsets"):
        RoboDojoPolicy(lambda obs: obs, action_type="ee", relative=True)
