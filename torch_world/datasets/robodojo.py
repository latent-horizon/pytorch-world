"""RoboDojo episodes in XPolicyLab's trajectory format, and the layout shared with ``evaluation.RoboDojoPolicy``."""

from pathlib import Path
from typing import BinaryIO

import torch
from torch import Tensor

from torch_world.datasets.images import resize_images

ROBODOJO_TASKS = (
    "align_blocks", "arrange_largest_number", "build_tower", "classify_objects", "classify_objects_by_language",
    "cover_blocks", "deposit_coin", "fasten_screws", "fill_egg_holder", "fill_pen_holder", "fold_clothes",
    "general_pickup", "hang_mugs", "imitate_sorting_sequence", "insert_key", "insert_tubes", "make_kong", "make_toast",
    "match_and_pick_from_conveyor", "organize_table", "pack_objects_into_box", "pick_from_conveyor_by_image",
    "play_Xylophone", "play_stacking_toy", "play_tic_tac_toe", "plug_in_charger", "pour_balls_into_vase",
    "pour_by_language", "pour_liquid_into_cup", "press_by_number", "push_T", "put_bottles_into_dustbin",
    "solve_equation", "sort_nesting_dolls_by_size", "stack_blocks", "stack_blocks_by_language", "stack_bowls",
    "store_laptop_and_headphones", "store_tools_in_toolbox", "swap_T", "swap_blocks", "sweep_blocks",
)  # fmt: skip
"""RoboDojo's 42 simulation tasks, as its evaluation client names them."""

DUAL_X5 = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
"""RoboDojo's default embodiment, two ARX X5 arms with grippers, as XPolicyLab's ``get_robot_action_dim_info``."""


def robot_layout(action_type: str = "joint", robot: dict | None = None, plural: bool = False) -> list[tuple[str, int]]:
    """Keys and widths of a packed robot vector ``[left arm, left gripper, right arm, right gripper]``.

    Runtime observations and actions use singular keys (``left_arm_joint_state``); trajectory files use plural ones
    (``left_arm_joint_states``). ``action_type="ee"`` replaces each arm's joints by its 7-value pose
    ``[x, y, z, qw, qx, qy, qz]``.
    """
    robot = DUAL_X5 if robot is None else robot
    if action_type not in ("joint", "ee"):
        raise ValueError("action_type must be 'joint' or 'ee'")
    if len(robot["arm_dim"]) != 2 or len(robot["ee_dim"]) != 2:
        raise ValueError("RoboDojo's target robots have two arms; pass arm_dim and ee_dim for both")
    s = "s" if plural else ""
    layout = []
    for side, arm_dim, ee_dim in zip(("left", "right"), robot["arm_dim"], robot["ee_dim"]):
        arm = (f"{side}_arm_joint_state{s}", arm_dim) if action_type == "joint" else (f"{side}_ee_pose{s}", 7)
        layout += [arm, (f"{side}_ee_joint_state{s}", ee_dim)]
    return layout


def pack(group, layout: list[tuple[str, int]]):
    """Concatenate the fields of ``layout`` from a mapping into one float32 array, checking each width."""
    import numpy as np

    parts = []
    for key, width in layout:
        if key not in group:
            raise KeyError(f"Missing robot field {key!r}")
        value = np.asarray(group[key], dtype=np.float32)
        if value.shape[-1] != width:
            raise ValueError(f"{key} has width {value.shape[-1]}, expected {width}")
        parts.append(value)
    return np.concatenate(parts, -1)


def load_robodojo(
    path: str | Path | BinaryIO,
    *,
    camera: str = "cam_head",
    size: int = 64,
    action_type: str = "joint",
    robot: dict | None = None,
) -> dict[str, Tensor]:
    """One RoboDojo episode, an XPolicyLab v1.0 HDF5 file, as tensors ``(1, T, ...)`` for a `TrajectoryDataset`.

    ``observations`` are ``camera`` frames ``(1, T, 3, size, size)`` in uint8, ``states`` and ``actions`` the packed
    robot vectors of `robot_layout` ``(1, T, D)`` in robot units: joint angles in radians and grippers in [0, 1].
    Episodes differ in length, so combine them with ``ConcatDataset``. ``path`` may be an open file, such as
    ``HfFileSystem().open(...)``, which reads only the arrays used rather than every camera. Encoded frames are
    decoded with XPolicyLab's ``decode_image_bit``, the only decoder that gets both of its byte formats right.
    """
    import numpy as np

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py to read RoboDojo episodes: pip install 'torch-world[robodojo]'") from exc
    with h5py.File(path, "r") as file:
        colors = file["vision"][camera]["colors"]
        if colors.dtype == np.uint8 and colors.ndim == 4:
            frames = resize_images(colors[()], size)
        else:
            try:
                from XPolicyLab.utils.process_data import decode_image_bit
            except ImportError as exc:
                raise ImportError("Encoded RoboDojo frames need XPolicyLab on the Python path to decode") from exc
            frames = torch.cat(
                [resize_images(decode_image_bit(colors[i : i + 64]), size) for i in range(0, len(colors), 64)]
            )
        layout = robot_layout(action_type, robot, plural=True)
        states, actions = pack(file["state"], layout), pack(file["action"], layout)
    if not len(frames) == len(states) == len(actions):
        raise ValueError(f"{path}: {len(frames)} frames, {len(states)} states and {len(actions)} actions differ")
    return {
        "observations": frames[None],
        "states": torch.from_numpy(states)[None],
        "actions": torch.from_numpy(actions)[None],
    }
