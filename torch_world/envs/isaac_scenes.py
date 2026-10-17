"""Our Isaac Lab tabletop tasks, built from engine primitives and a robot asset.

This optional module is imported only inside an Isaac Lab runtime. It does not
import, inherit or register any upstream task configuration.
"""

import torch
from isaaclab import sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab_assets import FRANKA_PANDA_MENAGERIE_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

# Fixed target poses make the task observable and reproducible across model runs.
TARGETS = {"reach": (0.55, 0.2, 0.35), "push": (0.6, 0.18, 0.035), "lift": (0.5, 0.0, 0.25)}


def target_error(env, task):
    target = torch.tensor(TARGETS[task], device=env.device)
    return target - object_position(env, task)


def object_position(env, task):
    if task == "reach":
        robot = env.scene["robot"]
        # Resolve once per environment, not once per policy step.
        if not hasattr(env, "_torch_world_hand"):
            env._torch_world_hand = robot.find_bodies("panda_hand")[0][0]
        position = robot.data.body_pos_w[:, env._torch_world_hand]
    else:
        position = env.scene["object"].data.root_pos_w
    return position - env.scene.env_origins


def observations(env, task):
    robot = env.scene["robot"]
    return torch.cat(
        (robot.data.joint_pos, robot.data.joint_vel, object_position(env, task), target_error(env, task)), dim=-1
    )


def distance_reward(env, task):
    return -target_error(env, task).norm(dim=-1)


def success(env, task):
    return target_error(env, task).norm(dim=-1) < 0.04


@configclass
class TableScene(InteractiveSceneCfg):
    robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim.CuboidCfg(
            size=(20, 20, 0.1),
            collision_props=sim.CollisionBaseCfg(),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.87, 0.89, 0.92)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.75)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim.CuboidCfg(
            size=(1.2, 1.0, 0.7),
            collision_props=sim.CollisionBaseCfg(),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.32, 0.39, 0.47)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, -0.35)),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim.DomeLightCfg(intensity=2500))


class JointRangePositionAction(mdp.JointPositionAction):
    """Map normalized actions to the asset's joint limits, including reachable grasps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = self._asset.data.joint_pos_limits.clone()[:, self._joint_ids]
        self._offset = limits.mean(-1)
        self._scale = (limits[..., 1] - limits[..., 0]) / 2


@configclass
class ArmActions:
    arm = mdp.JointPositionActionCfg(
        class_type=JointRangePositionAction,
        asset_name="robot",
        joint_names=["panda_joint.*"],
        use_default_offset=False,
    )
    gripper = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger_joint.*"], scale=0.02, offset=0.02, use_default_offset=False
    )


def camera_image(env):
    return env.scene["camera"].data.output["rgb"][..., :3].clone()


CAMERAS = {
    "task": ((1.2, -1.05, 1.05), (0.35, 0.0, 0.35)),
    "overview": ((1.65, -1.65, 1.5), (0.4, 0.0, 0.3)),
}


def build_config(task, batch_size, max_episode_steps, *, render_size=512, shadows=True, camera="task"):
    """Build a fresh scene and MDP; no task registry or upstream task inheritance."""
    if task not in TARGETS:
        raise ValueError(f"Unknown tabletop task: {task}")
    scene = TableScene(num_envs=batch_size, env_spacing=2.5)
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {tuple(CAMERAS)}")
    eye, lookat = CAMERAS[camera]
    rotation = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([lookat]), up_axis="Z", device="cpu")
    orientation = tuple(quat_from_matrix(rotation)[0].tolist())
    scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=eye, rot=orientation, convention="opengl"),
        width=render_size,
        height=render_size,
        data_types=["rgb"],
        spawn=sim.PinholeCameraCfg(focal_length=24, horizontal_aperture=24, clipping_range=(0.01, 10.0)),
        renderer_cfg=NewtonWarpRendererCfg(enable_shadows=shadows),
    )
    # Noncolliding marker remains visible in observations and the GT reference.
    scene.target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim.SphereCfg(radius=0.025, visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.08, 0.85, 0.3))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGETS[task]),
    )
    if task == "reach":
        scene.obstacle = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Barrier",
            spawn=sim.CuboidCfg(
                size=(0.08, 0.25, 0.3),
                collision_props=sim.CollisionBaseCfg(),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.25, 0.38, 0.65)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0, 0.15)),
        )
    else:
        scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim.CuboidCfg(
                size=(0.06, 0.06, 0.07),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.95, 0.32, 0.06)),
                rigid_props=sim.RigidBodyPropertiesCfg(),
                mass_props=sim.MassPropertiesCfg(mass=0.1),
                collision_props=sim.CollisionBaseCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.12, 0.035)),
        )
    policy = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
    policy.state = ObservationTermCfg(func=observations, params={"task": task})
    camera_group = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
    camera_group.image = ObservationTermCfg(func=camera_image)
    cfg = ManagerBasedRLEnvCfg(
        scene=scene,
        decimation=4,
        episode_length_s=max_episode_steps / 30,
        observations={"policy": policy, "camera": camera_group},
        actions=ArmActions(),
        events={"reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")},
        rewards={
            "distance": RewardTermCfg(func=distance_reward, weight=1.0, params={"task": task}),
            "effort": RewardTermCfg(func=mdp.action_l2, weight=-0.001),
        },
        terminations={
            "success": TerminationTermCfg(func=success, params={"task": task}),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        compute_final_obs=True,
    )
    cfg.sim.dt = 1 / 120
    cfg.sim.render_interval = 4
    cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=200, nconmax=60, integrator="implicitfast"),
        num_substeps=2,
        use_cuda_graph=True,
    )
    return cfg
