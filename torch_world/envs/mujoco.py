"""Shared parts of the MuJoCo environments: a light studio scene, batched simulation, rendering and goal ghosts."""

import os
import sys

import numpy as np
import torch
from torch import Tensor

SKY, FLOOR, TABLE = (0.98, 0.98, 0.97), (0.9, 0.895, 0.88), (0.9, 0.87, 0.82)
GHOST = (0.92, 0.41, 0.2, 0.35)  # goal markers, drawn in videos only
GHOST_GROUP = 5  # unused by Menagerie models, which keep collision geoms in group 3


def import_mujoco():
    """Import ``mujoco``, rendering offscreen with EGL on headless Linux."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    return mujoco


def look_at(eye, target, up=(0, 0, 1)) -> list[float]:
    """Quaternion of a camera at ``eye`` looking at ``target``."""
    mujoco = import_mujoco()
    forward = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    mat = np.stack([right, np.cross(right, forward), -forward], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat.tolist()


def studio(scale: float = 1.0, table_center=(0.0, 0.0), table_half: float = 1.0):
    """A new ``MjSpec`` with a soft sky, a directional key light with shadows and a light wooden table at z = 0."""
    mujoco = import_mujoco()
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.32] * 3
    spec.visual.headlight.diffuse = [0.3] * 3
    spec.visual.headlight.specular = [0.02] * 3
    getattr(spec.visual, "global").offwidth = 512
    getattr(spec.visual, "global").offheight = 512
    spec.visual.quality.shadowsize = 4096
    sky = spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    )
    sky.rgb1, sky.rgb2, sky.width, sky.height = SKY, (0.9, 0.9, 0.88), 512, 3072
    world = spec.worldbody
    world.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.6 * scale, -scale, 2.5 * scale],
        dir=[-0.25, 0.4, -1],
        diffuse=[0.55] * 3,
        specular=[0.1] * 3,
        castshadow=True,
        softness=0.3,
    )
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], rgba=[*FLOOR, 1])
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[table_half, table_half, 0.02],
        pos=[*table_center, -0.02],
        rgba=[*TABLE, 1],
    )
    return spec


def add_ghost(spec, geoms: list[dict]):
    """A mocap body named ``goal`` drawn with translucent ``geoms``, visible only when rendering with a goal."""
    mujoco = import_mujoco()
    ghost = spec.worldbody.add_body(name="goal", mocap=True, pos=[0, 0, -1])
    for g in geoms:
        ghost.add_geom(**g, rgba=list(GHOST), contype=0, conaffinity=0, group=GHOST_GROUP, mass=0)
    return mujoco


class MujocoEnv:
    """A batch of MuJoCo simulations, simulated sequentially on the CPU, with observations as rendered images.

    Subclasses build ``self.model`` and implement ``reset``, ``step``, ``_place(data, state)`` (set a simulation to a
    state), ``_state(data)`` and ``_ghost(goal)`` (the goal marker's position and quaternion), plus ``radius``,
    ``position``, ``distance``, ``success`` and ``tasks``.
    """

    def __init__(self, model, batch_size: int, size: int, camera: str, device=None):
        self.mujoco = import_mujoco()
        self.model, self.batch_size, self.size, self.camera, self.device = model, batch_size, size, camera, device
        self.data = [self.mujoco.MjData(model) for _ in range(batch_size)]
        self.renderers: dict = {}
        self.ghost = self.mujoco.MjvOption()
        self.ghost.geomgroup[GHOST_GROUP] = 1
        self.hidden = self.mujoco.MjvOption()
        self.hidden.geomgroup[GHOST_GROUP] = 0

    def _sync(self):
        s = np.array([self._state(b, d) for b, d in enumerate(self.data)])
        self.state = torch.tensor(s, dtype=torch.float32, device=self.device)

    def render(
        self,
        state: Tensor | None = None,
        camera: str | None = None,
        size: int | None = None,
        goal: Tensor | None = None,
    ) -> Tensor:
        """Images ``(B, 3, size, size)`` in ``[0, 1]`` of the running simulations, or of the given ``state``.

        With ``goal`` states ``(B, state_dim)``, a translucent marker shows each goal, for videos and figures; the
        model's observations never show it.
        """
        size, camera = size or self.size, camera or self.camera
        if (size, camera) not in self.renderers:
            self.renderers[size, camera] = self.mujoco.Renderer(self.model, size, size)
        renderer = self.renderers[size, camera]
        data = self.data if state is None else self._posed(state)
        goals = None if goal is None else goal.double().cpu().numpy()
        images = []
        for b, d in enumerate(data):
            if goals is not None:
                d.mocap_pos[0], d.mocap_quat[0] = self._ghost(goals[b])
                self.mujoco.mj_kinematics(self.model, d)
            renderer.update_scene(d, camera, scene_option=self.hidden if goals is None else self.ghost)
            images.append(renderer.render())
        return torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).float() / 255

    def _posed(self, state: Tensor) -> list:
        out = []
        for row in state.double().cpu().numpy():
            d = self.mujoco.MjData(self.model)
            self._place(d, row)
            out.append(d)
        return out

    def tasks(self, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Random start states and goal states that a scripted policy reaches from them; the batch ends at the starts."""
        raise NotImplementedError
