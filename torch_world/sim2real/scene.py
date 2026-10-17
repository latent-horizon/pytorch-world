"""Procedural chess geometry; visual and collision shapes share measurements."""

import numpy as np

from torch_world.envs.mujoco import import_mujoco


def square_position(task, square):
    if not isinstance(square, str) or len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError("Square must be algebraic notation a1..h8")
    xy = np.array([ord(square[0]) - ord("a"), int(square[1]) - 1]) * task.square_size
    c, s = np.cos(task.board_yaw), np.sin(task.board_yaw)
    return np.asarray(task.board_origin) + np.r_[np.array([[c, -s], [s, c]]) @ xy, 0.0]


def chess_scene(task, piece, color):
    """Callback for the existing arm/studio builder; piece origin is its base."""
    mj = import_mujoco()
    p = task.pieces[piece]

    def build(spec):
        appearance = task.appearance
        # Deterministic procedural grain: no photographs or generated assets are
        # baked into observations, and visual texture never changes collisions.
        y, x = np.mgrid[:128, :128] / 128
        rng = np.random.default_rng(19)
        grain = (
            0.5 * np.sin(2 * np.pi * (19 * y + 0.6 * np.sin(6 * x)))
            + 0.25 * np.sin(2 * np.pi * (47 * y + 0.3 * np.sin(11 * x)))
            + rng.normal(0, 0.15, x.shape)
        )
        for name, rgb in (("light", appearance.light), ("dark", appearance.dark), ("frame", appearance.frame)):
            pixels = np.clip(np.asarray(rgb)[None, None] * (1 + appearance.grain * grain[..., None]), 0, 1)
            texture = spec.add_texture(
                name=f"wood_{name}", type=mj.mjtTexture.mjTEXTURE_2D, width=128, height=128, nchannel=3
            )
            texture.data = (pixels * 255).astype(np.uint8).tobytes()
            material = spec.add_material(name=f"wood_{name}", specular=0.25, shininess=0.35)
            material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = f"wood_{name}"
        spec.add_material(name="brass", rgba=[0.72, 0.48, 0.15, 1], specular=0.7, shininess=0.6)
        rotation = np.array(
            [[np.cos(task.board_yaw), -np.sin(task.board_yaw)], [np.sin(task.board_yaw), np.cos(task.board_yaw)]]
        )
        quat = [np.cos(task.board_yaw / 2), 0, 0, np.sin(task.board_yaw / 2)]

        def case_geom(name, xy, z, size, material, collision=True):
            position = np.asarray(task.board_origin) + np.r_[rotation @ xy, z]
            spec.worldbody.add_geom(
                name=name,
                type=mj.mjtGeom.mjGEOM_BOX,
                pos=position.tolist(),
                size=size,
                quat=quat,
                material=material,
                contype=1 if collision else 0,
                conaffinity=1 if collision else 0,
            )

        side = task.square_size * 8
        centre = task.square_size * 3.5
        border, thickness = appearance.border, appearance.thickness
        if thickness > 0:
            case_geom(
                "board_case",
                [centre, centre],
                -thickness / 2 - appearance.surface,
                [side / 2 + border, side / 2 + border, thickness / 2],
                "wood_frame",
            )
        if border > 0:
            for axis in range(2):
                for sign in (-1, 1):
                    xy = np.array([centre, centre])
                    xy[axis] += sign * (side / 2 + border / 2)
                    size = [side / 2 + border, side / 2 + border, appearance.surface / 2]
                    size[axis] = border / 2
                    case_geom(f"board_rim_{axis}_{sign}", xy, -appearance.surface / 2, size, "wood_frame")
            # Two case catches on the near (-file) edge. Cosmetic metal does not
            # create extra collision surfaces or alter the grasp task.
            for i, along in enumerate((side * 0.22, side * 0.78)):
                xy = [-task.square_size / 2 - border - 0.005, along - task.square_size / 2]
                case_geom(f"latch_{i}", xy, -0.006, [0.006, 0.009, 0.0015], "brass", False)
                for sign in (-1, 1):
                    case_geom(
                        f"latch_rail_{i}_{sign}",
                        [xy[0] - 0.007, xy[1] + sign * 0.008],
                        -0.009,
                        [0.008, 0.001, 0.001],
                        "brass",
                        False,
                    )
                case_geom(f"latch_end_{i}", [xy[0] - 0.015, xy[1]], -0.009, [0.001, 0.008, 0.001], "brass", False)
        # Warm indoor illumination; keep its direction fixed in the base frame.
        spec.visual.headlight.ambient = [0.40, 0.39, 0.36]
        spec.visual.headlight.diffuse = [0.28, 0.27, 0.24]
        for light in spec.worldbody.lights:
            light.diffuse = [0.60, 0.57, 0.51]
            light.specular = [0.25, 0.20, 0.14]
            light.dir = [-0.3, 0.5, -1]
        # Tight shadow frustum keeps small chess pieces from casting blocky shadows.
        spec.visual.map.shadowscale = 0.3
        for file in "abcdefgh":
            for rank in "12345678":
                position = square_position(task, file + rank)
                position[2] -= appearance.surface / 2
                light = (ord(file) + int(rank)) % 2 == 0
                spec.worldbody.add_geom(
                    name=f"square_{file}{rank}",
                    type=mj.mjtGeom.mjGEOM_BOX,
                    pos=position.tolist(),
                    size=[task.square_size / 2, task.square_size / 2, appearance.surface / 2],
                    quat=[np.cos(task.board_yaw / 2), 0, 0, np.sin(task.board_yaw / 2)],
                    material="wood_light" if light else "wood_dark",
                    friction=[1.2, 0.01, 0.001],
                )
        body = spec.worldbody.add_body(name="piece", pos=square_position(task, "c4").tolist())
        body.add_freejoint(name="piece")
        rgba = [0.85, 0.81, 0.7, 1] if color == "white" else [0.09, 0.075, 0.06, 1]
        shapes = []

        def shape(kind, size, pos, fraction):
            shapes.append((kind, size, pos, fraction))

        r, neck, h = p.diameter / 2, p.neck_diameter / 2, p.height
        base_h = 0.12 * h
        shape(mj.mjtGeom.mjGEOM_CYLINDER, [r, base_h / 2, 0], [0, 0, base_h / 2], 0.45)
        shape(mj.mjtGeom.mjGEOM_CYLINDER, [neck, (h * 0.72 - base_h) / 2, 0], [0, 0, (h * 0.72 + base_h) / 2], 0.35)
        if piece == "pawn":
            radius = min(r * 0.65, h * 0.14)
            shape(mj.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], [0, 0, h - radius], 0.2)
        elif piece == "rook":
            shape(mj.mjtGeom.mjGEOM_CYLINDER, [r * 0.82, h * 0.14, 0], [0, 0, h * 0.86], 0.2)
        elif piece == "knight":
            shape(mj.mjtGeom.mjGEOM_BOX, [r * 0.52, neck * 0.7, h * 0.14], [r * 0.25, 0, h * 0.86], 0.2)
        elif piece == "bishop":
            shape(mj.mjtGeom.mjGEOM_ELLIPSOID, [neck, neck, h * 0.14], [0, 0, h * 0.86], 0.2)
        elif piece == "queen":
            shape(mj.mjtGeom.mjGEOM_CYLINDER, [r * 0.7, h * 0.09, 0], [0, 0, h * 0.81], 0.15)
            shape(mj.mjtGeom.mjGEOM_SPHERE, [h * 0.05, 0, 0], [0, 0, h * 0.95], 0.05)
        else:
            shape(mj.mjtGeom.mjGEOM_BOX, [h * 0.025, h * 0.025, h * 0.14], [0, 0, h * 0.86], 0.12)
            shape(mj.mjtGeom.mjGEOM_BOX, [h * 0.09, h * 0.025, h * 0.025], [0, 0, h * 0.91], 0.08)
        for i, (kind, size, pos, fraction) in enumerate(shapes):
            body.add_geom(
                name=f"piece_{i}",
                type=kind,
                size=size,
                pos=pos,
                rgba=rgba,
                mass=p.mass * fraction,
                friction=[p.friction, 0.01, 0.001],
                condim=6,
                solref=[0.01, 1],
            )

    return build
