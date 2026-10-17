"""Owned MuJoCo scenes and objectives; no upstream environment dependencies.

Scene builders run once. The simulation loop only integrates active lanes; reward
and success calculations operate on the entire batch with torch operations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from torch_world.envs.mjcf_assets import MjcfAttachment


@dataclass(frozen=True)
class Scene:
    """Complete task specification, with joint-coordinate targets and tolerances."""

    bodies: str
    actuators: str
    initial: tuple[float, ...]
    target: tuple[float, ...]
    indices: tuple[int, ...]
    tolerance: tuple[float, ...]
    angular: tuple[int, ...] = ()
    gravity: str = "0 0 -9.81"
    velocity_tolerance: float = 0.5
    follow_camera: bool = False
    directional_light: bool = False
    attachments: tuple[MjcfAttachment, ...] = ()
    success_bounds: tuple[tuple[int, float, float], ...] = ()

    def __post_init__(self):
        if not self.initial or len(self.initial) != len(self.target):
            raise ValueError("initial and target must have equal, nonzero coordinate counts")
        if not self.indices or len(self.indices) != len(self.tolerance):
            raise ValueError("Each scored coordinate needs a tolerance")
        if any(i < 0 or i >= len(self.initial) for i in self.indices):
            raise ValueError("Scored coordinates must index the scene configuration")
        if any(t <= 0 for t in self.tolerance) or self.velocity_tolerance <= 0:
            raise ValueError("Success tolerances must be positive")
        if any(i < 0 or i >= len(self.initial) or lo >= hi for i, lo, hi in self.success_bounds):
            raise ValueError("Success bounds need valid coordinate indices and increasing limits")

    def xml(self) -> str:
        xml = f'''<mujoco model="torch_world_task">
          <compiler angle="radian"/><option timestep="0.002" gravity="{self.gravity}" integrator="implicitfast"/>
          <default><joint damping="0.1" armature="0.01"/><geom friction="0.7 0.01 0.001" density="500" rgba="0.2 0.5 0.75 1"/>
            <motor ctrllimited="true" ctrlrange="-1 1"/><position ctrllimited="true"/></default>
          <asset>
            <texture type="skybox" builtin="gradient" rgb1=".85 .9 .96" rgb2=".98 .99 1" width="512" height="3072"/>
            <texture name="floor" type="2d" builtin="checker" rgb1=".87 .89 .91" rgb2=".94 .95 .96" width="512" height="512"/>
            <material name="floor" texture="floor" texrepeat="8 8" reflectance=".08"/>
          </asset>
          <visual><rgba haze=".93 .95 .98 1"/><map znear=".01"/></visual>
          <worldbody><light directional="{str(self.directional_light).lower()}" pos="-2 -3 5" dir=".3 .3 -1" diffuse=".6 .6 .6"/>
          <light directional="{str(self.directional_light).lower()}" pos="2 1 3" dir="-.3 -.2 -1" diffuse=".2 .2 .2" castshadow="false"/>
          <geom type="plane" size="5 5 .1" friction=".01 .001 .001" material="floor" rgba="1 1 1 1"/>
          {self.bodies}</worldbody>
          <actuator>{self.actuators}</actuator></mujoco>'''
        if not self.attachments:
            return xml
        root = ET.fromstring(xml)
        for attachment in self.attachments:
            attachment.attach(root)
        return ET.tostring(root, encoding="unicode")

    def assets(self) -> dict[str, bytes]:
        """Binary assets for MuJoCo's virtual filesystem, including wheel installs."""
        return {name: data for attachment in self.attachments for name, data in attachment.assets().items()}


def motor(joint: str, gear: float) -> str:
    return f'<motor joint="{joint}" gear="{gear}"/>'


def servo(joint: str, lower: float, upper: float, kp: float = 500) -> str:
    return f'<position joint="{joint}" kp="{kp}" kv="40" ctrlrange="{lower} {upper}"/>'


def cartpole(double=False):
    second = (
        '<body pos="0 0 .6"><joint name="elbow" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 .6" size=".035"/></body>'
        if double
        else ""
    )
    body = f"""<body pos="0 0 .8"><joint name="cart" type="slide" axis="1 0 0" range="-1.8 1.8"/>
    <geom type="box" size=".15 .12 .08"/><body><joint name="pole" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 .6" size=".035"/>{second}</body></body>"""
    # The hinge starts near upright for double-pole stabilization; single pole swings up.
    initial = (0.0, 0.3, -0.3) if double else (0.0, 2.7)
    n = len(initial)
    return Scene(body, motor("cart", 35), initial, (0.0,) * n, tuple(range(n)), (0.15,) * n, tuple(range(1, n)))


def reaction_wheel():
    return Scene(
        """<body pos="0 0 .8"><joint name="pole" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 .6" size=".04"/><body pos="0 0 .6"><joint name="wheel" axis="0 1 0"/>
      <geom type="cylinder" size=".15 .04" euler="1.5708 0 0"/></body></body>""",
        motor("wheel", 2),
        (2.8, 0.0),
        (0.0, 0.0),
        (0,),
        (0.15,),
        (0,),
    )


def ball_plate():
    return Scene(
        """<body pos="0 0 .7"><joint name="roll" axis="1 0 0" range="-.2 .2"/><joint name="pitch" axis="0 1 0" range="-.2 .2"/>
      <geom type="box" size=".5 .5 .025"/><body pos="0 0 .065"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".04" rgba=".98 .39 .08 1" friction=".02 .001 .001"/></body></body>""",
        servo("roll", -0.2, 0.2) + servo("pitch", -0.2, 0.2),
        (0.0, 0.0, -0.25, 0.2),
        (0.0, 0.0, 0.2, -0.2),
        (2, 3),
        (0.05, 0.05),
    )


def quadrotor(payload=False):
    load = (
        """<body pos="0 0 -.05"><joint name="cable" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 -.45" size=".01" mass=".02"/>
      <geom type="sphere" pos="0 0 -.45" size=".06" mass=".15"/></body>"""
        if payload
        else ""
    )
    body = f"""<body pos="0 0 1"><joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><joint name="pitch" axis="0 1 0"/>
      <geom type="box" size=".25 .06 .03" mass=".6"/><site name="left" pos="-.22 0 0"/><site name="right" pos=".22 0 0"/>{load}</body>"""
    controls = "".join(
        f'<general site="{s}" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="0 8"/>' for s in ("left", "right")
    )
    n = 4 if payload else 3
    return Scene(
        body,
        controls,
        (0.0,) * n,
        (0.6, 0.3, 0.0) + ((0.0,) if payload else ()),
        tuple(range(n)),
        (0.1,) * n,
        tuple(range(2, n)),
        follow_camera=True,
        directional_light=True,
    )


def parking():
    body = """<body pos="0 0 .12"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" axis="0 0 1"/>
      <geom type="box" size=".22 .13 .07" mass="2"/><geom type="sphere" pos=".16 0 -.08" size=".04" friction=".001 .001 .001"/>
      <body pos="0 .16 0"><joint name="left" axis="0 1 0"/><geom type="cylinder" size=".12 .025" euler="1.5708 0 0"/></body>
      <body pos="0 -.16 0"><joint name="right" axis="0 1 0"/><geom type="cylinder" size=".12 .025" euler="1.5708 0 0"/></body></body>
      <geom type="box" pos=".65 .38 .15" size=".4 .05 .15"/><geom type="box" pos=".65 -.38 .15" size=".4 .05 .15"/>"""
    return Scene(
        body,
        motor("left", 2) + motor("right", 2),
        (0.0,) * 5,
        (0.65, 0.0, 0.0, 0.0, 0.0),
        (0, 1, 2),
        (0.12, 0.1, 0.15),
        (2,),
    )


def hopper():
    body = """<body pos="0 0 .9"><joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><joint name="pitch" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".08"/><body><joint name="hip" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.35" size=".05"/><body pos="0 0 -.35"><joint name="knee" axis="0 1 0" range="-1.8 .1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/><geom type="sphere" pos="0 0 -.4" size=".07"/></body></body></body>
      <geom type="box" pos=".55 0 .1" size=".08 .4 .1"/>"""
    return Scene(
        body,
        motor("hip", 30) + motor("knee", 30),
        (0.0, 0.0, 0.0, 0.1, -0.2),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0, 1, 2),
        (0.15, 0.2, 0.2),
        (2,),
    )


def arm(spatial=False):
    root = '<joint name="yaw" axis="0 0 1"/>' if spatial else ""
    body = f"""<body pos="0 0 .6">{root}<joint name="shoulder" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .4 0 0" size=".04"/>
      <body pos=".4 0 0"><joint name="elbow" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .35 0 0" size=".035"/></body></body>
      <geom type="box" pos=".4 0 .25" size=".08 .12 .25"/>"""
    joints = ("yaw", "shoulder", "elbow") if spatial else ("shoulder", "elbow")
    n = len(joints)
    return Scene(
        body,
        "".join(motor(j, 12) for j in joints),
        (0.0,) * n,
        ((0.7,) if spatial else ()) + (-0.8, 1.2),
        tuple(range(n)),
        (0.12,) * n,
        tuple(range(n)),
    )


def pusher(mode="push"):
    friction = ".03 .001 .001" if mode == "slide" else ".6 .01 .001"
    gate = (
        '<geom type="box" pos=".25 .38 .15" size=".04 .22 .15"/><geom type="box" pos=".25 -.38 .15" size=".04 .22 .15"/>'
        if mode == "gate"
        else ""
    )
    body = f'''<body pos="0 0 .1"><joint name="hand_x" type="slide" axis="1 0 0" range="-.7 .8"/><joint name="hand_y" type="slide" axis="0 1 0" range="-.6 .6"/>
      <geom type="cylinder" size=".06 .1"/></body><body pos="0 0 .06"><joint name="object_x" type="slide" axis="1 0 0"/><joint name="object_y" type="slide" axis="0 1 0"/>
      <joint name="object_z" type="slide" axis="0 0 1"/><joint name="object_yaw" axis="0 0 1"/><geom type="box" size=".08 .08 .06" friction="{friction}" rgba=".85 .45 .15 1"/></body>{gate}'''
    return Scene(
        body,
        servo("hand_x", -0.7, 0.8) + servo("hand_y", -0.6, 0.6),
        (-0.35, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.35, 0.0, 1.25 if mode == "slide" else 0.55, 0.0, 0.0, 0.0),
        (2, 3),
        (0.07, 0.07),
    )


def gantry(tool, extra, initial, target, indices, tolerance):
    """Cartesian tool driven by physical position actuators, with contact-only objects."""
    body = f"""<body pos="0 0 .5"><joint name="x" type="slide" axis="1 0 0" range="-.6 .7"/><joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/>
      <joint name="z" type="slide" axis="0 0 1" range="-.45 .4"/>{tool}</body>{extra}"""
    controls = servo("x", -0.6, 0.7) + servo("y", -0.5, 0.5) + servo("z", -0.45, 0.4)
    return Scene(body, controls, initial, target, indices, tolerance)


def peg():
    tool = '<geom type="cylinder" pos="0 0 -.12" size=".025 .12"/>'
    fixture = "".join(
        f'<geom type="box" pos="{x} {y} .08" size="{sx} {sy} .08"/>'
        for x, y, sx, sy in (
            (0.3, 0.13, 0.16, 0.09),
            (0.3, -0.13, 0.16, 0.09),
            (0.17, 0, 0.09, 0.04),
            (0.43, 0, 0.09, 0.04),
        )
    )
    return gantry(tool, fixture, (0.0, 0.0, 0.0), (0.3, 0.0, -0.2), (0, 1, 2), (0.012, 0.012, 0.025))


def drawer():
    tool = '<geom type="box" pos="0 0 -.08" size=".025 .025 .08"/>'
    fixture = """<geom type="box" pos=".45 0 .12" size=".2 .25 .12"/><body pos=".3 0 .3"><joint name="drawer" type="slide" axis="1 0 0" range="-.4 0" damping="2"/>
      <geom type="box" size=".18 .2 .03"/><geom type="box" pos="-.2 0 .06" size=".02 .08 .04"/></body>"""
    return gantry(tool, fixture, (-0.25, 0.0, 0.0, 0.0), (-0.35, 0.0, -0.08, -0.35), (3,), (0.04,))


def door():
    tool = '<geom type="sphere" size=".05"/>'
    fixture = """<body pos=".3 -.25 .3"><joint name="door" axis="0 0 1" range="0 1.6" damping=".3"/>
      <geom type="box" pos="0 .25 0" size=".025 .25 .3"/><geom type="sphere" pos="-.06 .43 .12" size=".04"/></body>"""
    return gantry(tool, fixture, (-0.3, 0.1, 0.0, 0.0), (0.1, 0.1, 0.0, 1.3), (3,), (0.12,))


def grasp(stack=False):
    # Three Cartesian axes and eight passive/actuated linkage joints precede the
    # free object. The Robotiq tendon is the fourth control (0=open, 255=closed).
    body = """<body name="stage" pos="0 0 .5">
      <joint name="x" type="slide" axis="1 0 0" range="-.6 .7"/>
      <joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/>
      <joint name="z" type="slide" axis="0 0 1" range="-.32 .2"/>
      <inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>
      <body name="gripper_mount" quat="0 1 0 0"/>
    </body>"""
    extra = """<body name="block" pos="0 0 .03"><freejoint name="object"/>
      <geom name="block" type="box" size=".03 .03 .03" mass=".08" rgba=".85 .45 .15 1" friction="2 .01 .001"/></body>"""
    extra += '<geom type="box" pos=".4 0 .08" size=".07 .07 .08" rgba=".8 .5 .2 1"/>' if stack else ""
    height = 0.19 if stack else 0.03
    robot = (0.0,) * 11
    return Scene(
        body + extra,
        "".join(
            f'<position joint="{joint}" kp="2000" kv="100" ctrlrange="{lo} {hi}"/>'
            for joint, lo, hi in (("x", -0.6, 0.7), ("y", -0.5, 0.5), ("z", -0.32, 0.2))
        ),
        robot + (0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0),
        robot + (0.4, 0.0, height, 1.0, 0.0, 0.0, 0.0),
        (11, 12, 13),
        (0.03, 0.03, 0.02),
        velocity_tolerance=0.05,
        attachments=(MjcfAttachment("robotiq_2f85/2f85.xml", "gripper_mount"),),
        # Require open fingers and a withdrawn tool, not a cube still held at GT.
        success_bounds=((3, -0.01, 0.02), (7, -0.01, 0.02), (2, height - 0.20, 0.21)),
    )


SCENES: dict[str, Callable[[], Scene]] = {
    "CartpoleSwingup": cartpole,
    "DoubleCartpole": lambda: cartpole(True),
    "ReactionWheel": reaction_wheel,
    "BallPlate": ball_plate,
    "PlanarQuadrotor": quadrotor,
    "PayloadTransport": lambda: quadrotor(True),
    "DifferentialParking": parking,
    "HurdleHopper": hopper,
    "ObstacleArm": arm,
    "SpatialArm": lambda: arm(True),
    "PuckPush": pusher,
    "PuckSlide": lambda: pusher("slide"),
    "BoxGate": lambda: pusher("gate"),
    "PegInsertion": peg,
    "DrawerOpen": drawer,
    "DoorOpen": door,
    "PickPlace": grasp,
    "BlockStack": lambda: grasp(True),
}
