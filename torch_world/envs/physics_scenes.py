"""Owned robotic-arm scene used by the LeWAM study."""

from dataclasses import dataclass


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
        return xml

    def assets(self) -> dict[str, bytes]:
        return {}


def motor(joint: str, gear: float) -> str:
    return f'<motor joint="{joint}" gear="{gear}"/>'


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


SCENES = {"ObstacleArm": arm}
