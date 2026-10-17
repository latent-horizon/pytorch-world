"""Photo-informed setup and explicit simulation evaluation gates."""

import math

from torch_world.sim2real.config import BoardAppearance, ChessTaskConfig, SO101Config, WristView


def reference_setup(*, size=140, seed=7):
    """250 mm board supplied by the owner; remaining geometry is unverified.

    A board-landmark fit estimates a 22.65 mm border and 25.59 mm squares.
    The supplied 1 mm thickness describes the surface here; the pictured case
    is estimated at 15 mm. Pose, lens and piece dimensions still need measuring.
    The base-frame placement is one joint-feasible estimate, not uniquely
    recoverable from a single image. Fit evidence is in the notebook's
    ``assets/rest-view-fit.json``. This preset must never set ``measured=True``
    or create verified calibration.
    """
    # Menagerie's camera_mount expressed in gripperframe coordinates. The old
    # centre-line camera was behind the wrong face of the jaws. A small downward
    # optical tilt frames the jaw tips like the photograph; this is an estimate,
    # not a measured transform for the owner's external camera.
    config = SO101Config(
        size=size,
        seed=seed,
        approach_pitch=1.2,
        workspace_low=(0.12, -0.20, 0.006),
        workspace_high=(0.40, 0.20, 0.21),
        wrist_view=WristView(
            eye=(-0.045027, 0.054578, -0.014960),
            target=(0.03315960, -0.00856035, -0.01432363),
            up=(0.54775127, 0.83660463, -0.00782521),
            fovy=54.6595806,
            aspect=16 / 9,
        ),
    )
    task = ChessTaskConfig(
        square_size=(0.25 - 2 * 0.0226530304) / 8,
        board_origin=(0.304881946, -0.040383449, 0.016),
        appearance=BoardAppearance(border=0.0226530304, thickness=0.015, surface=0.001),
        hover_height=0.080,
    )
    return config, task


def render_reference_pose(env, *, size=512):
    """Estimated near-closed resting pose for photo comparison, never a command.

    For ``reference_setup()``: the photograph has no joint telemetry. These
    estimated angles reproduce its resting view rather than the working approach.
    Restore all integration state so a preview cannot change an episode.
    """
    import numpy as np

    state_kind = env.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(env.mj.mj_stateSize(env.model, state_kind))
    env.mj.mj_getState(env.model, env.data, state, state_kind)
    try:
        env.data.qpos[env.controller.indices] = [-0.15523063, -1.65340481, 1.45587451, 0.73718530, -1.59054507]
        env.data.qpos[env.grip_joint] = env.model.joint("gripper").range[0]
        env.mj.mj_forward(env.model, env.data)
        return env.render_view(size=size, native=True)
    finally:
        env.mj.mj_setState(env.model, env.data, state, state_kind)
        env.mj.mj_forward(env.model, env.data)


def simulation_gate(result, *, minimum_episodes=24, nominal_success=0.9, randomized_success=0.8):
    """A research progression criterion, never authorization to actuate a robot.

    Only held-out learned-controller outcomes count. Scripted success, lower
    training loss and truncated smoke runs cannot pass this gate.
    """
    reasons = []
    control_dt = result.get("config", {}).get("control_dt", float("nan"))
    if not math.isfinite(control_dt) or control_dt <= 0:
        reasons.append("A finite positive control interval is required.")
    if result.get("goal_protocol") == "scripted_keyframes":
        reasons.append("Scripted intermediate-goal assistance is a diagnostic, not autonomous final-goal validation.")
    if result.get("smoke", True):
        reasons.append("Smoke runs cannot establish manipulation performance.")
    for domain, required in (("nominal", nominal_success), ("randomized", randomized_success)):
        records = result.get("evaluation", {}).get(domain, {}).get("learned", {}).get("episodes", [])
        if len(records) < minimum_episodes:
            reasons.append(f"{domain}: need at least {minimum_episodes} held-out learned-policy episodes.")
        rate = sum(bool(r.get("success")) for r in records) / max(1, len(records))
        if rate < required:
            reasons.append(f"{domain}: success {rate:.1%} is below {required:.0%}.")
        if any(r.get("failure") for r in records):
            reasons.append(f"{domain}: controller/planning failures remain.")
        latencies = [r.get("max_planning_seconds", float("inf")) for r in records]
        if any(not math.isfinite(t) or t < 0 or t > control_dt for t in latencies):
            reasons.append(f"{domain}: planning latency is invalid or exceeds the control interval.")
    return {"passed": not reasons, "reasons": reasons, "physical_trial_enabled": False}
