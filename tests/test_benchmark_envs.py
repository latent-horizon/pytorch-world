"""Contract, dynamics and integration tests for the expanded task catalogue."""

import importlib.util
import subprocess
import sys
from collections import deque

import pytest
import torch

from torch_world.envs import EnvSpec, get_spec, list_envs, make, register_env, register_task
from torch_world.evaluation import benchmark


def solve(env):
    queue, seen = deque([(env.initial, [])]), {env.initial}
    while queue:
        state, path = queue.popleft()
        if env.success(torch.tensor([state]), env.goal).item():
            return path
        for a in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = env.advance(state, a)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [a]))
    raise AssertionError("Unsolvable built-in level")


def test_lazy_catalog_and_distinct_recipes():
    assert len(list_envs()) >= 25
    assert len(list_envs(backend="mujoco")) == 21
    assert len({(s.entry_point, repr(dict(s.kwargs))) for s in list_envs()}) == len(list_envs())
    code = "import sys; import torch_world.envs; assert not any(m in sys.modules for m in ['mujoco','gymnasium','newton','warp','isaaclab'])"
    subprocess.run([sys.executable, "-c", code], check=True)
    with pytest.raises(KeyError, match="Unknown environment"):
        make("missing")


def test_contributor_registration_and_task_override():
    id = "TestContributorMaze-v0"
    register_env(
        EnvSpec(id, "torch_world.envs.puzzles:PlanningPuzzle", "torch", "Contributor map", kwargs={"task": "maze"})
    )
    with pytest.raises(ValueError, match="already registered"):
        register_env(get_spec(id))
    register_task("TestContributorShort-v0", id, description="Short budget", max_episode_steps=2)
    env = make("TestContributorShort-v0", size=8)
    env.reset()
    env.step(torch.zeros(1, 2))
    env.step(torch.zeros(1, 2))
    assert env.transition.truncated.all()
    assert "max_episode_steps" not in get_spec(id).kwargs


@pytest.mark.parametrize("id", ["Maze-v0", "KeyDoor-v0", "Sokoban-v0", "IceMaze-v0", "Checkpoints-v0"])
def test_puzzles_are_solvable_and_terminal_states_absorb(id):
    env = make(id, size=18)
    path = solve(env)
    obs = env.reset()
    assert obs.shape == (1, 3, 18, 18) and 0 <= obs.min() <= obs.max() <= 1
    for action in path:
        env.step(torch.tensor([action]))
    assert env.transition.success.all() and env.transition.terminated.all()
    final = env.state.clone()
    assert torch.equal(env.render(final), env.render())
    env.step(torch.tensor([[1.0, 1.0]]))
    assert torch.equal(env.state, final) and env.transition.reward.item() == 0


def test_prerequisites_box_deadlock_and_ice_dynamics():
    key = make("KeyDoor-v0")
    # Crossing the doorway is impossible before acquiring the key.
    assert key.advance((3, 2, 0, 0, -1, -1), (1, 0))[:2] == (3, 2)
    assert key.advance((3, 2, 1, 0, -1, -1), (1, 0))[:2] == (4, 2)
    assert key.advance((2, 3, 0, 0, -1, -1), (0, 1))[2] == 1
    box = make("Sokoban-v0")
    assert box.advance((2, 1, 0, 0, 1, 1), (-1, 0)) == (2, 1, 0, 0, 1, 1)
    ice = make("IceMaze-v0")
    assert ice.advance(ice.initial, (1, 0))[:2] == (6, 1)
    checkpoints = make("Checkpoints-v0")
    assert checkpoints.advance((6, 3, 0, 0, -1, -1), (0, 1))[3] == 0
    assert checkpoints.advance((6, 3, 0, 1, -1, -1), (0, 1))[3] == 2


def test_collect_alignment_masks_and_seeds():
    env = make("Maze-v0", batch_size=2, size=8, max_episode_steps=2)
    first = env.collect(4, generator=torch.Generator().manual_seed(5))
    second = env.collect(4, generator=torch.Generator().manual_seed(5))
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert first["valid"].tolist() == [[True, True, False, False]] * 2
    assert torch.equal(first["next_observations"][:, 0], first["observations"][:, 1])
    assert first["actions"][:, 2:].count_nonzero() == 0
    assert env.collect(1)["actions"].shape == (2, 1, 2)
    with pytest.raises(ValueError):
        env.collect(0)
    with pytest.raises(ValueError):
        env.step(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        env.step(torch.full((2, 2), float("nan")))


def test_benchmark_cutoff_and_success():
    env = make("Maze-v0", max_episode_steps=3)
    result = benchmark(env, {"wait": lambda obs: torch.zeros(len(obs), 2)}, seeds=[1, 2], steps=2)["wait"]
    assert result["lengths"].tolist() == [2, 2]
    assert result["truncated"].all() and not result["success"].any()
    assert torch.allclose(result["returns"], torch.tensor([-0.02, -0.02]))


HAS_GYM = importlib.util.find_spec("mujoco") is not None


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
@pytest.mark.parametrize("spec", list_envs(backend="mujoco"), ids=lambda spec: spec.id)
def test_mujoco_seeded_episode_and_boundaries(spec):
    with make(spec.id, batch_size=2, max_episode_steps=2, observation="state") as env:
        first = env.reset(seed=17)
        second = env.reset(seed=17)
        assert torch.equal(first, second)
        assert first.shape == (2, env.state_dim)
        obs = env.step(torch.full((2, env.action_dim), 10.0))
        assert torch.isfinite(obs).all() and torch.isfinite(env.transition.reward).all()
        env.step(torch.zeros(2, env.action_dim))
        assert env.transition.done.all()
        final = env.state.clone()
        env.step(torch.ones(2, env.action_dim))
        assert torch.equal(env.state, final) and not env.transition.reward.any()


@pytest.mark.skipif(importlib.util.find_spec("newton") is None, reason="requires Newton extra")
@pytest.mark.parametrize("id", ["NewtonPendulum-v0", "NewtonAcrobot-v0"])
def test_newton_real_dynamics_seed_and_goal(id, tmp_path):
    import warp

    warp.config.kernel_cache_dir = str(tmp_path / "warp")
    with make(id, batch_size=2, max_episode_steps=2, size=16) as env:
        obs = env.reset(seed=2)
        state = env.state.clone()
        assert torch.equal(env.reset(seed=2), obs)
        env.step(torch.ones(2, env.action_dim))
        assert not torch.equal(state, env.state) and torch.isfinite(env.state).all()
        assert env.success(env.goal, env.goal).all()
        assert not env.success(torch.zeros_like(env.goal), env.goal).any()
        env.step(torch.zeros(2, env.action_dim))
        assert env.transition.truncated.all()


def test_lewm_backward_and_planning_on_new_environment():
    from torch_world.models import LeWM
    from torch_world.planning import CEM

    env = make("KeyDoor-v0", batch_size=4, size=8)
    data = env.collect(3, generator=torch.Generator().manual_seed(2))
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
    model = LeWM(
        encoder, 2, embed_dim=16, history=1, depth=1, num_heads=2, head_dim=8, proj_dim=32, mlp_dim=32, dropout=0
    )
    output = model(data["pixels"][:, :2], data["actions"][:, :2])
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        z = model.encode(data["pixels"][:, :1])
        goal = model.encode(env.render(env.goal)[:, None])[:, 0]
        action = CEM(2, 2, num_samples=8, num_elites=2, iterations=1, bounds=(-1, 1))(model, z, goal)
    assert action.shape == (4, 2, 2) and torch.isfinite(env.step(action[:, 0])).all()


def test_isaac_autoreset_replaces_terminal_observation_and_freezes_lanes():
    from torch_world.envs.benchmark import BenchmarkEnv
    from torch_world.envs.isaaclab import IsaacLabEnv

    class Simulator:
        def step(self, action):
            return (
                {"policy": torch.tensor([[100.0], [2.0]])},
                torch.ones(2),
                torch.tensor([True, False]),
                torch.zeros(2, dtype=torch.bool),
                {"final_obs": {"policy": torch.tensor([[9.0], [2.0]])}},
            )

    env = object.__new__(IsaacLabEnv)
    BenchmarkEnv.__init__(env, batch_size=2, max_episode_steps=2)
    env.env, env.action_dim, env.observation_key = Simulator(), 1, "policy"
    env.success_fn = None
    env._obs = torch.zeros(2, 1)
    obs = env.step(torch.zeros(2, 1))
    assert obs[:, 0].tolist() == [9, 2]
    assert env.transition.success is None  # Custom tasks must supply their own success predicate.
    assert env.transition.terminated.tolist() == [True, False]
    obs = env.step(torch.zeros(2, 1))
    assert obs[:, 0].tolist() == [9, 2]
    assert env.transition.reward.tolist() == [0, 1]
    assert env.transition.done.all()


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
@pytest.mark.parametrize("id", [s.id for s in list_envs(backend="mujoco")])
def test_mujoco_rgb_observations(id):
    with make(id, observation="rgb", size=16) as env:
        first = env.reset(seed=4)
        second = env.step(torch.zeros(1, env.action_dim))
        assert first.shape == second.shape == (1, 3, 16, 16)
        assert torch.isfinite(second).all() and 0 <= second.min() <= second.max() <= 1


@pytest.mark.parametrize(
    "id", [s.id for s in list_envs(backend="torch") if not s.id.startswith("Test") and "legacy" not in s.tags]
)
def test_vectorized_puzzles_match_scalar_rules(id):
    from torch_world.benchmarks.experiment import reachable_states

    env = make(id, size=8)
    states = reachable_states(env)
    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        actions = torch.tensor(direction).expand(len(states), -1)
        expected = torch.tensor([env.advance(s, direction) for s in states.tolist()])
        actual = env._advance_batch(states, actions, torch.ones(len(states), dtype=torch.bool))
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
@pytest.mark.parametrize(
    "spec", [s for s in list_envs(backend="mujoco") if "legacy" not in s.tags], ids=lambda spec: spec.id
)
def test_physics_reference_is_nonpenetrating_and_render_has_no_side_effect(spec):
    with make(spec.id, observation="state") as env:
        env.reset(seed=7)
        initial = env.state.clone()
        reference = env.render_target()
        target = env.goal.clone()
        assert torch.equal(reference[:, env._indices], target[:, env._indices])
        assert torch.equal(initial, env.state)
        assert env.success(target, env.goal).all()
        assert not env.success(initial, env.goal).any()
        env.reset(target)
        assert min((c.dist for c in env.data[0].contact), default=0) >= -1e-4


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
def test_pushing_transfers_contact_force_to_passive_object():
    with make("PuckPush-v0", observation="state") as env:
        env.reset(seed=7)
        start = env.state[0, 2].item()
        # Move the actuated tool across the object; object joints have no actuators.
        control = torch.tensor([[(0.6 - env._mid[0]) / env._scale[0], 0.0]])
        for _ in range(100):
            env.step(control)
        assert env.state[0, 2] > start + 0.2
        assert env.model.nu == 2


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
@pytest.mark.parametrize("name", ["PickPlace-v0", "BlockStack-v0"])
def test_robotiq_grasps_lifts_and_releases_with_four_controls(name, tmp_path, monkeypatch):
    # Packaged meshes must resolve independently of the process working directory.
    monkeypatch.chdir(tmp_path)
    with make(name, observation="state", batch_size=3) as env:
        env.reset(seed=7)
        assert env.action_dim == 4
        data, model = env.data[0], env.model
        aperture = abs(data.geom("right_pad1").xpos[0] - data.geom("left_pad1").xpos[0]) - 0.008
        assert aperture == pytest.approx(0.085, abs=0.001)
        assert model.actuator_trntype[-1] == env.mj.mjtTrn.mjTRN_TENDON
        assert model.actuator_ctrlrange[-1].tolist() == [0, 255]
        goal_held = env.goal.clone()
        goal_held[:, [3, 7]] = 0.27
        assert not env.success(goal_held, env.goal).any()
        goal_near_tool = env.goal.clone()
        goal_near_tool[:, 2] = -0.31
        assert not env.success(goal_near_tool, env.goal).any()

        policy = env.reference_policy()
        for step in range(policy.max_steps):
            env.step(policy(None))
            if step == 89:
                block = model.geom("block").id
                pads = {model.geom(f"{side}_pad{pad}").id for side in ("left", "right") for pad in (1, 2)}
                assert any(block in c.geom and pads.intersection(c.geom) for c in data.contact)
            if step == 124:
                assert (env.state[:, 13] > 0.30).all()
                assert not env.transition.success.any()
        assert env.transition.success.all()
        assert torch.allclose(env.state[:, 11:14], env.goal[:, 11:14], atol=0.002)
        assert (env.state[:, [3, 7]].abs() < 0.01).all()


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
def test_contributors_can_register_a_scene_without_editing_dispatch():
    from torch_world.envs import Scene

    scene = Scene(
        bodies='<body pos="0 0 1"><joint name="slide" type="slide" axis="1 0 0"/><geom type="sphere" size=".1"/></body>',
        actuators='<motor joint="slide" gear="1"/>',
        initial=(0.0,),
        target=(0.5,),
        indices=(0,),
        tolerance=(0.05,),
    )
    register_env(
        EnvSpec(
            "ContributorPhysics-v0",
            "torch_world.envs.physics_tasks:MujocoTask",
            "mujoco",
            "A contributor-owned scene",
            kwargs={"scene": scene},
        )
    )
    with make("ContributorPhysics-v0", observation="state") as env:
        env.reset(seed=7)
        env.step(torch.ones(1, 1))
        assert env.state[0, 0] > 0


def test_builtins_are_grouped_into_categories():
    from torch_world.envs import CATEGORIES

    builtins = [s for s in list_envs() if not s.id.startswith(("Test", "Contributor"))]
    assert {s.category for s in builtins} == set(CATEGORIES)
    assert sum(len(list_envs(category=c)) for c in CATEGORIES) == len(builtins)
    assert {s.id for s in list_envs(category="puzzles")} == {
        "Maze-v0",
        "KeyDoor-v0",
        "Sokoban-v0",
        "IceMaze-v0",
        "Checkpoints-v0",
    }


def test_named_classes_cover_catalogue_without_loading_simulators():
    from torch_world import envs

    for spec in list_envs():
        if spec.id.startswith(("Test", "Contributor")) or "legacy" in spec.tags:
            continue
        cls = getattr(envs, spec.id.removesuffix("-v0"))
        assert issubclass(cls, envs.BenchmarkEnv)
        assert get_spec(cls) == spec
        assert callable(cls.show_env) and callable(cls.show_success)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from torch_world.envs import KeyDoor, PegInsertion, NewtonPendulum, FrankaCubeLift; "
                'assert not any(m in sys.modules for m in ["mujoco", "newton", "warp", "isaaclab"])'
            ),
        ],
        check=True,
    )


def test_named_constructor_defaults_and_overrides_match_registry():
    from torch_world.envs import KeyDoor

    with KeyDoor(batch_size=2, size=8, max_episode_steps=3) as direct, make(KeyDoor, batch_size=2, size=8) as factory:
        assert type(direct) is type(factory) is KeyDoor
        assert direct.task == "key_door" and direct.max_episode_steps == 3
        assert torch.equal(direct.reset(seed=7), factory.reset(seed=7))
        assert direct.spec.id == "KeyDoor-v0"
        assert get_spec(direct) == get_spec(KeyDoor)


def test_stochastic_policy_evaluation_is_seed_matched():
    from torch_world.envs import Maze

    class RandomPolicy:
        def reset(self):
            self.actions = []

        def __call__(self, obs):
            action = torch.rand(len(obs), 2) * 2 - 1
            self.actions.append(action.clone())
            return action

    first, second = RandomPolicy(), RandomPolicy()
    with Maze(size=8) as world:
        benchmark(world, {"first": first, "second": second}, seeds=[21], steps=10)
    assert torch.equal(torch.stack(first.actions), torch.stack(second.actions))


def test_sokoban_custom_goal_next_to_wall_has_valid_gt():
    from torch_world.envs import Sokoban

    with Sokoban(level=("######", "#G.BA#", "#....#", "######"), size=8) as world:
        world.reset(world.goal)
        assert world.success(world.state, world.goal).all()
        assert world.state[0, :2].tolist() == [2, 1]


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
def test_custom_mujoco_scene_maps_position_to_velocity_after_free_joint():
    from torch_world.envs import Scene
    from torch_world.envs.physics_tasks import MujocoTask

    scene = Scene(
        bodies="""<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
        <body pos="2 0 1"><joint name="hinge"/><geom type="box" size=".1 .2 .1"/></body>""",
        actuators='<motor joint="hinge" gear="1"/>',
        initial=(0, 0, 1, 1, 0, 0, 0, 0),
        target=(0.2, 0, 1, 1, 0, 0, 0, 0.1),
        indices=(0, 7),
        tolerance=(0.05, 0.05),
    )
    with MujocoTask(scene=scene, observation="state") as world:
        assert world.model.nq == 8 and world.model.nv == 7
        assert world.success(world.goal, world.goal).all()
        moving = world.goal.clone()
        moving[:, 14] = 1.0  # hinge velocity: qvel[6], not qpos[7]
        assert not world.success(moving, world.goal).any()
        target = world.render_target()
        assert target.isfinite().nonzero()[:, 1].tolist() == [0, 7, 8, 14]


def test_render_quality_defaults_and_independent_model_resolution():
    from torch_world.envs import QUALITY_PRESETS, Maze

    with Maze(size=24) as world:
        assert world.quality == "high"
        assert world.reset().shape == (1, 3, 24, 24)
        assert world.render_frame().shape == world.render_goal_frame().shape == (1, 3, 512, 512)
    states = []
    for quality, preset in QUALITY_PRESETS.items():
        with Maze(quality=quality) as world:
            assert world.reset().shape[-1] == preset.observation_size
            world.step(torch.tensor([[1.0, 0.0]]))
            states.append(world.state.clone())
            assert world.render_frame().shape[-1] == preset.render_size
    assert all(torch.equal(states[0], state) for state in states)
    with pytest.raises(ValueError, match="quality"):
        Maze(quality="unknown")
    with pytest.raises(ValueError, match="render_size"):
        Maze(render_size=0)


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
def test_mujoco_camera_frames_independent_of_state_observations():
    from torch_world.envs import BallPlate

    with BallPlate(observation="state", render_size=64, camera="task") as world:
        observation = world.reset(seed=7)
        state = world.state.clone()
        frame, gt = world.render_frame(), world.render_goal_frame()
        assert observation.ndim == 2 and frame.shape == gt.shape == (1, 3, 64, 64)
        assert not torch.equal(frame, gt)
        assert torch.equal(world.state, state)
        assert world.model.vis.quality.offsamples == 4
    with pytest.raises(ValueError, match="camera"):
        BallPlate(camera="unknown")


def test_isaac_terminal_camera_and_state_are_preserved_separately():
    from torch_world.envs.benchmark import BenchmarkEnv
    from torch_world.envs.isaaclab import IsaacLabEnv

    class Simulator:
        def step(self, action):
            return (
                {"policy": torch.ones(1, 24), "camera": torch.full((1, 8, 8, 3), 200, dtype=torch.uint8)},
                torch.ones(1),
                torch.ones(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {
                    "final_obs": {
                        "policy": torch.zeros(1, 24),
                        "camera": torch.full((1, 8, 8, 3), 100, dtype=torch.uint8),
                    }
                },
            )

    env = object.__new__(IsaacLabEnv)
    BenchmarkEnv.__init__(env, size=4, render_size=8)
    env.env, env.action_dim, env.observation_key = Simulator(), 1, "camera"
    env._builtin_scene, env.success_fn = True, None
    env._obs, env._frame = torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    env.state = torch.ones(1, 24)
    env.step(torch.zeros(1, 1))
    assert env.render_frame().shape == (1, 3, 8, 8)
    assert (env.render_frame() == 100).all()
    torch.testing.assert_close(env.render(), torch.full((1, 3, 4, 4), 100 / 255))
    assert env.state.eq(0).all() and env.transition.success.all()
    env.step(torch.zeros(1, 1))
    assert (env.render_frame() == 100).all() and env.state.eq(0).all()


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
def test_quality_presets_do_not_change_mujoco_dynamics():
    from torch_world.envs import BallPlate

    states = []
    for quality in ("low", "medium", "high"):
        with BallPlate(quality=quality, observation="state") as world:
            world.reset(seed=7)
            for action in ((0.2, -0.4), (-0.3, 0.8), (0.0, 0.0)):
                world.step(torch.tensor([action]))
            states.append(world.state.clone())
    assert all(torch.equal(states[0], state) for state in states)


@pytest.mark.skipif(not HAS_GYM, reason="requires benchmark extra")
@pytest.mark.parametrize("position", [(4.0, 0.0), (0.8, 2.65)])
def test_free_flight_presentation_keeps_robot_visible_without_moving_model_camera(position):
    from torch_world.envs import PlanarQuadrotor

    with PlanarQuadrotor(size=32, render_size=128) as world:
        state = world.reset(seed=7)
        world_state = world.state.clone()
        far = world.state.clone()
        far[:, :2] = torch.tensor(position)
        before = world.render(far)
        frame = world.render_frame(far)
        blue = (frame[:, 2] > frame[:, 0] * 1.3) & (frame[:, 2] > frame[:, 1] * 1.15)
        assert blue.sum() > 10
        assert torch.equal(world.render(far), before)
        assert torch.equal(world.state, world_state)
        assert torch.equal(world.render(), state)


def test_mujoco_references_reach_success():
    """Reference assets must remain executable under the supported engine version."""
    import json
    from importlib.resources import files

    from torch_world.envs import list_envs, make

    mujoco = pytest.importorskip("mujoco")
    if mujoco.__version__ != "3.6.0":
        pytest.skip("Stored reference trajectories require MuJoCo 3.6.0 (benchmark extra)")
    paths = files("torch_world.envs").joinpath("reference_actions")
    for spec in list_envs(backend="mujoco"):
        path = paths.joinpath(f"{spec.id}.json")
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        with make(
            spec.id,
            observation="state",
            max_episode_steps=max(300, record.get("max_steps", len(record.get("actions", [])))),
        ) as world:
            world.reset(seed=record["seed"])
            policy = world.reference_policy()
            for _ in range(policy.max_steps):
                world.step(policy(None))
                if world.transition.success[0]:
                    break
            assert world.transition.success[0], spec.id
