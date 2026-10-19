"""Built-in recipes: distinct tasks, not duplicate difficulty aliases."""

import re

from torch_world.envs.registry import EnvSpec, register_env

CATEGORIES = {
    "puzzles": "Planning puzzles",
    "control": "Control",
    "navigation": "Navigation and locomotion",
    "manipulation": "Manipulation",
}
"""Built-in categories, in display order, mapped to their titles."""

_CATEGORY = {
    **dict.fromkeys(("Maze", "KeyDoor", "Sokoban", "IceMaze", "Checkpoints"), "puzzles"),
    **dict.fromkeys(
        ("CartpoleSwingup", "DoubleCartpole", "ReactionWheel", "BallPlate", "NewtonPendulum", "NewtonAcrobot"),
        "control",
    ),
    **dict.fromkeys(
        ("PointMass", "Macaw", "PlanarQuadrotor", "PayloadTransport", "DifferentialParking", "HurdleHopper"),
        "navigation",
    ),
}


OGBENCH = (
    ("pointmaze-medium-navigate-v0", "navigation", "Navigate a point mass through a medium maze"),
    ("antmaze-medium-navigate-v0", "navigation", "Navigate a quadruped ant through a medium maze"),
    ("visual-antmaze-medium-navigate-v0", "navigation", "Navigate the ant through a medium maze from pixels"),
    ("humanoidmaze-medium-navigate-v0", "navigation", "Navigate a humanoid through a medium maze"),
    ("antsoccer-arena-navigate-v0", "navigation", "Dribble a ball to a goal with the ant"),
    ("cube-single-play-v0", "manipulation", "Move one cube to a goal pose with a UR5e arm"),
    ("visual-cube-single-play-v0", "manipulation", "Move one cube to a goal pose from pixels"),
    ("cube-double-play-v0", "manipulation", "Rearrange two cubes into goal poses"),
    ("visual-cube-double-play-v0", "manipulation", "Rearrange two cubes into goal poses from pixels"),
    ("scene-play-v0", "manipulation", "Reach a goal configuration of a cube, drawer, window and buttons"),
    ("visual-scene-play-v0", "manipulation", "Reach a goal scene configuration from pixels"),
    ("puzzle-3x3-play-v0", "manipulation", "Solve a 3x3 lights-out button puzzle with the arm"),
    ("visual-puzzle-3x3-play-v0", "manipulation", "Solve the 3x3 button puzzle from pixels"),
)
"""OGBench tasks in the catalog, one per family and observation type, by dataset name; `OGBench` takes any other."""


ROBOCASA = {
    "atomic_seen": (
        "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug", "NavigateKitchen", "OpenCabinet",
        "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
        "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
    ),
    "composite_seen": (
        "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher", "PackIdenticalLunches", "PreSoakPan",
        "PrepareCoffee", "RinseSinkBasin", "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation",
        "StackBowlsCabinet", "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce",
    ),
    "composite_unseen": (
        "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments", "CuttingToolSelection",
        "GarnishPancake", "GatherTableware", "HeatKebabSandwich", "MakeIceLemonade", "PanTransfer", "PortionHotDogs",
        "RecycleBottlesByType", "SeparateFreezerRack", "WaffleReheat", "WashFruitColander", "WeighIngredients",
    ),
}  # fmt: skip
"""RoboCasa's 50 target tasks by task set, as its dataset registry names them; `RoboCasa` takes any other task."""


DMCONTROL = (
    ("acrobot-swingup", "control", "Swing up an underactuated two-link acrobot"),
    ("ball_in_cup-catch", "control", "Swing a ball on a string into a cup"),
    ("cartpole-balance", "control", "Keep a pole upright on a moving cart"),
    ("cartpole-balance_sparse", "control", "Keep a pole upright on a cart, with a sparse reward"),
    ("cartpole-swingup", "control", "Swing up and balance a pole on a cart"),
    ("cartpole-swingup_sparse", "control", "Swing up and balance a pole on a cart, with a sparse reward"),
    ("cheetah-run", "navigation", "Run forward with a planar cheetah"),
    ("finger-spin", "control", "Spin a hinged body with a two-link finger"),
    ("finger-turn_easy", "control", "Turn a hinged body to a wide target angle"),
    ("finger-turn_hard", "control", "Turn a hinged body to a narrow target angle"),
    ("hopper-hop", "navigation", "Hop forward with a planar one-legged hopper"),
    ("hopper-stand", "navigation", "Stand a planar one-legged hopper upright"),
    ("pendulum-swingup", "control", "Swing up a torque-limited pendulum"),
    ("quadruped-run", "navigation", "Run forward with a quadruped"),
    ("quadruped-walk", "navigation", "Walk forward with a quadruped"),
    ("reacher-easy", "control", "Reach a large target with a two-link arm"),
    ("reacher-hard", "control", "Reach a small target with a two-link arm"),
    ("walker-run", "navigation", "Run forward with a planar biped"),
    ("walker-stand", "navigation", "Stand a planar biped upright"),
    ("walker-walk", "navigation", "Walk forward with a planar biped"),
)
"""DMControl tasks in the catalog, DreamerV3's 20, as ``domain-task``; `DMControl` takes any other suite task."""


METAWORLD = (
    ("reach-v3", "Reach a goal position with the Sawyer arm"),
    ("push-v3", "Push a puck to a goal position"),
    ("pick-place-v3", "Pick up a puck and place it at a goal"),
    ("door-open-v3", "Open a door with a revolving joint"),
    ("drawer-open-v3", "Pull a drawer open"),
    ("drawer-close-v3", "Push a drawer closed"),
    ("button-press-topdown-v3", "Press a button from above"),
    ("peg-insert-side-v3", "Insert a peg sideways into a hole"),
    ("window-open-v3", "Slide a window open"),
    ("window-close-v3", "Slide a window closed"),
)
"""Meta-World tasks in the catalog, the MT10 set; `MetaWorld` takes any of the 50 v3 tasks."""


MANISKILL = (
    ("PickCube-v1", "Pick up a cube and move it to a goal position with a Panda"),
    ("PushCube-v1", "Push a cube to a goal region"),
    ("PullCube-v1", "Pull a cube back to a goal region"),
    ("PokeCube-v1", "Poke a cube to a goal with a peg held in the gripper"),
    ("StackCube-v1", "Stack one cube on another"),
    ("PlaceSphere-v1", "Place a sphere into a shallow bin"),
    ("RollBall-v1", "Roll a ball into a goal region"),
    ("LiftPegUpright-v1", "Lift a lying peg into an upright pose"),
    ("PullCubeTool-v1", "Use an L-shaped tool to pull a cube within reach"),
    ("PushT-v1", "Push a T-shaped block into a goal pose with a stick"),
    ("PegInsertionSide-v1", "Insert a peg sideways into a box's hole"),
    ("PlugCharger-v1", "Plug a charger into a wall receptacle"),
)
"""ManiSkill tabletop tasks in the catalog, none needing asset downloads; `ManiSkill` takes any other task."""


MYOSUITE = (
    ("myoFingerReachRandom-v0", "control", "Reach random targets with a muscle-driven finger"),
    ("myoFingerPoseRandom-v0", "control", "Hold random joint poses with a muscle-driven finger"),
    ("myoElbowPose1D6MRandom-v0", "control", "Hold random elbow angles with six muscles"),
    ("myoHandReachRandom-v0", "control", "Reach random fingertip targets with a 39-muscle hand"),
    ("myoHandPoseRandom-v0", "control", "Hold random hand poses with 39 muscles"),
    ("myoHandKeyTurnRandom-v0", "manipulation", "Turn a key between thumb and index finger"),
    ("myoHandObjHoldRandom-v0", "manipulation", "Hold an object of random shape and size"),
    ("myoHandPenTwirlRandom-v0", "manipulation", "Twirl a pen to random orientations in the hand"),
    ("myoHandReorient8-v0", "manipulation", "Reorient objects of eight shapes in the hand"),
    ("myoLegWalk-v0", "navigation", "Walk forward on a musculoskeletal leg model"),
)
"""MyoSuite tasks in the catalog, from finger to leg; `MyoSuite` takes any other registered environment."""


def _sentence(name):
    words = re.findall(r"[A-Z][a-z]*", name)
    return " ".join([words[0]] + [w.lower() for w in words[1:]])


def _add(id, entry, backend, description, tags, reference, **kwargs):
    category = _CATEGORY.get(id.removesuffix("-v0"), "manipulation")
    register_env(EnvSpec(id, f"torch_world.envs.{entry}", backend, description, tags, kwargs, reference, category))


def register_builtins():
    for name, environment, backend, description, options in (
        ("PointMass", "PointMass", "torch", "Move a point mass to a target in the unit square", {}),
        ("Macaw", "Macaw", "torch", "Fly the animated macaw to a target position", {}),
        ("PushT", "PushT", "mujoco", "Push a T-shaped block to a fixed target pose", {}),
        ("RobotCube", "RobotCube", "mujoco", "UR10e and Robotiq 2F-85 cube pick-and-place", {"robot": "ur10e"}),
        ("RobotCubeSO101", "RobotCube", "mujoco", "SO-101 cube pick-and-place", {"robot": "so101"}),
    ):
        _add(
            f"{name}-v0",
            "legacy:LegacyTask",
            backend,
            description,
            ("legacy", "goal"),
            "https://github.com/latent-horizon/pytorch-world/blob/main/docs/guides/environments.md",
            environment=environment,
            options=options,
        )
    puzzles = {
        "Maze": ("maze", "Route around walls and dead ends", ("navigation", "planning")),
        "KeyDoor": ("key_door", "Collect a key before crossing a locked door", ("planning", "prerequisites")),
        "Sokoban": ("sokoban", "Push a box to a target without trapping it", ("planning", "irreversible")),
        "IceMaze": ("ice", "Slide until blocked; plan stopping positions", ("planning", "momentum")),
        "Checkpoints": (
            "checkpoints",
            "Visit two checkpoints in order before reaching the exit",
            ("planning", "sequence"),
        ),
    }
    for name, (task, description, tags) in puzzles.items():
        _add(
            f"{name}-v0",
            "puzzles:PlanningPuzzle",
            "torch",
            description,
            tags,
            "https://minigrid.farama.org/",
            task=task,
            max_episode_steps=100,
        )
    scenes = {
        "CartpoleSwingup": "Torque-limited cart-pole swing-up",
        "DoubleCartpole": "Underactuated double-pole stabilization",
        "ReactionWheel": "Reaction-wheel pendulum balance",
        "BallPlate": "Tilt a plate to position a constrained ball",
        "PlanarQuadrotor": "Differential-thrust planar flight",
        "PayloadTransport": "Transport a suspended payload without swing",
        "DifferentialParking": "Park a differential-drive robot between barriers",
        "HurdleHopper": "Hop a single articulated leg across a hurdle",
        "ObstacleArm": "Two-link arm positioning around an obstacle",
        "SpatialArm": "Three-axis spatial arm positioning",
        "PuckPush": "Contact-only planar object pushing",
        "PuckSlide": "Slide an object beyond the tool workspace by releasing momentum",
        "BoxGate": "Push a box through a narrow gate",
        "PegInsertion": "Insert a Cartesian peg into a clearance fixture",
        "DrawerOpen": "Contact a handle and pull a sliding drawer",
        "DoorOpen": "Push a hinged door open",
        "PickPlace": "Grasp and relocate a free cube with a Robotiq 2F-85",
        "BlockStack": "Stack a free cube on a raised block with a Robotiq 2F-85",
    }
    manipulation = {
        "ObstacleArm",
        "SpatialArm",
        "PuckPush",
        "PuckSlide",
        "BoxGate",
        "PegInsertion",
        "DrawerOpen",
        "DoorOpen",
        "PickPlace",
        "BlockStack",
    }
    for name, description in scenes.items():
        _add(
            f"{name}-v0",
            "physics_tasks:MujocoTask",
            "mujoco",
            description,
            ("manipulation" if name in manipulation else "control", "planning", "goal"),
            "https://mujoco.readthedocs.io/",
            task=name,
        )
    for name, links, actuated in (("Pendulum", 1, True), ("Acrobot", 2, False)):
        _add(
            f"Newton{name}-v0",
            "newton:NewtonPendulum",
            "newton",
            f"{'Torque-limited swing-up' if links == 1 else 'Underactuated two-link swing-up'} with Newton physics",
            ("control", "planning", "goal"),
            "https://newton-physics.github.io/newton/",
            links=links,
            fully_actuated=actuated,
            max_episode_steps=300,
        )
    for name, category, description in OGBENCH:
        register_env(
            EnvSpec(
                f"OGBench/{name}",
                "torch_world.envs.ogbench:OGBench",
                "ogbench",
                description,
                ("ogbench", "planning", "goal"),
                {"name": name},
                "https://github.com/seohongpark/ogbench",
                category,
            )
        )
    for task_set, tasks in ROBOCASA.items():
        kind, seen = task_set.split("_")
        for task in tasks:
            register_env(
                EnvSpec(
                    f"RoboCasa/{task}",
                    "torch_world.envs.robocasa:RoboCasa",
                    "robocasa",
                    f"{_sentence(task)} ({kind}, {seen} in pretraining)",
                    ("robocasa", kind, seen),
                    {"task": task},
                    "https://robocasa.ai",
                    "manipulation",
                )
            )
    suites = (
        ("DMControl", "dmcontrol", "https://github.com/google-deepmind/dm_control", DMCONTROL),
        ("MetaWorld", "metaworld", "https://github.com/Farama-Foundation/Metaworld", METAWORLD),
        ("ManiSkill", "maniskill", "https://github.com/haosulab/ManiSkill", MANISKILL),
        ("MyoSuite", "myosuite", "https://github.com/MyoHub/myosuite", MYOSUITE),
    )
    for prefix, backend, reference, tasks in suites:
        for name, *category, description in tasks:
            register_env(
                EnvSpec(
                    f"{prefix}/{name}",
                    f"torch_world.envs.{backend}:{prefix}",
                    backend,
                    description,
                    (backend,),
                    {"name": name},
                    reference,
                    category[0] if category else "manipulation",
                )
            )
    for name, task, description in (
        ("FrankaObstacleReach", "reach", "Franka reaching around a tabletop barrier"),
        ("FrankaTablePush", "push", "Franka contact pushing on a tabletop"),
        ("FrankaCubeLift", "lift", "Franka grasping and lifting a free cube"),
    ):
        _add(
            f"{name}-v0",
            "isaaclab:IsaacLabEnv",
            "isaaclab",
            description,
            ("manipulation", "planning"),
            "https://isaac-sim.github.io/IsaacLab/",
            task=task,
        )
