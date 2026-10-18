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
