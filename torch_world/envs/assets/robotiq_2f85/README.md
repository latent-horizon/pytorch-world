# Robotiq 2F-85 model

Vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/robotiq_2f85), commit `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`.

The MJCF and meshes are unchanged. They describe the Robotiq 85 mm adaptive gripper, derived from ROS-Industrial's public URDF, under the included BSD-2-Clause [LICENSE](LICENSE). This is a simulation approximation, not a manufacturer-certified digital twin.

Our task assembly mounts the model downward on a Cartesian stage, preserves its two four-bar linkages, tendon drive, contact exclusions, pad friction, inertias and force limits, and isolates its defaults from the enclosing scene. The surrounding tasks, controls, objectives and demonstrations are implemented in torch-world.
