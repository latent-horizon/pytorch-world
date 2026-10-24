# API reference

Generated from the docstrings of `torch_world`.

| module | contents |
|---|---|
| [`torch_world.nn`](nn.md) | ready encoders `CLSEncoder`, `PixelPatches`, `PatchTokens` (DINOv2) and `VideoTokens` (V-JEPA 2), `TextEncoder` and `WordTokenizer`; `VisionTransformer` (image and video), `MaskedPredictor`, `Transformer`, `Block`, `AdaLNBlock`, `Attention`, `MLP`, sin-cos and rotary positions |
| [`torch_world.masks`](masks.md) | `MultiBlockMask` (2D), `MultiBlock3DMask` (video), `frame_causal_mask`, `apply_masks` |
| [`torch_world.losses`](losses.md) | `SIGReg`, `info_nce` |
| [`torch_world.models`](models.md) | the `JEPA` and `WorldModel` base classes and the models |
| [`torch_world.planning`](planning.md) | `CEM`, for any `WorldModel`, scoring the final or every imagined step; `MPPI`, TD-MPC2's planner for reward |
| [`torch_world.envs`](envs.md) | `RobotCube` (reach or pick-and-place) and `PushT` (MuJoCo), the pixel worlds `Macaw`, `PointMass` and `TwoRooms`; `run_policy` |
| [`torch_world.datasets`](datasets.md) | `TrajectoryDataset`, `MedMNIST`, `MedicalVQA` |
| [`torch_world.evaluation`](evaluation.md) | protocols that run a model and score it: `goal_reaching` for planners (success rates, videos) and `goal_policy` to plan toward goal images, `vqa` and `compare` for question answering |
| [`torch_world.metrics`](metrics.md) | the metrics of the papers, on plain tensors: linear, k-NN and attentive probes; RankMe and LiDAR; rollout error, SSIM, PSNR and Chamfer distance; recall@k and top-k accuracy; for robotics, success rates with confidence intervals, steps to success, progress, SPL and smoothness |
| [`torch_world.utils`](utils.md) | `fit`, a training loop; `ema_update`, `linear_schedule`, `recompute_batchnorm` |
| [`torch_world.plots`](plots.md) | matplotlib figures for notebooks (optional dependency) |
