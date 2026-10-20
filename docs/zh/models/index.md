# 模型

模型按**家族**组织。同一家族的模型采用相同的学习方式和基类，因此可以在相同的数据、环境和规划器上互相替换和比较。目前的家族有 **JEPA**、**DINO** 和 **TD-MPC**；后续还会加入更多家族和模型。

<div class="grid cards tw-models tw-models--families">

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    预测目标的*表征*，而非其像素；引入动作后即成为隐空间世界模型。

-   :material-school-outline:{ .lg } **[DINO](dino/index.md)**

    在可学习原型上进行自蒸馏，以及基于其冻结特征的世界模型。

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    无解码器的隐空间动力学，通过 TD 学习训练，并以回报为目标进行规划。

</div>

## 模型一览 {#every-model-at-a-glance}

| 家族 | 分组 | 模型 | 输入 | 基类 | notebook |
|---|---|---|---|---|---|
| JEPA | 动作 | [LeWM](jepa/lewm.md) | 帧 + 动作 | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | 动作 | [LeWAM](jepa/lewam.md) | 帧 + 动作（+ 目标图像） | `WorldModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | 动作 | [RoboJEPA](jepa/robojepa.md) | 多视角帧 + 动作 + 状态 | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | 视觉 | [I-JEPA](jepa/ijepa.md) | 图像 | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | 视觉 | [LeJEPA](jepa/lejepa.md) | 图像 | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | 视觉 | [V-JEPA](jepa/vjepa.md) | 视频 | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | 视觉 | [LeVJEPA](jepa/levjepa.md) | 视频 | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | 视觉-语言 | [VL-JEPA](jepa/vljepa.md) | 图像或视频 + 文本 | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| DINO | 动作 | [DINO-WM](dino/dinowm.md) | 帧 + 动作 + 本体感知 | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| DINO | 视觉 | [DINOv2](dino/dinov2.md) | 图像 | `nn.Module` | [vision/dinov2](../examples/vision/dinov2.ipynb) |
| TD-MPC | 控制 | [TD-MPC2](tdmpc/tdmpc2.md) | 状态或帧 + 动作 + 奖励 | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |

每个模型的 `forward` 都返回一个包含标量 `"loss"` 的 dict。每个 `WorldModel` 还提供 `encode`、`predict` 和 `rollout`，并可直接使用 [`CEM`](../guides/planning.md) 进行规划；TD-MPC2 使用 `MPPI` 以奖励为目标进行规划。如需添加模型，请参阅[接入你自己的模型](../guides/custom-model.md)。
