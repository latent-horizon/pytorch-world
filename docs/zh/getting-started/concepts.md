# 核心概念

## 什么是世界模型 {#what-a-world-model-is}

**世界模型**是对环境如何变化的一种学习得到的模型。智能体执行动作，动作改变世界中隐藏的**状态**，而智能体只能获得该状态的部分**观测**。世界模型刻画的正是这一循环：根据已有的观测和一个动作，预测接下来会发生什么。能够预测自身动作后果的智能体，可以在真正行动之前先行演练：它可以进行规划，在自己的想象中学习行为，或者模拟罕见事件，而不必等待它们真实发生。

这一思想由来已久。Kenneth Craik 在 1943 年提出，心智借助现实的"小尺度模型"进行推理；Schmidhuber 在 1990 年训练循环网络来预测自身未来的输入；Ha 和 Schmidhuber 的 *World Models*（2018）则让智能体在自己生成的梦境中接受训练。此后，这一领域沿着两个问题分化：**模型输出什么**，以及**模型被训练去预测什么**。

## 世界模型输出什么 {#what-a-world-model-outputs}

不同的使用者需要对同一个智能体–世界循环的不同投影。World Labs 的[功能分类法](https://www.worldlabs.ai/blog/taxonomy-of-world-models)按输出对世界模型进行分类：

| 类型 | 输出 | 评判标准 | 示例 |
|---|---|---|---|
| **渲染器** | 观测：供观看者使用的像素，通常根据用户输入实时生成 | 视觉保真度 | Genie 3、交互式视频生成器 |
| **模拟器** | 状态：可供程序或人查询的几何、物理与动力学 | 结构与物理准确性 | 3D 场景生成器、基于物理引擎的数字孪生 |
| **规划器** | 动作：给定观测和目标，下一步该做什么 | 智能体是否到达目标 | 基于模型的控制、视觉-语言-动作模型、世界动作模型 |

这几类模型都需要掌握几何、物理和动力学方面的知识，而且它们正在趋于融合：渲染器开始以动作为条件，规划器则以预训练的视频渲染器作为骨干网络。

`torch_world` 关注的是**规划器**：这里的每个世界模型都是为了选择动作而存在。在内部，每个模型仍以各自的方式进行渲染或模拟，而各个模型家族的区别在于它们预测什么。

## 世界模型预测什么 {#what-a-world-model-predicts}

| 方法 | 预测内容 | 训练方式 | 在 `torch_world` 中 | 其他实现 |
|---|---|---|---|---|
| **隐空间预测** | 下一观测的*嵌入* | 嵌入空间中的预测误差，外加防止坍缩的一项 | [JEPA](../models/jepa/index.md)（LeWM、RoboJEPA）、[DINO-WM](../models/dino/dinowm.md) | V-JEPA 2-AC |
| **价值等价隐变量** | 下一隐状态、奖励以及后续的价值 | 隐空间一致性与时序差分学习，无解码器 | [TD-MPC2](../models/tdmpc/tdmpc2.md) | MuZero |
| **生成式视频与动作** | 未来帧（以视频 VAE 隐变量表示）*以及*产生这些帧的动作块 | 去噪（flow matching） | [世界动作模型](../models/wam/index.md)（DreamZero）、[LeWAM](../models/jepa/lewam.md) | 基于视频预训练的机器人策略 |
| **重建** | 下一隐状态，以及由其解码出的像素 | 观测的似然 | | PlaNet、Dreamer |

预测像素迫使模型解释画面中的一切，包括对控制无关紧要的纹理和噪声；只预测嵌入或价值，则让模型把容量花在可预测且相关的内容上，代价是需要另寻办法防止表征坍缩。生成式模型虽然要为像素付出代价，却能继承大规模视频数据集中关于世界如何运动的知识。

这些家族中还包括本身并非世界模型的**表征学习器**，例如 I-JEPA、V-JEPA、LeJEPA 和 [DINOv2](../models/dino/dinov2.md)。它们为世界模型预训练所依赖的编码器：DINO-WM 就是在冻结的 DINOv2 特征上进行规划。

## 预测表征，而非像素 {#predict-representations-not-pixels}

隐空间方法是 `torch_world` 中最早的模型家族。**联合嵌入预测架构**（JEPA）分别编码*上下文*和*目标*，并训练一个预测器，将上下文的嵌入映射到目标的嵌入。它从不重建像素，因此模型容量用于可预测的内容，而不是纹理和噪声。

难点在于坍缩：常数嵌入可以完美地预测自身。`torch_world` 中的模型通过以下方式之一避免坍缩：

| 策略 | 做法 | 模型 |
|---|---|---|
| EMA 教师 | 目标来自编码器的缓慢更新副本，并使用 stop-gradient | I-JEPA, V-JEPA, DINOv2 |
| 分布正则化 | [SIGReg](../api/losses.md) 推动嵌入趋向各向同性高斯分布 | LeJEPA, LeVJEPA, LeWM, LeWAM |
| 冻结编码器 | 预训练编码器完全不参与训练 | DINO-WM, RoboJEPA |
| 对比学习 | [InfoNCE](../api/losses.md) 将不匹配的样本对推开 | VL-JEPA |
| 奖励与价值 | 隐状态必须预测奖励和价值，而常数无法做到 | TD-MPC2 |

## 从 JEPA 到世界模型 {#from-jepa-to-world-model}

当上下文是当前、目标是未来，且预测器还读取所执行的动作时，JEPA 就成为一个**隐空间世界模型**：它在嵌入空间中想象动作的后果。规划随后搜索那些想象结果最接近目标嵌入的动作。

<figure class="tw-diagram">
<svg viewBox="0 0 820 330" role="img" aria-labelledby="twd-title twd-desc" xmlns="http://www.w3.org/2000/svg">
<title id="twd-title">隐空间世界模型，朝目标进行规划</title>
<desc id="twd-desc">观测被编码为隐状态 z_t。预测器将 z_t 和动作 a_t 映射为预测的下一隐状态，并展开多步。代价函数比较最终预测的隐状态与编码后的目标图像，规划器调整动作以降低代价。</desc>
<defs>
<marker id="twd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head" d="M0 1 L9 5 L0 9 z"/></marker>
<marker id="twd-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head-accent" d="M0 1 L9 5 L0 9 z"/></marker>
<clipPath id="twd-clip-obs"><rect x="18" y="72" width="76" height="76" rx="12"/></clipPath>
<clipPath id="twd-clip-goal"><rect x="18" y="217" width="76" height="76" rx="12"/></clipPath>
</defs>
<text class="twd-lane" x="18" y="40">想象</text>
<text class="twd-lane" x="18" y="206">目标</text>
<image href="../../assets/diagrams/macaw_observation.png" x="18" y="72" width="76" height="76" clip-path="url(#twd-clip-obs)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="72" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="170" text-anchor="middle">观测</text>
<line class="twd-line" x1="100" y1="110" x2="128" y2="110"/>
<rect class="twd-box" x="130" y="88" width="96" height="44" rx="12"/>
<text x="178" y="116" text-anchor="middle">编码</text>
<line class="twd-line" x1="226" y1="110" x2="257" y2="110"/>
<circle class="twd-z" cx="288" cy="110" r="29"/>
<text class="twd-math" x="286" y="116" text-anchor="middle">z<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="317" y1="110" x2="352" y2="110"/>
<rect class="twd-box twd-box-strong" x="354" y="88" width="104" height="44" rx="12"/>
<text x="406" y="116" text-anchor="middle">预测</text>
<line class="twd-line" x1="458" y1="110" x2="489" y2="110"/>
<circle class="twd-zhat" cx="520" cy="110" r="29"/>
<text class="twd-math" x="514" y="116" text-anchor="middle">ẑ<tspan class="twd-sub" dy="4">t+1</tspan></text>
<path class="twd-line" d="M520 80 C520 34 406 34 406 86"/>
<text class="twd-caption" x="463" y="40" text-anchor="middle">展开</text>
<line class="twd-line" x1="549" y1="110" x2="618" y2="110"/>
<text class="twd-caption twd-math" x="583" y="98" text-anchor="middle">ẑ<tspan class="twd-sub" dy="3">T</tspan></text>
<rect class="twd-box twd-box-cost" x="620" y="88" width="120" height="44" rx="12"/>
<text x="680" y="116" text-anchor="middle">代价</text>
<rect class="twd-box" x="370" y="190" width="72" height="34" rx="17"/>
<text class="twd-math" x="402" y="213" text-anchor="middle">a<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="406" y1="190" x2="406" y2="134"/>
<path class="twd-line twd-plan" d="M650 134 L650 207 L444 207"/>
<text class="twd-caption twd-plan-text" x="548" y="198" text-anchor="middle">规划：降低代价</text>
<image href="../../assets/diagrams/macaw_goal.png" x="18" y="217" width="76" height="76" clip-path="url(#twd-clip-goal)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="217" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="315" text-anchor="middle">目标图像</text>
<line class="twd-line" x1="100" y1="255" x2="128" y2="255"/>
<rect class="twd-box" x="130" y="233" width="96" height="44" rx="12"/>
<text x="178" y="261" text-anchor="middle">编码</text>
<line class="twd-line" x1="226" y1="255" x2="257" y2="255"/>
<circle class="twd-zgoal" cx="288" cy="255" r="29"/>
<text class="twd-math" x="282" y="261" text-anchor="middle">z<tspan class="twd-sub" dy="4">goal</tspan></text>
<path class="twd-line" d="M317 255 L710 255 L710 134"/>
</svg>
<figcaption>编码当前状态，逐个动作地想象未来，并通过搜索想象结果最接近目标的动作来进行规划。</figcaption>
</figure>

## 选择动作的三种方式 {#three-ways-to-choose-actions}

这里的每个世界模型都使用同一个 `rollout` 进行想象；它们的区别在于用什么来评判想象出的未来，以及候选动作从何而来。

| 规划方式 | 想象未来的评分依据 | 候选动作来源 | 模型 |
|---|---|---|---|
| 使用 [`CEM`](../guides/planning.md) 到达目标 | 最终隐状态与目标隐状态之间的距离（`cost`） | 高斯分布，按最优样本重新拟合 | LeWM, DINO-WM, RoboJEPA, 任意 `WorldModel` |
| 使用 [`MPPI`](../guides/planning.md#mppi-planning-for-reward) 最大化奖励 | 想象的奖励加上视野末端的学习价值 | 高斯分布与策略先验 | TD-MPC2 |
| 使用 `WAMPlanner` 进行策略采样 | 模型自身朝向目标的动力学 | 模型的动作头（`sample_actions`） | DreamZero（与 `DreamZeroPolicy` 闭环运行）、LeWAM |

## 基类 {#two-base-classes}

每个模型都继承自 [`torch_world.models`](../api/models.md) 中的三个类之一。

| 基类 | 你需要实现 | 你将获得 |
|---|---|---|
| `JEPA` | `forward(...) -> {"loss": ...}` | `make_target_encoder`, `update_target(momentum)` |
| `WorldModel` | `encode`, `predict(z, actions)`, `forward` | `rollout`, `cost`, 以及使用 [`CEM`](../guides/planning.md) 进行规划 |
| `WorldActionModel` | 一个 `WorldModel`，外加 `sample_actions`、`predict_future` 和 `imagine` | 使用 `WAMPlanner` 进行规划；当隐变量有解码器时提供 `decode` |

## 约定 {#conventions}

**损失字典。** `forward` 返回一个 dict，其中包含用于反向传播的标量 `"loss"`，以及用于日志记录的各个命名分量（`"pred_loss"`、`"sigreg_loss"` 等）。

**时间维度是第 1 维。** episode 和隐状态的形状为 `(B, T, ...)`。对于 `WorldModel`，`predict(z, actions)` 将 `z[:, t]` 和 `actions[:, t]` 映射为对 `z[:, t + 1]` 的预测，并对所有 `t` 同时进行（teacher forcing）。

**动作与状态对齐。** 在收集的 episode 中，`actions[:, t]` 作用于 `states[:, t]` 并产生 `states[:, t + 1]`。最后一个动作是填充值。

**历史长度。** `WorldModel.history` 是预测器能看到的过去隐状态的数量。`rollout` 以该长度滑动窗口；`None` 表示使用全部历史。

**掩码是索引张量。** 掩码是一个 `(B, K)` 张量，存放需要*保留*的 token 索引；掩码策略返回由这些张量组成的列表，每个块一个。[`apply_masks`](../api/masks.md) 负责按索引收集，并将多个掩码沿 batch 维堆叠。

**视频采用通道优先格式。** `VisionTransformer` 接受图像 `(B, C, H, W)` 或视频 `(B, C, T, H, W)`；环境产生的 episode 形状为 `(B, T, C, H, W)`，因此在送入视频编码器前需交换第 1 维和第 2 维。

## 延伸阅读 {#further-reading}

- World Labs，[A functional taxonomy of world models](https://www.worldlabs.ai/blog/taxonomy-of-world-models)，2026：渲染器、模拟器与规划器。
- [World model (artificial intelligence)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence))，Wikipedia：历史与方法。
- Ha and Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)，2018。
- LeCun，[A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)，2022：JEPA 的提出。
- Hafner et al.，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（PlaNet），2019，以及 [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)（DreamerV3），2023。
- Schrittwieser et al.，[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)（MuZero），2020。
