# モデル

モデルは**ファミリー**ごとにまとめられています。同じファミリーのモデルは学習方法と基底クラスを共有しているため、同じデータ、環境、プランナーの上で入れ替えて比較できます。ファミリーは **JEPA**、**DINO**、**TD-MPC** です。ファミリーとモデルは今後も追加されます。

<div class="grid cards tw-models tw-models--families">

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    ターゲットのピクセルではなく*表現*を予測します。行動を加えると潜在世界モデルになります。

-   :material-school-outline:{ .lg } **[DINO](dino/index.md)**

    学習されたプロトタイプへの自己蒸留と、その凍結特徴量の上に構築した世界モデル。

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    TD 学習で訓練するデコーダ不要の潜在ダイナミクスで、リターンを最大化するようにプランニングします。

</div>

## 全モデル一覧 {#every-model-at-a-glance}

| ファミリー | グループ | モデル | 入力 | 基底クラス | ノートブック |
|---|---|---|---|---|---|
| JEPA | 行動 | [LeWM](jepa/lewm.md) | フレーム + 行動 | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | 行動 | [LeWAM](jepa/lewam.md) | フレーム + 行動（+ ゴール画像） | `WorldModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | 行動 | [RoboJEPA](jepa/robojepa.md) | 多視点フレーム + 行動 + 状態 | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | 視覚 | [I-JEPA](jepa/ijepa.md) | 画像 | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | 視覚 | [LeJEPA](jepa/lejepa.md) | 画像 | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | 視覚 | [V-JEPA](jepa/vjepa.md) | 動画 | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | 視覚 | [LeVJEPA](jepa/levjepa.md) | 動画 | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | 視覚言語 | [VL-JEPA](jepa/vljepa.md) | 画像または動画 + テキスト | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| DINO | 行動 | [DINO-WM](dino/dinowm.md) | フレーム + 行動 + 固有受容感覚 | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| DINO | 視覚 | [DINOv2](dino/dinov2.md) | 画像 | `nn.Module` | [vision/dinov2](../examples/vision/dinov2.ipynb) |
| TD-MPC | 制御 | [TD-MPC2](tdmpc/tdmpc2.md) | 状態またはフレーム + 行動 + 報酬 | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |

すべてのモデルの `forward` は、スカラーの `"loss"` を含む dict を返します。すべての `WorldModel` は `encode`、`predict`、`rollout` も備え、そのまま [`CEM`](../guides/planning.md) でプランニングできます。TD-MPC2 は `MPPI` で報酬に向けてプランニングします。モデルを追加するには、[独自のモデルを持ち込む](../guides/custom-model.md)を参照してください。
