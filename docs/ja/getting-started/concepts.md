# コンセプト

## 世界モデルとは {#what-a-world-model-is}

**世界モデル**とは、環境がどのように変化するかを学習したモデルです。エージェントが行動すると、その行動は世界の隠れた**状態**を変化させ、エージェントはその状態の部分的な**観測**を受け取ります。世界モデルはこのループを捉えます。つまり、これまでの観測と行動から、次に何が起こるかを予測します。自分の行動の結果を予測できるエージェントは、実行する前にそれを予行演習できます。プランニングを行うことも、自らの想像の中で振る舞いを学習することも、まれな事象を待つ代わりにシミュレートすることもできます。

このアイデアは古くからあります。Kenneth Craik は 1943 年に、心は現実の「小規模なモデル」を用いて推論すると提唱しました。Schmidhuber は 1990 年に、自身の将来の入力を予測するリカレントネットワークを学習させ、Ha と Schmidhuber の *World Models*（2018）は、エージェントを自ら作り出した夢の中で学習させました。それ以来、この分野は 2 つの問いに沿って分かれてきました。**モデルが何を出力するか**、そして**何を予測するように学習されるか**です。

## 世界モデルが出力するもの {#what-a-world-model-outputs}

利用者が異なれば、同じエージェントと世界のループについて必要とする射影も異なります。World Labs の[機能的分類](https://www.worldlabs.ai/blog/taxonomy-of-world-models)は、世界モデルを出力によって分類します。

| 種類 | 出力 | 評価基準 | 例 |
|---|---|---|---|
| **レンダラー** | 観測：視聴者向けのピクセル。多くの場合、ユーザー入力からリアルタイムに生成 | 視覚的な忠実度 | Genie 3、インタラクティブな動画生成器 |
| **シミュレータ** | 状態：プログラムや人が問い合わせられる幾何、物理、ダイナミクス | 構造的・物理的な正確さ | 3D シーン生成器、物理エンジンによるデジタルツイン |
| **プランナー** | 行動：観測とゴールが与えられたときに次に何をするか | エージェントがゴールに到達するかどうか | モデルベース制御、vision-language-action モデル、world action モデル |

これらの種類は、幾何、物理、ダイナミクスについて知っておくべきことを共有しており、互いに収束しつつあります。レンダラーは行動条件付きになり、プランナーは事前学習済みの動画レンダラーをバックボーンとして用います。

`torch_world` が扱うのは**プランナー**です。ここにあるすべての世界モデルは行動を選ぶために存在します。内部的には、それぞれが独自の方法でレンダリングやシミュレーションを行っており、ファミリーごとに何を予測するかが異なります。

## 世界モデルが予測するもの {#what-a-world-model-predicts}

| アプローチ | 予測するもの | 学習方法 | `torch_world` での実装 | その他 |
|---|---|---|---|---|
| **潜在予測** | 次の観測の*埋め込み* | 埋め込み空間での予測誤差と、崩壊を防ぐ項 | [JEPA](../models/jepa/index.md)（LeWM、RoboJEPA）、[DINO-WM](../models/dino/dinowm.md) | V-JEPA 2-AC |
| **価値等価な潜在表現** | 次の潜在表現、報酬、およびその後の価値 | 潜在の一貫性と TD 学習、デコーダなし | [TD-MPC2](../models/tdmpc/tdmpc2.md) | MuZero |
| **ダイナミクスと行動を同時に** | 未来の潜在表現*と*、それを生み出す行動チャンク | 潜在予測に加え、行動に対する flow matching | [LeWAM](../models/jepa/lewam.md) | 事前学習済み動画拡散モデル上の世界行動モデル |
| **再構成** | 次の潜在状態と、そこから得られるピクセル | 観測の尤度 | | PlaNet、Dreamer |

ピクセルを予測すると、モデルはフレーム内のすべて、つまり制御には関係のないテクスチャやノイズまで説明しなければなりません。埋め込みや価値だけを予測すれば、予測可能で関連性のある部分に容量を使えますが、その代わりに表現の崩壊を防ぐ別の手段が必要になります。動画を生成するモデルはピクセルのコストを払いますが、大規模な動画データセットが持つ、世界がどう動くかについての知識を受け継ぎます。

これらのファミリーには、それ自体は世界モデルではない**表現学習器**も含まれます。I-JEPA、V-JEPA、LeJEPA、[DINOv2](../models/dino/dinov2.md) などです。これらは世界モデルの土台となるエンコーダを事前学習します。DINO-WM は凍結した DINOv2 の特徴量の上でプランニングを行います。

## ピクセルではなく表現を予測する {#predict-representations-not-pixels}

潜在アプローチは `torch_world` で最も古いファミリーです。**結合埋め込み予測アーキテクチャ**（JEPA）は、*コンテキスト*と*ターゲット*をエンコードし、コンテキストの埋め込みをターゲットの埋め込みへ写す予測器を学習します。ピクセルを再構成することは決してないため、モデルの容量はテクスチャやノイズではなく、予測可能な部分に使われます。

難しいのは崩壊（collapse）です。定数の埋め込みは自分自身を完璧に予測できてしまいます。`torch_world` のモデルは、次のいずれかの方法でこれを回避します。

| 戦略 | 方法 | モデル |
|---|---|---|
| EMA 教師 | ターゲットは、ゆっくり更新されるエンコーダのコピーから stop-gradient 付きで得る | I-JEPA, V-JEPA, DINOv2 |
| 分布正則化 | [SIGReg](../api/losses.md) が埋め込みを等方ガウス分布に近づける | LeJEPA, LeVJEPA, LeWM, LeWAM |
| 凍結エンコーダ | 事前学習済みエンコーダを一切学習しない | DINO-WM, RoboJEPA |
| 対照学習 | [InfoNCE](../api/losses.md) が一致しないペアを引き離す | VL-JEPA |
| 報酬と価値 | 潜在表現は報酬と価値を予測しなければならず、定数ではそれができない | TD-MPC2 |

## JEPA から世界モデルへ {#from-jepa-to-world-model}

コンテキストが現在、ターゲットが未来であり、さらに予測器が実行された行動も読み込むとき、JEPA は**潜在世界モデル**になります。つまり、行動の結果を埋め込み空間で想像します。プランニングは、想像された結果がゴールの埋め込みに最も近くなる行動を探索することになります。

<figure class="tw-diagram">
<svg viewBox="0 0 820 330" role="img" aria-labelledby="twd-title twd-desc" xmlns="http://www.w3.org/2000/svg">
<title id="twd-title">潜在世界モデルによるゴールへのプランニング</title>
<desc id="twd-desc">観測は潜在表現 z_t にエンコードされます。予測器は z_t と行動 a_t を予測された次の潜在表現へ写し、複数ステップにわたってロールアウトします。コストは最終的な予測潜在表現とエンコードされたゴール画像を比較し、プランナーはそれを下げるように行動を調整します。</desc>
<defs>
<marker id="twd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head" d="M0 1 L9 5 L0 9 z"/></marker>
<marker id="twd-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head-accent" d="M0 1 L9 5 L0 9 z"/></marker>
<clipPath id="twd-clip-obs"><rect x="18" y="72" width="76" height="76" rx="12"/></clipPath>
<clipPath id="twd-clip-goal"><rect x="18" y="217" width="76" height="76" rx="12"/></clipPath>
</defs>
<text class="twd-lane" x="18" y="40">想像</text>
<text class="twd-lane" x="18" y="206">ゴール</text>
<image href="../../assets/diagrams/macaw_observation.png" x="18" y="72" width="76" height="76" clip-path="url(#twd-clip-obs)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="72" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="170" text-anchor="middle">観測</text>
<line class="twd-line" x1="100" y1="110" x2="128" y2="110"/>
<rect class="twd-box" x="130" y="88" width="96" height="44" rx="12"/>
<text x="178" y="116" text-anchor="middle">エンコード</text>
<line class="twd-line" x1="226" y1="110" x2="257" y2="110"/>
<circle class="twd-z" cx="288" cy="110" r="29"/>
<text class="twd-math" x="286" y="116" text-anchor="middle">z<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="317" y1="110" x2="352" y2="110"/>
<rect class="twd-box twd-box-strong" x="354" y="88" width="104" height="44" rx="12"/>
<text x="406" y="116" text-anchor="middle">予測</text>
<line class="twd-line" x1="458" y1="110" x2="489" y2="110"/>
<circle class="twd-zhat" cx="520" cy="110" r="29"/>
<text class="twd-math" x="514" y="116" text-anchor="middle">ẑ<tspan class="twd-sub" dy="4">t+1</tspan></text>
<path class="twd-line" d="M520 80 C520 34 406 34 406 86"/>
<text class="twd-caption" x="463" y="40" text-anchor="middle">ロールアウト</text>
<line class="twd-line" x1="549" y1="110" x2="618" y2="110"/>
<text class="twd-caption twd-math" x="583" y="98" text-anchor="middle">ẑ<tspan class="twd-sub" dy="3">T</tspan></text>
<rect class="twd-box twd-box-cost" x="620" y="88" width="120" height="44" rx="12"/>
<text x="680" y="116" text-anchor="middle">コスト</text>
<rect class="twd-box" x="370" y="190" width="72" height="34" rx="17"/>
<text class="twd-math" x="402" y="213" text-anchor="middle">a<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="406" y1="190" x2="406" y2="134"/>
<path class="twd-line twd-plan" d="M650 134 L650 207 L444 207"/>
<text class="twd-caption twd-plan-text" x="548" y="198" text-anchor="middle">プランニング：コストを下げる</text>
<image href="../../assets/diagrams/macaw_goal.png" x="18" y="217" width="76" height="76" clip-path="url(#twd-clip-goal)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="217" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="315" text-anchor="middle">ゴール画像</text>
<line class="twd-line" x1="100" y1="255" x2="128" y2="255"/>
<rect class="twd-box" x="130" y="233" width="96" height="44" rx="12"/>
<text x="178" y="261" text-anchor="middle">エンコード</text>
<line class="twd-line" x1="226" y1="255" x2="257" y2="255"/>
<circle class="twd-zgoal" cx="288" cy="255" r="29"/>
<text class="twd-math" x="282" y="261" text-anchor="middle">z<tspan class="twd-sub" dy="4">goal</tspan></text>
<path class="twd-line" d="M317 255 L710 255 L710 134"/>
</svg>
<figcaption>現在をエンコードし、行動を 1 つずつ与えて未来を想像し、想像された結果がゴールに最も近くなる行動を探索してプランニングします。</figcaption>
</figure>

## 行動を選ぶ 3 つの方法 {#three-ways-to-choose-actions}

ここにあるすべての世界モデルは同じ `rollout` で想像を行います。違いは、想像された未来を何に照らして評価するか、そして候補となる行動をどこから得るかです。

| プランニング | 想像された未来の評価基準 | 候補の出どころ | モデル |
|---|---|---|---|
| [`CEM`](../guides/planning.md) によるゴール到達 | 最終的な潜在表現とゴールの潜在表現との距離（`cost`） | 上位の候補に再フィットするガウス分布 | LeWM, DINO-WM, RoboJEPA, 任意の `WorldModel` |
| [`MPPI`](../guides/planning.md#mppi-planning-for-reward) による報酬 | 想像された報酬と、ホライズン終端での学習済み価値 | ガウス分布と方策の事前分布 | TD-MPC2 |
| `LeWAMPlanner` による方策サンプリング | ゴールに向かうモデル自身のダイナミクス | モデルの flow matching 行動ヘッド | LeWAM |

## 2 つの基底クラス {#two-base-classes}

すべてのモデルは、[`torch_world.models`](../api/models.md) にある 2 つのクラスのいずれかを継承します。

| 基底クラス | 実装するもの | 得られるもの |
|---|---|---|
| `JEPA` | `forward(...) -> {"loss": ...}` | `make_target_encoder`, `update_target(momentum)` |
| `WorldModel` | `encode`, `predict(z, actions)`, `forward` | `rollout`, `cost`, [`CEM`](../guides/planning.md) によるプランニング |

## 規約 {#conventions}

**損失の dict。** `forward` は、逆伝播させるスカラーの `"loss"` と、ログ用の名前付き構成要素（`"pred_loss"`、`"sigreg_loss"` など）を含む dict を返します。

**時間は次元 1。** エピソードと潜在表現は `(B, T, ...)` です。`WorldModel` では、`predict(z, actions)` が `z[:, t]` と `actions[:, t]` を `z[:, t + 1]` の予測へ写し、これをすべての `t` について一度に行います（teacher forcing）。

**行動は状態に揃える。** 収集されたエピソードでは、`actions[:, t]` は `states[:, t]` で適用され、`states[:, t + 1]` を生み出します。最後の行動はパディングです。

**履歴。** `WorldModel.history` は、予測器が参照する過去の潜在表現の数です。`rollout` はその長さのウィンドウをスライドさせます。`None` は過去全体を意味します。

**マスクはインデックステンソル。** マスクは*保持する*トークンのインデックスからなる `(B, K)` テンソルであり、マスキング戦略はブロックごとに 1 つずつ、そのリストを返します。[`apply_masks`](../api/masks.md) はそれらを gather し、複数のマスクをバッチ次元に積み重ねます。

**動画はチャネルファースト。** `VisionTransformer` は画像 `(B, C, H, W)` または動画 `(B, C, T, H, W)` を受け取ります。環境から得られるエピソードは `(B, T, C, H, W)` なので、動画エンコーダに入力する前に次元 1 と 2 を入れ替えてください。

## 参考文献 {#further-reading}

- World Labs, [A functional taxonomy of world models](https://www.worldlabs.ai/blog/taxonomy-of-world-models), 2026：レンダラー、シミュレータ、プランナー。
- [World model (artificial intelligence)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)), Wikipedia：歴史とアプローチ。
- Ha and Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122), 2018.
- LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf), 2022：JEPA の提案。
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（PlaNet）, 2019、および [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)（DreamerV3）, 2023.
- Schrittwieser et al., [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)（MuZero）, 2020.
