# CRR 实验与方案设计日志

> 版本：v1.0
> 整理日期：2026-10-29
> 覆盖范围：2026-09 至 2026-10-29，CRR 路线 P0/E1 到 E19d、R1/R3，以及官方 MOTIP A6000 复现
> 维护位置：`source/MOTIP_crr_e19c_cleanscratch_v1/CRR_EXPERIMENT_LOG.md`
> 性质：工程与研究记录，不是论文；所有结论都标注证据强度与适用边界

---

## 0. 阅读说明

### 0.1 指标口径

所有跟踪指标均为 **官方 DanceTrack val**、**TrackEval** 计算，推理协议除非特别说明均为：

| 项 | 值 |
|---|---|
| Assignment | object-max |
| DET_THRESH | 0.3 |
| NEWBORN_THRESH | 0.6 |
| ID_THRESH | 0.2 |
| MISS_TOLERANCE | 30 |
| INFERENCE_MAX_LONGER | 1440 |

指标含义：

| 指标 | 含义 | 方向 |
|---|---|---|
| HOTA | 检测与关联的联合精度 | 越高越好 |
| DetA | 检测精度 | 越高越好 |
| AssA | 关联精度 | 越高越好 |
| AssRe / AssPr | 关联召回 / 关联精度 | 越高越好 |
| IDF1 | 身份 F1 | 越高越好 |
| MOTA | 多目标跟踪精度 | 越高越好 |
| IDSW | ID 切换次数 | 越低越好 |
| Frag | 轨迹碎片数 | 越低越好 |
| IDs | 轨迹总数 | 越低越好（官方权重 1713） |

**关键锚点（官方 MOTIP 权重，val）**：

```
HOTA 64.099 | DetA 75.401 | AssA 54.745 | AssRe 61.453 | AssPr 70.882
IDF1 69.291 | IDSW 2682 | Frag 4279 | IDs 1713
```

### 0.2 结论强度标记

| 标记 | 含义 |
|---|---|
| **A（已测）** | 有同协议、可复核的完整评估数字 |
| **B（强证据）** | 有代码、配置、日志或对照审计支持，但缺少完整评估 |
| **C（假设）** | 机制上合理，尚未被实验隔离 |
| **D（未验证）** | 只有设计或启动脚本，没有训练输出 |

本日志严格区分“已经发生的事实”和“对机制的解释”。R1/R3 的很多判断属于 B/C，不应写成 A。

### 0.3 实验命名

| 前缀 | 含义 |
|---|---|
| E | exploratory，探索性实验 |
| R | reset / redesign，架构重写后的实验 |
| CS | clean scratch，从零训练、不 resume |
| cont7 | 续训 7 个 epoch 的配方 |
| ABL | ablation，消融组 |
| OFF | 官方 MOTIP 权重锚点 |

### 0.4 证据来源

| 来源 | 位置 |
|---|---|
| 官方权重锚点 | 既有 4090 复现记录，后被 A6000 复现再次校准 |
| E5/E18/E19/E19b/E19c 结果 | `G:\mot_stage\motip_paradigm_reset\...\*_RESULTS.txt` |
| E15 复盘 | `crr_e15_armlite_postmortem_v1\FINDINGS.md` |
| E17 规格 | `crr_e17_geom_scorepath_v1\CRR_E17_SPEC.md` |
| 各实验冻结配置 | 各输出目录的 `frozen_config.yaml` |
| R1/R3 日志 | `motip_paradigm_reset\r1_*\train\log.txt` |
| 官方复现日志 | `official_motip_dancetrack_a6000_seed42_v2\train\log.txt` |
| 代码现状 | `source/MOTIP_crr_e19c_cleanscratch_v1` |

---

## 1. 研究命题与路线

### 1.1 核心命题

CRR（Causal Recurrent Representation / 因果实体表示）把身份定义为：

> **身份是由历史观测生成的动态因果实体状态；数字 ID 只是访问该状态的地址，不是需要学习的类别。**

由此推出三条硬约束：

1. **严格因果**：当前帧的决策只能使用更早的 mention，未来帧不得影响过去。
2. **重命名等变**：对 ID 地址做任意置换，实体分数列必须同步置换，模型输出不变。
3. **无固定 ID 词典**：不学习“第 k 个 ID 长什么样”，而是从历史 mention 生成实体表示，再用当前 query 去读。

这与官方 MOTIP 的 ID Decoder 有本质区别：官方是**可学习 ID 词典 + 软注意力 + 线性分类**；CRR 是**历史生成的实体状态 + 地址聚合 + 点积读出**。

### 1.2 分阶段路线

来自 `CRR_FULL_ROADMAP.md`：

| 阶段 | 名称 | 核心内容 | 状态 |
|---|---|---|---|
| P0 | Causal pointer | 学习 query/key 度量，严格因果 mask，max mention 检索 | 已实现并作为基线 |
| E1 | Query-conditional causal entity state | 可靠性/时间感知注意力，从多 mention 生成动态状态；零初始化残差保证 step-0 等价 P0 | 已实现 |
| E2 | Relational current-set reasoner | 当前检测之间做置换等变 self-attention，把一对一冲突解决移入表示 | 路线图阶段，未实现 |
| E3 | Context-conditioned NEW competition | NEW 相对最佳已有实体、margin、不确定性、轨迹年龄与集合上下文评分 | 路线图阶段，未实现 |
| E4 | Structured allocation + robust history curriculum | 训练目标对齐增强后的“已有实体 + 复制 NEW”分配矩阵；历史 corruption 从遮挡到换 ID 渐进 | 路线图阶段，未实现 |

实验纪律（路线图原文要点）：

- E1/E2/E3/E4 顺序加入，每一步都有匹配消融。
- smoke 只证明能跑，不证明精度；可学习阶段至少 3 个完整 epoch。
- 架构决策用训练侧开发协议；锁定验证集留给冻结候选。
- 失败的 B3/B6 后处理 gate 不再搬进 decoder；可靠性以软方式进入表示。
- 最终报告必须包含精度、参数量、FLOPs、收敛曲线、seed 稳定性，并同时对比 P0 与成熟官方 ID Decoder。

### 1.3 早期设计基线

#### P0 — Causal pointer

- query/key 各自线性投影后做归一化余弦相似度，乘 `sqrt(d)`。
- 历史按时间严格 mask，当前 query 只能看更早 mention。
- 每个 ID 地址取 mention 最大值（`amax`）作为实体分数。
- 优点：路径短、梯度直接、天然等变。
- 缺点：单点极值，对污染 mention 极敏感；没有多帧证据聚合。

#### E1 — Entity state residual

- 对每个 ID 地址的历史 mention 做可靠性/时间感知 softmax 聚合，得到动态实体状态。
- 实体状态分数与 P0 的 max 分数通过**零初始化残差**融合，保证初始化时与 P0 完全一致。
- 五条不变量：严格因果、重命名等变、P0-exact 初始化、NEW 始终可达、梯度可达全部件。
- 验证协议：40-step smoke → 3 个 train32 epoch → 从 P0 checkpoint 9 继续。

#### Module A v2

`MODULE_A_DESIGN.txt` 记录：

> P0 max + bounded zero-init top-k causal consensus residual，仅 `consensus_mlp` 可训练。

这是“最小可学习补丁”的思路：不改官方主干，只在分数路径上加一个有界的、零初始化的共识残差。

---

## 2. 实验时间线总表

| 日期 | 实验 | 目的 | 关键改动 | 结果 | 判定 |
|---|---|---|---|---|---|
| 2026-09 | E5 | 建立 CRR 可用的端到端配方 | 4 帧梯度 + 26 帧无梯度；三个 CRR 训练机制；NEW weight 1.5 | 峰值 HOTA 62.484 / IDs 5060 | 可用基线，但 IDs 远高于官方 1713 |
| 2026-09 | E5o | 冷启动原始对照 | 从零训练 | ep0–2 IDs 28217→9715→7626 | 证明满 LR ep2 存在相变 |
| 2026-09 | E6 | 在分数路径加运动/几何残差 | 加性 residual | AssA 48.505→42.435；IDSW 25081→44920 | 路线关闭 |
| 2026-09-21 | E7a | 有状态 trust gate，仅架构 | 从 E5 ckpt5 出发，gate loss 0 | 结果未随归档保留 | 分支未继续 |
| 2026-10-04 | E8 | armlite 离线头/阈值控制 | 仅改 ID_THRESH 或 head | 两臂 exit=0 | 触发 E15 复盘 |
| 2026-10-05 | E15 | 阈值控制崩溃复盘 | 审计闭环 | 25/25 序列 accepted_existing=0 | 阈值会改写历史，闭环崩溃 |
| 2026-10 | E13 | 训练侧特征 dump | 40 序列 train dumps | 无训练指标 | 诊断基础设施 |
| 2026-10 | E14 | 解耦 rerank，写回轨迹状态 | `log score + α·IoU` | 损失 0.5–2.2 HOTA | 路线关闭 |
| 2026-10 | E16 | 只重写输出 ID 的 rerank | label-only | 仍负收益 | 推理侧 rerank 关闭 |
| 2026-10-05/08 | E17 | 几何进入分数路径 | SPEC，未实现 | 离线 AUC 0.8954→0.9791 | 有信息，不等于有收益 |
| 2026-10-09→14 | E18 | 消融 E5 的三个 CRR 机制 | 三项 OFF，switch/occlusion 仍 0.5 | ep6 HOTA 62.299 / IDs 2311 | 机制非必须；配方成为新基线 |
| 2026-10-14→18 | E19 | 从零训练 + bounded NEW prototype | 官方配方 + CRR pointer | ep0–3 IDs 30298/43552/33466/43890 | 冷启动失败 |
| 2026-10-17→18 | E19b | 只换官方无界线性 NEW head | 回滚 E19 几何修复 | ep0 HOTA 50.348 / IDs 49175 | 单换 NEW head 不够 |
| 2026-10-19→26 | E19c | 三项修复包 | switch→0、解除零初始化休眠、可学习 logit scale | ep0 52.469 / 1246；ep1 48.432 | 冷启动改善，未证明收敛 |
| 2026-10-26 | E19d | context-aware NEW head | 仅启动脚本 | 无训练输出 | 未验证 |
| 2026-10-26→29 | R1 v1/v2/v3 | 用等变集合解码器替换官方 ID Decoder | 只改关联头 | v2 ep0 HOTA 41.726 / IDs 2179 | 关联退化，NEW 失能 |
| 2026-10-28→29 | R3（修改前） | 六层等变 decoder + E19c 证据 | 深层注意力、amax、GRU | 无 eval，id_loss 停在 1.29 | 关联目标学得慢 |
| 2026-10-29 | R3（修改后） | 七项修复 | 见 §5.2 | 未训练 | 待验证 |
| 2026-10-29 | 官方 A6000 v2 | 独立复现官方 64.099 | 官方配置，从零 | 进行中 | 唯一有效官方基线 |

---

## 3. 实验详录

### 3.1 E5 — CRR 的第一个可用配方

**目的**：在官方 DanceTrack 配方上，把官方 ID Decoder 换成 CRR causal entity pointer，验证 CRR 能否端到端学出关联。

**配方**（`crr_e5_e2e_train40_cont7_seed42_v1`）：

- 数据：`motip_official_train40_v1`，4 个梯度帧 + 26 个无梯度帧。
- 优化：LR 1e-4，warmup 0，7 epoch，milestones [3, 6]，gamma 0.2，batch 1，seed 42。
- 增强：occlusion 0.5，switch 0.5（官方增强）。
- CRR 机制：history blackout 0.6、prefix 0.25、结构化 history corruption 0.15→0.05、NEW class weight 1.5。
- 从 `crr_e5_e2e_train40_epoch12_seed42_v1/checkpoint_4.pth` 续训。

**逐 epoch 结果**（HOTA / DetA / AssA / AssRe / AssPr / IDF1 / IDSW / Frag / IDs）：

| ep | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58.657 | 72.703 | 47.603 | 52.265 | 76.295 | 60.558 | 9378 | 5600 | 8963 |
| 1 | 58.259 | 72.634 | 47.036 | 51.989 | 76.534 | 60.031 | 15271 | 4954 | 15872 |
| 2 | 59.490 | 72.595 | 49.087 | 53.149 | 77.576 | 61.345 | 7428 | 5163 | 6776 |
| 3 | 60.830 | 74.465 | 49.978 | 54.955 | 77.160 | 62.564 | 5676 | 4535 | 5507 |
| 4 | 62.270 | 74.935 | 52.067 | 56.424 | 78.438 | 64.180 | 6230 | 4433 | 5675 |
| 5 | **62.484** | 75.081 | 52.261 | 57.159 | 77.012 | 64.922 | 5597 | 4714 | **5060** |
| 6 | 62.273 | 74.785 | 52.144 | 56.700 | 79.367 | 63.648 | 12248 | 4553 | 12499 |

**结论（A）**：

- E5 是 CRR 第一个能稳定收敛到 62+ HOTA 的配方。
- 峰值在 ep5：HOTA 62.484、IDs 5060；距离官方 64.099 / 1713 仍有明显差距。
- ep6 IDs 从 5060 跳到 12499，说明闭环在训练后段仍不稳定。
- 因此 E5 只能作为“可用基线”，不能作为“已解决”。

**对后续的约束**：

- 任何新变体至少要看 3 个完整 epoch，不能拿单点 loss 判定成败。
- 必须观察 IDs 是否进入稳定盆地，而不是只看 HOTA 峰值。

---

### 3.2 E5o — 冷启动相变对照

**目的**：提供“从零训练时，关联何时进入稳定状态”的参照曲线。

**结果**：

| ep | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.685 | 72.618 | 41.461 | 46.136 | 75.671 | 55.846 | 27939 | 5164 | 28217 |
| 1 | 56.037 | 71.027 | 44.499 | 49.313 | 74.507 | 57.774 | 9894 | 5401 | 9715 |
| 2 | **58.144** | 71.622 | 47.453 | 52.514 | 74.795 | 60.993 | 7868 | 5669 | **7626** |
| 3 | 57.761 | 72.483 | 46.312 | 51.318 | 75.247 | 59.876 | 8311 | 5056 | 8715 |
| 4 | 57.142 | 73.084 | 44.934 | 49.154 | 76.983 | 58.350 | 15075 | 5258 | 14908 |

**结论（A）**：

- 冷启动存在明显相变：IDs 28217 → 9715 → 7626，发生在满 LR 的 ep2。
- 相变之后并不自动稳定：ep3–4 IDs 又回升到 8715 / 14908。
- 这条曲线是 E19/E19b/E19c 的**关键对照**：如果一个“修复版”到 ep2–3 还没有出现 ID 数量级下降，就不能声称冷启动被解决。

---

### 3.3 E6 — 运动/几何加性残差

**目的**：在分数路径上叠加运动/几何 residual，改善关联。

**结果（A，但原始文件未随本次归档下载）**：

- 相同权重下，通道开启：AssA 48.505 → 42.435，IDSW 25081 → 44920。
- 即使通道关闭，fresh Adam 端到端微调本身也造成明显退化。

**结论（A/B）**：

- 未校准的加性几何项不能修补关联，反而放大 IDSW。
- 优化器重置本身是强混杂：不能把微调退化归因于新模块，也不能把新模块的失败完全归因于模块。
- **路线关闭**：不再用未校准运动/几何加性 logit。

---

### 3.4 E7 / E8 / E15 — trust gate、离线头与阈值闭环

#### E7a — stateful trust gate

**动机（B）**：g-gate 诊断测得 E5 的 gate 接近饱和：`mean_g = 0.918`，85.8% 的有效项落在 [0.8, 0.9]。而 `g == 0` 消融显示：HOTA −7.26、AssA −11.00、IDSW ×5.75，说明 identity memory 通道承载了大部分关联质量。

**设计**：

- 在冻结的 E5 gate 路径上，加每实体 recurrent trust estimate。
- `POINTER_TRUST_GATE_LOSS_WEIGHT = 0.0`，只验证架构能否在纯 ID CE 下自己学会去饱和。
- 从 E5 checkpoint_5 出发，6 epoch，trust scale 0.5。
- LR 重新钉到归档的绝对尺度：backbone 2e-6、linear 1e-6、dictionary/rest 2e-5、新 trust 通道 1e-4，避免“未衰减 LR 冲击”污染结论。

**结果**：本次归档只保留了配置，没有保留逐 epoch 结果。**判定：D（未完成验证）**，该分支未继续。

#### E8 — armlite 控制臂

- `cfg_ctrl_id0828.yaml`：只把 `ID_THRESH` 从 0.2 改为 0.828212，其余完全一致。
- `cfg_head_tau52.yaml`：head 变体，`ID_THRESH` 仍 0.2。
- 两臂均 exit=0（2026-10-04）。

#### E15 — 控制臂复盘（A）

**问题**：一个只提高阈值的控制臂，为什么 AssA 掉到 0.112？

**审计结果（25/25 序列）**：

| 量 | anchor（ID_THRESH 0.2） | ctrl（0.828212） |
|---|---|---|
| objects_scored | 相同 | 相同 |
| accepted_existing | 4516–17127 / 序列 | **全部为 0** |
| existing_below_id_threshold | 0–854 | 全部对象 |
| mean best_existing_score | 0.879–0.998 | 0.064–0.178 |
| median ratio ctrl/anchor | — | **0.112** |
| tracks_created | 5–1076 | 1142–16424 |
| id lifetime | 12–989 帧 | **1.00 帧** |
| ids per 100 detections | 0.1–8.2 | **100.0** |

**机制（B）**：

1. 控制臂在每个序列第 2 帧就开始发散。
2. 阈值本身并不难达到：anchor 的分数分布里 96.22% 的候选能过 0.828。
3. 真正发生的是**闭环崩溃**：拒绝已有 ID → 轨迹状态被删除/回收 → 后续分数趋于均匀（≈1/9）→ 更多拒绝 → 每个检测都新建轨迹。

**结论（A/B）**：

- “armlite 证明离线头不能迁移”这一结论被**撤回**。
- 阈值不是普通调参：它会改变后续历史分布，把系统推入另一个吸引子。
- **约束**：不能先把 R1/R3 当成阈值问题；任何阈值改动都必须做闭环审计（accepted_existing、below_threshold、tracks_created、id lifetime）。

---

### 3.5 E13 / E14 / E16 / E17 — 诊断数据与推理侧 rerank

#### E13 — 训练侧特征 dump

- 对 40 个 train 序列导出特征与分数，用于离线诊断。
- 没有模型训练指标；价值在于让 E17 的 pre-flight 可以在 **train only** 上完成，不碰 val gt。

#### E14 — 解耦 rerank（写回轨迹状态）

- 方案：`log score + α·IoU` 后处理，并把结果写回 tracker 状态。
- 结果：损失 0.5–2.2 HOTA。
- 复盘估计：至少 **97.8%** 的损失来自状态反馈，而不是排序本身。

#### E16 — label-only rerank

- 只重写输出 ID，不回写状态。
- 结果：仍为负收益。

**结论（A）**：推理侧 rerank 路线关闭。离线排序信号即使有判别力，也不能直接转化为闭环收益。

#### E17 — 几何进入分数路径（SPEC，未实现）

**Pre-flight（train only，40 序列）**：

| 量 | 值 |
|---|---|
| rank-1 vs rank-2 对数 | 344,653 |
| 可翻转池（GT 持有者在 rank 2） | 2,337（0.678%） |
| `iou_d` 单变量 AUC | 0.974 |
| `disp_d` 单变量 AUC | 0.033（反向 0.967） |
| `logit_gap`（现有分数路径已知） | 0.896 |
| OOF AUC，仅分数 | 0.8954 |
| OOF AUC，分数 + 几何 | **0.9791（+0.0838）** |
| 每序列 AUC | min 0.732，median 0.982 |

**结论（B/C）**：

- 几何确实携带强判别信息。
- 但 E16 已证明“离线有信息”不等于“闭环有收益”。
- 该 SPEC 明确未实现，不能当成已验证修复。

---

### 3.6 E18 — 消融 E5 的三个 CRR 训练机制

**目的**：E5 的配方里有三个官方 MOTIP 没有的机制：

1. history blackout 0.6（其中 25% 是 prefix mask）；
2. 结构化 history corruption 0.15→0.05；
3. NEW class weight 1.5。

E18 把三项全部关闭，其余保持与 E5 一致（同一 checkpoint、同一 7 epoch 课程、同一 LR/milestones、switch/occlusion 仍 0.5）。

**配方**：

- 从 `crr_e5_e2e_train40_epoch12_seed42_v1/checkpoint_4.pth` 续训。
- 7 epoch，LR 1e-4，warmup 0，milestones [3, 6]，gamma 0.2。
- NEW weight 1.0，blackout/prefix/corruption 全 0。

**逐 epoch 结果**：

| ep | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.692 | 72.002 | 46.580 | 51.023 | 75.903 | 58.661 | 8542 | 5158 | 9077 |
| 1 | 57.663 | 72.371 | 46.228 | 51.021 | 75.136 | 59.771 | 8355 | 5312 | 8738 |
| 2 | 55.335 | 70.551 | 43.717 | 48.995 | 73.254 | 57.197 | 4576 | 6048 | **4001** |
| 3 | 61.155 | 73.906 | 50.932 | 55.481 | 78.007 | 63.092 | 4936 | 4799 | 4595 |
| 4 | 61.109 | 73.757 | 50.975 | 55.365 | 78.210 | 63.093 | 4329 | 4644 | 4137 |
| 5 | 61.153 | 74.592 | 50.411 | 55.039 | 77.741 | 62.841 | 5307 | 4731 | 4731 |
| 6 | **62.299** | 74.584 | 52.348 | 56.717 | 78.482 | 63.933 | **2886** | 4477 | **2311** |

**对照（同 ep6）**：

| 实验 | HOTA | IDs | IDSW |
|---|---:|---:|---:|
| E5 | 62.273 | 12499 | 12248 |
| E18 ABL | **62.299** | **2311** | **2886** |

**结论（A）**：

- 三个 CRR 机制不是达到 E5 水平的必要条件；关闭后 HOTA 持平，IDs 反而低一个数量级。
- ep2 仍出现一次明显回落（HOTA 55.335、IDs 4001），说明冷启动/相位调整仍存在。
- 三项是一起关闭的，**不能**单独归因 NEW weight 或某一项。
- E18 配方（官方增强 + 三项 OFF + NEW weight 1.0）成为后续 E19 系列的干净起点。
- E18 也证明：**switch=0.5 并非对所有 CRR 变体必然致命**。这与 E19c 的记录并不矛盾——E19c 的风险组合是 “switch 0.5 + amax/单点极值”，E18 的 pointer 聚合会稀释污染。

---

### 3.7 E19 — bounded NEW prototype 冷启动失败

**目的**：在 E18 配方基础上，从零训练 CRR，修复 E18 诊断出的 NEW 几何不对称：实体分数是 `cos × sqrt(d)` 的有界分数，而 NEW 是**无界线性 logit**。

**改动**：NEW logit 改为与实体 logits 同几何的**归一化可学习 prototype**（cosine × sqrt(d)）。

**结果**：

| ep | HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 51.461 | 72.329 | 36.930 | 40.973 | 75.625 | 51.721 | 30440 | 5417 | 30298 |
| 1 | 48.306 | 72.278 | 32.570 | 35.610 | 79.275 | 46.738 | 40910 | 5167 | 43552 |
| 2 | 51.729 | 73.021 | 36.896 | 40.705 | 78.038 | 51.169 | 32197 | 5650 | 33466 |
| 3 | 51.636 | 71.253 | 37.654 | 40.557 | 83.202 | 50.587 | 43227 | 6135 | 43890 |

**判定（A）**：远低于预注册门槛（峰值 HOTA ≥ 63.5 且 IDs ≤ 2500）。没有出现 E5o 式相变。

**机制解释（B）**：

```
关联不足 → 更多 NEW 判决 → 地址池膨胀 → softmax 稀释
→ 正确地址分数落到 ID_THRESH 0.2 以下 → 更多 NEW
```

这是“冷启动自锁”，表型是 IDs 爆炸，不等同于 E15 的“分数均匀盆地”。

**约束**：NEW 几何是 CRR 特有的风险点；但 E19 同时改了架构和 NEW head，不能单独证明 bounded prototype 是唯一根因。

---

### 3.8 E19b — 只换官方无界线性 NEW head

**目的**：E19 的 bounded NEW prototype 是否是冷启动失败的 cause？把 NEW head 回滚为官方无界线性 logit，其余不变（源码与 E18 完全一致，零代码改动）。

**结果（ep0）**：

| HOTA | DetA | AssA | AssRe | AssPr | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.348 | 73.423 | 34.748 | 38.241 | 80.669 | 48.422 | 48472 | 5292 | **49175** |

**结论（A）**：

- 只换 NEW head **没有**解除冷启动；IDs 甚至更高。
- E19 的失败不能只归因于 bounded prototype；但从零训练的 CRR 在官方 recipe 下确实存在更深层问题。
- **约束**：不要重复“只换 NEW head”这一思路（这是 R1 修复时的重要先验）。

---

### 3.9 E19c — 三项修复包

**目的**：在 E19b 基础上，一次加入三个修复：

1. `AUG_TRAJECTORY_SWITCH_PROB 0.5 → 0.0`；
2. 解除 reliability / residual / id_gate 的零初始化休眠；
3. 加入可学习 logit scale `s`，`mention_scores = (cos × sqrt(d)) × s`，init 1.0。

**为什么关 switch（B）**：

- 官方 switch 是逐帧、逐槽的 Bernoulli 重排，只交换轨迹特征索引和 mask，不交换 ID 标签。
- 官方 decoder 的多层软注意力会平均掉污染帧；CRR 的 `base_max`（amax）是单点极值，**一个污染帧就能把 query 的自相似证据放进错误实体列**。
- E4 曾观察到标签级污染对外观 pointer 的严重破坏；冻结期实验均用 switch 0.0。
- 因此 E19c 关闭 switch，保留 occlusion。

**结果**：

| ep | HOTA | DetA | AssA | AssRe | AssPr | MOTA | IDF1 | IDSW | Frag | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.469 | 74.484 | 37.237 | 40.305 | 76.921 | 85.162 | 51.689 | 2623 | 4763 | **1246** |
| 1 | 48.432 | 73.748 | 32.162 | 34.332 | 79.770 | 82.765 | 44.989 | 未记录 | 未记录 | 未记录 |

**训练状态**：

- ep0、ep1 完成评估；ep2 训练到 22,280 / 40,636 后停止（2026-10-26 18:49）。
- 日志末：loss 2.1724（均值 2.2496），detr 2.1030（2.1783），id_loss 0.0695（0.0713）。
- ep2 没有评估，因此没有“最终收敛”证据。

**结论（A/B）**：

- ep0 IDs 从 E19b 的 49175 降到 1246，冷启动表型显著改善。
- 但 ep1 HOTA 回落到 48.432、AssA 32.162、IDF1 44.989，说明尚未进入稳定盆地。
- 三项混杂（switch、零初始化、logit scale），**不能**把收益全部归给 switch。
- 预注册判据（峰值 HOTA ≥ 63.5 且 IDs ≤ 2500）**未达到，也未证伪**，因为训练没有跑完。

**对 R1/R3 的直接约束**：

- E19c 明确记录：`amax` 对单帧 switch 污染最敏感。
- R1 v3 / R3 若同时使用 `amax` + switch 0.5，就处在历史标记的高风险区。
- 但 E18 也表明 switch 0.5 不是对所有 CRR 变体都致命，必须做本实验的单变量 A/B，不能照搬。

---

### 3.10 E19d — context-aware NEW head（未验证）

- 设计：NEW 分数相对最佳已有实体、margin、不确定性、轨迹年龄与集合上下文。
- 输出目录 `crr_e19d_symctx_newhead_seed42_v1` 只有启动脚本/配置，没有训练输出。
- **判定：D（未验证）**。不能当成历史已证明有效的修复。

---

### 3.11 R1 — 用 EquivariantSetDecoder 替换官方 ID Decoder

#### 目的与设计

R1 的目标是**最干净的单变量对照**：数据、增强、优化器、LR schedule、推理阈值、seed 全部沿用官方 DanceTrack 配方，只把官方 ID Decoder 换成 R1 的等变集合解码器。配套单测 `test_r1_config_only_changes_architecture_and_runtime` 强制“只许改架构和运行时”。

R1 解码器：

- 历史 mention 按 ID 地址做因果注意力池化，生成实体 token；
- 当前检测 query 对**单个历史 mention** 做带相对时间偏置的因果交叉注意力；
- 读出用 `query · entity` 点积，而不是 ID 专属分类权重；
- 额外一个可学习 NEW token，形成“50 个 ID 地址 + 1 个 NEW”的读出。

#### 版本与状态

| 版本 | 目录 | 状态 |
|---|---|---|
| v1（非 mention） | `r1_equivariant_set_decoder_seed42_v1` | 跑到 3220 步，RC=-1，无 eval |
| mention v1 | `r1_equivariant_set_decoder_mention_seed42_v1` | 跑到 24000/40636，id_loss 2.35→≈0.50，中断 |
| mention v2 | `r1_equivariant_set_decoder_mention_seed42_v2` | 10-27 18:31 从零重启，完成 ep0 |
| mention v3 | `r1_equivariant_set_decoder_mention_readout_seed42_v3` | direct mention/amax + switch 0.5，跑到 12100 步 |

#### R1 v2 ep0 结果（A）

| HOTA | DetA | AssA | AssRe | AssPr | MOTA | IDF1 | IDSW | IDs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41.726 | 71.508 | 24.577 | 27.142 | 66.225 | 76.963 | 37.718 | 11195 | 2179 |

对照 E19c ep0：HOTA 52.469、DetA 74.484、AssA 37.237、IDF1 51.689、IDs 1246。

**差距集中在关联，不在检测**：DetA 只差 2.976，AssA 差 12.660，IDF1 差 13.971。

#### Assignment audit（A，同验证集、同 object-max）

| 量 | E19c | R1 v2 |
|---|---:|---:|
| objects_scored | 215,316 | 229,025 |
| accepted_existing | 209,084 | 222,147 |
| **new_wins** | **613** | **10** |
| conflicts | 5,618 | 6,574 |
| below_threshold | 1 | 289 |
| tracks_created | 1,246 | 2,179 |
| NEW median score | 4.56e-05 | **1.74e-08** |
| existing median score | — | ≈0.965 |

**解释（B）**：

- R1 不是“不会接受已有 ID”——它接受了 222,147 次。
- 真正异常的是 NEW 分支几乎失去竞争能力：主动 NEW 只有 10 次。
- 大量新轨迹来自：冲突失败 → 被动转 NEW；已有 ID 分数低于阈值 → 新建。
- 这是 NEW 与已有实体的**相对校准崩坏**，不是整个 softmax 不会预测。

#### 根因排序（B/C）

1. **NEW 分支特异性失能**（B，已由 audit 坐实表型）。
2. **v2 的 query 无关历史预汇聚**（B）：`_build_entity_tokens` 的池化权重只由历史外观和时间决定，不看当前 query；同一个 ID 对所有 query 只有一个平均向量，会削弱甚至抵消最佳证据。
3. **switch=0.5 重新引入**（B）：与 amax 组合，形成污染放大器。
4. **训练—推理错位**（B）：训练是逐 query CE，推理是 object-max 一对一仲裁；训练没有优化“同一 ID 不应被多个 query 同时占用”。
5. **NEW 监督过弱**（B）：NEW 的 GT 比例约 0.23%，`new_class_weight = 1.0`，平均 loss 被已有 ID 主导。

#### 混杂因素

- **数据 split 不同**：E19c 用 `motip_official_train40_v1`，R1 用 `train`。因此 R1 与 E19c 同时改变了 decoder、switch、split。
- **源码没有冻结**：R1 v2 只冻结配置和启动器，`GIT_VERSION: null`，没有提交号；v2 实现只能从会话记录恢复。

#### R1 v3

- 改回 direct query–mention 打分 + 按 ID `amax`，恢复 switch 0.5。
- 跑到 ep0 step 12100，累计 id_loss 1.2926（step 12000 瞬时 1.0538）。
- 无 eval；RC=-1 外部终止。
- **判定**：v3 不是根因修复的完整验证；它只修了“query 无关预汇聚”，没有解决 NEW 校准、历史污染、训练—推理错位。

---

### 3.12 R3 — 更深的等变集合解码器

#### 目的

在 R1 的基础上增加容量与实体证据：用六层 query–history 交叉注意力 + E19c 风格的实体统计残差 + GRU 身份记忆，试图同时获得“深度表示”和“直接实体证据”。

#### 修改前设计（已训练到 step 12100）

| 维度 | 实现 |
|---|---|
| 关联主路径 | 六层 query–history cross-attention，随机初始化 |
| query 看历史 | 先混合全部因果历史，再按 ID 做 readout |
| 实体评分 | 点积 + mention `amax`，再叠加 8 维实体统计残差 |
| 实体边界 | cross-attention 不知道 ID，最后 scatter 才使用 |
| NEW | `new_token` 同时参与注意力上下文和 NEW 评分 |
| 分数尺度 | 原始点积 × `hidden_dim^-0.5`（固定） |
| 身份记忆 | 每实体因果 GRU，tanh 门控 |
| switch | **0.5** |
| 中间层 ID 监督 | **关闭**（`USE_AUX_LOSS=False`） |

#### 修改前证据（B）

| step | R3 id_loss | E19c id_loss |
|---:|---:|---:|
| 1,000 | 1.9166 | 1.7940 |
| 5,000 | 1.2198 | 0.2000 |
| 10,000 | 1.1397 | 0.1282 |
| 11,000 | — | 0.1005 |

- R3 没有完整 ep0 评估，不能把 R2 的 41.726 归给 R3。
- 同阶段 DETR loss 与 E19c 相近（≈1.94 vs ≈2.0），说明检测器不是首先崩溃的部分。
- 训练停在 ep0 step 12100（2026-10-29 10:12），RC=-1，无训练侧 Python traceback；随后官方复现于 11:09 启动。

#### 根因排序（B）

1. **switch=0.5 × mention amax = 污染放大器**：一个错误但高分的 mention 就能决定整个实体分数。
2. **六层 decoder 增加优化难度，却没有中间层 ID 监督**：模型要同时学会上下文提取、身份信息保留、实体可分、NEW 校准。
3. **cross-attention 混合实体，最终 readout 又丢掉直接证据**：层数增加，但做身份判断时可直接使用的信息反而更少。
4. **NEW 耦合与固定缩放**：NEW token 同时做上下文和 readout；固定 `d^-0.5` 缺少尺度自由度。次级风险，尚无 R3 的 NEW margin 证据。

#### 修改后设计（当前代码，未训练）

2026-10-29 按确认方案落地，共七项：

**（1）关闭 switch**：`AUG_TRAJECTORY_SWITCH_PROB: 0.5 → 0.0`，occlusion 仍 0.5。

**（2）启用中间层辅助 ID 监督**：`USE_AUX_LOSS: true`；每层 logits 都进入 `IDCriterion`。

**（3）实体证据直接参与评分**：保留点积 + `amax` 作为 base，叠加 8 维实体统计残差（时间、历史支持数、可靠性），并按实体因果历史运行 GRU 身份记忆，经门控融合；NEW 列不被历史残差改写。

**（4）NEW 独立校准**：

```python
new_base = new_query_head(query_proj(current))      # 零初始化
context  = logmeanexp(entity_scores)                # 有效实体列
new_score = new_base - sigmoid(new_context_logit) * context
```

- `new_context_logit` 初始化为 `log(1/3)`，即 `sigmoid = 0.25`。
- NEW head 读取的是 **decoder 之前的 `query_proj` 输出**，不是已经被历史混合过的 query。
- 目的：让 NEW 与已有实体共享“当前检测证据”，同时用最佳已有实体分数做负反馈。

**（5）同帧冲突抑制**：

```python
column_max = max_over_queries(entity_scores)        # 每帧、每实体列
gap        = clamp(column_max - score, min=0)
penalty    = sigmoid(conflict_logit) * gap
score      = score - penalty                        # 列最大值不变
```

- `conflict_logit` 初始化为 `log(1/3)`，即惩罚强度 0.25。
- 只在同一帧内沿 query 维竞争，不跨帧；无效 query / 空实体列不参与。
- 目的：给 object-max 仲裁中的落败 query 提供次优实体的学习压力，同时保持实体重命名等变。

**（6）Query/History 特征对称性**：

- `TrajectoryModeling` 现在对 `trajectory_features` 与 `unknown_features` 应用**同一套**逐 token 变换：`adapter → norm → FFN → norm`。
- 历史 prototype 混合（causal / symmetric）仍只作用于历史，不改变当前 query。
- 目的：消除“历史 token 经过额外变换、当前 query 没有”的结构不对称。

**（7）保留并接线**：GRU 身份记忆、8 维实体统计残差、`amax` base、六层辅助输出。

**测试（A）**：

- `tests/test_equivariant_set_decoder.py` + `tests/test_r3_entity_refinement_v1.py` + `tests/test_r3_new_calibration_conflicts_symmetry.py`：**30/30 通过**。
- 新增 11 项用例覆盖：NEW 使用实体 logmeanexp 负反馈、无证据时上下文为 0、NEW head 读 pre-decoder query、列最大值不变、逐帧独立、padding 不参与、重命名等变、梯度可达；Query/History 共享变换、query 变换逐点不混合、query 路径有梯度。
- 全量本地测试 33 项，32 通过、1 项错误：`test_periodic_checkpoint_records_completed_step_and_epoch_end_resets_position` 调用 `train_one_epoch` 缺少 `detr_criterion_batch_len`。这是既有签名不同步，与本次修改无关；未修改远端训练代码来“凑绿”。

**状态**：**未训练**。现有 R3 日志全部属于修改前版本；不能把修改前的 id_loss 或任何指标当作修改后结果。

---

### 3.13 官方 MOTIP A6000 复现

#### 目的

此前的 64.099 是在 RTX 4090 上复现的。为排除硬件/环境差异，按官方配置在 RTX A6000 上完整重跑，作为唯一有效的官方基线。

#### 配置

| 项 | 值 |
|---|---|
| 源码 | `source/MOTIP_official_a6000_v1` |
| 数据 | DanceTrack train |
| 预训练 | `r50_deformable_detr_coco_dancetrack.pth`（仅 DETR） |
| 初始化 | 从零，不 resume |
| Epochs | 10 |
| Steps/epoch | 40,636 |
| Batch size | 1 |
| LR | 1e-4，warmup 1 epoch |
| Milestones | [6, 9]，gamma 0.1 |
| Weight decay | 5e-4 |
| Max clip norm | 0.1 |
| Seed | 42 |
| switch / occlusion | 0.5 / 0.5 |
| USE_AUX_LOSS | false |
| Eval | 每 epoch 官方 val，object-max 协议 |

#### 运行时状态（2026-10-29 23:04，只读观测）

| 项 | 值 |
|---|---|
| 运行目录 | `official_motip_dancetrack_a6000_seed42_v2` |
| 启动时间 | 2026-10-29 11:09 |
| 进度 | epoch 0，step **18,760 / 40,636（46.2%）** |
| 速度 | 2.35 s/step |
| 到 epoch 结束 | ≈14 小时 16 分 |
| loss | 2.5014（累计均值 4.0911） |
| detr_loss | 1.8946（1.9602） |
| id_loss | 0.6068（2.1309） |
| GPU | RTX A6000，利用率 53%，22,904 / 49,140 MiB，85°C |
| 异常 | 日志尾部无 Traceback / OOM / NaN |

> 注意：GPU 利用率 53% 是瞬时采样，训练进程仍在；不要据此判断卡死。

#### 与 R3 的损失对照（修改前 R3，B）

| Step | R3 总 loss | R3 DETR | R3 ID | 官方总 loss | 官方 DETR | 官方 ID |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 3.6950 | 1.7784 | 1.9166 | 5.6355 | 1.7185 | 3.9170 |
| 5,000 | 3.1985 | 1.9787 | 1.2198 | 4.6937 | 1.9776 | 2.7161 |
| 10,000 | 3.2479 | 2.1082 | 1.1397 | 4.2270 | 1.9120 | 2.3151 |
| 12,000 | 2.9040 | 1.8502 | 1.0538 | 3.3443 | 1.9127 | 1.4316 |

**解读（B）**：

- 两者都打印 `detr_loss + id_loss × 1.0` 的总损失。
- R3 总损失更低，主要因为 ID 分支参数化不同：R3 的 `amax`/点积读出头与官方 ID 词典分类头的 logit 尺度、类别数、监督密度都不同，**绝对 loss 不可直接比较**。
- DETR 部分接近（≈1.9），说明检测器收敛状态相近。
- 判断关联好坏应看同阶段下降趋势、NEW margin、accepted/new/conflict 比例，而不是总 loss 大小。

---

## 4. 横向结论

### 4.1 已关闭的路线

| 路线 | 证据 | 结论 |
|---|---|---|
| 未校准运动/几何加性 residual | E6 | 关闭 |
| 推理侧 IoU/listwise rerank 写回状态 | E14 | 关闭（≥97.8% 损失来自状态反馈） |
| 只重写输出 ID 的 rerank | E16 | 关闭 |
| 单纯调 `ID_THRESH` / `NEWBORN_THRESH` | E15 | 关闭（闭环崩溃风险） |
| 只换 NEW head | E19b | 不足以修复冷启动 |
| 把离线几何信号直接搬进闭环 | E17 + E16 | 无收益证据，不优先 |
| 在混杂优化器状态下端到端微调后归因新模块 | E6 | 方法学禁止 |

### 4.2 已确认的机制（B）

1. **冷启动存在相变**：E5o IDs 28217→9715→7626（ep2）；修复版必须在 ep2–3 出现数量级下降，否则不能声称解决冷启动。
2. **switch 污染对 `amax` 尤其危险**：E19c 记录 + `transforms.py` 只换索引不换标签；E18 表明对软聚合 pointer 不致命，因此是“组合风险”，不是普适毒药。
3. **NEW 校准是独立故障面**：R1 v2 的 `new_wins=10` vs E19c `613`，且 NEW 中位分数低三个数量级。
4. **闭环会放大局部决策**：E15 证明阈值拒绝会改写后续历史，导致分数均匀、轨迹爆炸。
5. **训练—推理错位普遍存在**：逐 query CE 训练 + object-max 一对一仲裁；E19c 也如此，但 R1/R3 的更弱 readout 会放大它。

### 4.3 未决问题

| 问题 | 现状 | 需要的证据 |
|---|---|---|
| E19c 的收益有多少来自 switch？ | 三项混杂 | 固定 checkpoint 的 clean/switch A/B |
| R3 修改后能否达到 E19c ep0 水平？ | 未训练 | 从零一个 epoch 的 eval |
| 六层 decoder 是否必要？ | 只有修改前 id_loss 证据 | 层数单变量消融 |
| NEW 校准的 0.25 初值是否合适？ | 仅测试通过 | 训练后 NEW margin / new_wins |
| 同帧冲突抑制是否真的减少冲突？ | 仅单测 | conflict 计数 + object-max 仲裁统计 |
| Query/History 对称性是否是缺陷？ | 事实成立，未证实有害 | 单变量 A/B |

---

## 5. 当前状态（2026-10-29 23:05）

| 项 | 状态 |
|---|---|
| 官方 A6000 复现 v2 | **运行中**，ep0 step 18,760/40,636 |
| R3 修改后代码 | 已落地，30/30 针对性测试通过，**未训练** |
| R3 修改前训练 | 停在 ep0 step 12,100，RC=-1，无 eval |
| E19c | 停在 ep2 step 22,280，ep0/ep1 有 eval，ep2 无 eval |
| R1 v2 | ep0 完成，HOTA 41.726；未继续 |
| R1 v3 | 停在 ep0 step 12,100，无 eval |
| GPU | RTX A6000，官方复现占用约 22.9 GB |

**原则**：在官方复现完成前，不动它的进程、目录和配置；R3 的下一步训练必须等 GPU 空闲或用户明确安排。

---

## 6. 下一轮实验队列

### P1 — 完成官方复现（进行中）

- 目标：10 epoch 完整跑完，得到 A6000 上的官方 HOTA/AssA/IDF1/IDs。
- 用途：唯一有效官方基线；校准此前 4090 的 64.099。
- 禁止：中途改配置、改数据、插入评估。

### P2 — R3 修改后从零训练一个完整 epoch

**目的**：验证七项修复是否把 R3 从“id_loss 停滞、NEW 失能”拉回 E19c ep0 水平。

**冻结前必须写入配置**：

- 源码：当前 `MOTIP_crr_e19c_cleanscratch_v1`；
- 数据：与 E19c 相同的 `motip_official_train40_v1`（不再用 R1 的 `train`，消除 split 混杂）；
- seed 42、官方 LR/schedule、从零、只加载 DETR 预训练；
- switch 0、occlusion 0.5、aux loss on、NEW 校准/冲突抑制/对称性 on。

**ep0 必读指标**：

- HOTA / DetA / AssA / AssRe / AssPr / IDF1 / IDSW / IDs；
- `new_wins`、`accepted_existing`、`conflicts`、`below_threshold`、`tracks_created`；
- NEW 中位分数、已有 ID 中位分数、best-existing margin。

**建议判据（启动前冻结，避免事后解释）**：

| ep0 表现 | 判定 | 动作 |
|---|---|---|
| HOTA ≥ 52.0 且 IDs ≤ 1500 | 修复有效 | 继续跑到 ≥3 epoch |
| HOTA 48–52 或 IDs 1500–5000 | 部分有效 | 先看 NEW/conflict 审计，再决定 |
| HOTA < 48 或 IDs > 10000 | 修复不足 | 停止，回到 readout/优化问题 |

> 这些阈值是建议，不是历史预注册；必须在训练启动前写进实验配置，避免移动靶。

### P3 — 单变量消融（仅在 P2 有效后）

顺序：

1. `USE_AUX_LOSS` on/off；
2. NEW 校准 on/off（含初值 0.25 vs 其他）；
3. 同帧冲突抑制 on/off；
4. Query/History 对称性 on/off；
5. decoder 层数（6 vs 3 vs 1）。

每次只改一项，其余冻结；至少看 3 个 epoch 的峰值，不看尾点。

### P4 — 暂不优先

- 任何 IoU/listwise 后处理 rerank；
- 单纯阈值搜索；
- 未校准几何/运动加性项；
- 在官方复现未完成时比较不同硬件的绝对指标。

---

## 7. 复现索引

### 7.1 服务器路径

| 内容 | 路径 |
|---|---|
| 当前源码 | `G:\mot_win\New_MOT_v2\motip_paradigm_reset\source\MOTIP_crr_e19c_cleanscratch_v1` |
| 官方复现源码 | `...\source\MOTIP_official_a6000_v1` |
| 官方复现输出 | `...\official_motip_dancetrack_a6000_seed42_v2` |
| R3 修改前输出 | `...\r1_equivariant_set_decoder_mention_readout_seed42_v3` |
| R1 v2 输出 | `...\r1_equivariant_set_decoder_mention_seed42_v2` |
| E19c 输出 | `...\crr_e19c_cleanscratch_fixpkg_seed42_v1` |
| 旧归档根 | `G:\mot_stage\motip_paradigm_reset` |

### 7.2 关键结果文件

| 实验 | 文件 |
|---|---|
| E5/E18 | `crr_e18_ablation_cont7_seed42_v1\E18_AB_RESULTS.txt` |
| E19 | `crr_e19_cleanscratch_seed42_v1\E19_CS_RESULTS.txt` |
| E19b | `crr_e19b_cleanscratch_officialnew_seed42_v1\E19B_CS_RESULTS.txt` |
| E19c | `crr_e19c_cleanscratch_fixpkg_seed42_v1\E19C_CS_RESULTS.txt` |
| E15 | `crr_e15_armlite_postmortem_v1\FINDINGS.md` |
| E17 | `crr_e17_geom_scorepath_v1\CRR_E17_SPEC.md` |

### 7.3 本地快照

| 内容 | 路径 |
|---|---|
| 最新源码快照 | `C:\Users\24138\Desktop\第一版工作\MOTIP_crr_r3_latest_20261029` |
| 传输归档（2026-10-29 22:49 快照，不含本日志） | `C:\Users\24138\Desktop\第一版工作\MOTIP_crr_r3_latest_20261029.tar.gz` |
| 归档 SHA-256 | `b685fd9539da7632674c48432a8cf8668f73417d800be92e8f1a35f8536db816` |
| 历史证据包 | `C:\Users\24138\Desktop\第一版工作\r3_update_work\history_sources` |

### 7.4 关键代码位置（当前快照）

| 内容 | 位置 |
|---|---|
| NEW head / 校准参数 | `models/motip/equivariant_set_decoder.py:90` |
| NEW head 零初始化 | `models/motip/equivariant_set_decoder.py:121` |
| 同帧冲突抑制 | `models/motip/equivariant_set_decoder.py:365` |
| NEW 校准 | `models/motip/equivariant_set_decoder.py:380` |
| Query/History 共享变换 | `models/motip/trajectory_modeling.py:62` |
| R3 配置 | `configs/r1_equivariant_set_decoder_v1.yaml:10` |
| 总损失聚合 | `train.py:540` |

---

## 8. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-10-29 | 首次整理：E5–E19d、R1/R3、官方 A6000 复现；新增七项修复记录与当前状态 |

---

## 附录 A：实验配方速查

| 实验 | 初始化 | switch | CRR 三机制 | NEW head | AUX | 关键目的 |
|---|---|---|---|---|---|---|
| E5 | E5 epoch12 ckpt4 | 0.5 | ON | — | — | 可用基线 |
| E18 | E5 epoch12 ckpt4 | 0.5 | OFF | — | — | 消融三机制 |
| E19 | 从零 | 0.5 | OFF | bounded prototype | — | 冷启动 + NEW 几何 |
| E19b | 从零 | 0.5 | OFF | 官方线性 | — | 隔离 NEW head |
| E19c | 从零 | **0.0** | OFF | 官方线性 | — | 三项修复包 |
| R1 v2 | 从零 | 0.5 | — | 可学习 NEW token | — | 等变 decoder 单变量 |
| R1 v3 | 从零 | 0.5 | — | 可学习 NEW token | — | direct mention/amax |
| R3 修改前 | 从零 | 0.5 | — | NEW token 耦合 | OFF | 深层 decoder |
| R3 修改后 | 从零 | **0.0** | — | **独立校准** | **ON** | 七项修复 |

## 附录 B：证据边界

1. **R1 v2 与 E19c 不可直接比较**：split 不同（`train` vs `motip_official_train40_v1`），且 R1 源码未冻结。
2. **E19c 三项修复混杂**：不能把 ep0 ID 改善全部归因 switch。
3. **R3 修改前无 eval**：所有“R3 比 E19c 差”的说法只能基于 id_loss 与结构分析，不能基于 HOTA。
4. **R3 修改后未训练**：当前代码的七项修复只有单测证据，没有精度证据。
5. **官方复现未完成**：64.099 仍是 4090 旧记录；A6000 结果出来前，不应把它当作新硬件锚点。
6. **损失不可跨头比较**：R3 六层辅助 CE 的均值、E19c 单层 pointer CE、官方 ID 词典 CE 是三种不同参数化；只有 DETR loss 可横向看趋势。
