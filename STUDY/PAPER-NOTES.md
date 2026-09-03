# 论文笔记：Efficient Online RL with Offline Data (RLPD)

> arXiv: 2302.02948。本文件用于记录论文阅读要点与复现对照数字。
> 风格：散文式，见 `notes/TEMPLATE.md`。

## 一句话
RLPD = 用离线数据集（prior data）加速在线 RL：每步训练时把 offline batch 与 online batch 交错混合，配合大 ensemble critic + LayerNorm，替代 CQL 式保守项，达到"少样本高效在线学习"。

## 核心机制（待 M1-M4 精读后补全）
- offline/online 混合：`offline_ratio=0.5`，每步各抽半 batch 交错
- 大 ensemble：`num_qs=10`，`num_min_qs=2`（antmaze/像素为 1）
- critic LayerNorm
- `backup_entropy` 默认 True
- 高 UTD ratio（`utd_ratio=20`）：每 env step 更新 20 次

## Table 1 基线数字（复现对照口径）
> 从论文抽取。**口径**：最后 10 次评估的平均 return（确定性策略，每 5000 步评估）。

| 任务 | RLPD（均值±std） | 备注 |
|---|---|---|
| halfcheetah-expert-v2 | `待填` | |
| walker2d-expert-v2 | `待填` | |
| hopper-expert-v2 | `待填` | |
| antmaze-umaze-v2 | `待填` | |

## 阅读进度
- [ ] 摘要 + 引言
- [ ] 方法（offline/online 混合、ensemble、UTD）
- [ ] 实验（Table 1 口径确认）
- [ ] 与 CQL / IQL / DrQ 的关系