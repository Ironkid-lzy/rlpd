# 论文笔记：Efficient Online RL with Offline Data (RLPD)

> arXiv: 2302.02948。本文件用于记录论文阅读要点与复现对照数字。
> 风格：散文式，见 `notes/TEMPLATE.md`。

## 一句话
RLPD = 用离线数据集（prior data）加速在线 RL：每步训练时把 offline batch 与 online batch 交错混合，配合大 ensemble critic + LayerNorm，替代 CQL 式保守项，达到"少样本高效在线学习"。

## 摘要（已读，2026-09-03）
- 作者：Philip J. Ball, Laura Smith, Ilya Kostrikov, Sergey Levine（2023-02-06, cs.LG/cs.AI）
- 问题：在线 RL 的样本效率与探索仍是主要挑战；引入 offline data（人类专家轨迹或次优探索策略的轨迹）是常用手段，但此前方法依赖大量修改与额外复杂度。
- 核心问题：能否**直接套用现有 off-policy 方法**来利用 offline data 做在线学习？
- 答案：可以，但需要一组**最小但关键**的改动才能获得可靠性能。
- 结果：正确应用这些简单建议，在多个竞争性 benchmark 上比现有方法提升 **2.5×**，且**无额外计算开销**。
- 代码：https://github.com/ikostrikov/rlpd（即本仓库）

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
