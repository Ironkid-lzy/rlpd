# 复现结果对照表

> 记录正式训练结果，与论文 Table 1 对照。**口径必须一致**：论文用最后 10 次评估的平均 return（每 5000 步评估一次、确定性策略），我们同样取最后 10 次 `evaluation/return` 的均值。

## 环境与配置
- 训练机 GPU：`NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell sm_120), 8 GB`
- 驱动 / CUDA：`570.211.01 / CUDA 12.8`
- 依赖锁定文件：`STUDY/env-lock/linux-2026-09-13.txt`
- wandb 项目：`rlpd_locomotion`（正式复现） / `rlpd_ablation`（R1 消融） / `rlpd_r2`（R2 数据质量阶梯）
- ⚠️ 2026-09-15 发现并修复了"同一 seed 不可复现"的缺陷（两个漏种的 RNG），
  详见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。**R1 是修复前跑的，其误差棒偏乐观。**

## 消融实验报告

- **R1 多 seed 消融（5 seeds × A/B。
  - ⚠️ 报告顶部的勘误：其中"medium 末点与 expert 持平"这一条已被 R2 推翻。
- **R2 数据质量阶梯（5 seeds × 5 组，halfcheetah，100k 步）**：见 [`REPORT-R2.md`](REPORT-R2.md)
  - 原始数据：`results/r2/r2_5seed_eval_return.csv`、`results/r2/r2_vs_r1_prefix25k.txt`
  - 一句话结论：100k 下数据质量阶梯严格单调
    （expert 11447 > medium 10118 > medium-replay 9358 > random 8946 > 纯在线 7611）；
    但 expert 数据在 10–20k 区间反而最差，交叉点约在 21k
  - 一句话结论：离线数据与高 UTD 都显著提升样本效率；medium 数据前期优于 expert（约 2.3×）、
    末点持平（差 3% 以内），且跨种子稳定性远好于 expert。

## 对照表

| 任务 | 论文 Table 1（均值±std） | 本复现 seed=42 | 本复现 seed=1 | 本复现 seed=2 | 备注 |
|---|---|---|---|---|---|
| halfcheetah-expert-v2 | `待填（M1 从论文抽取）` | | | | |
| walker2d-expert-v2 | `待填` | | | | |
| hopper-expert-v2 | `待填` | | | | |

## 单次运行记录
- 命令（含全部 flag）：
- 起止时间 / 时长：
- wandb run 链接：
- 关键曲线：training/critic_loss、training/q、evaluation/return
- 与论文差异说明（如有）：
