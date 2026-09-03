# 复现结果对照表

> 记录正式训练结果，与论文 Table 1 对照。**口径必须一致**：论文用最后 10 次评估的平均 return（每 5000 步评估一次、确定性策略），我们同样取最后 10 次 `evaluation/return` 的均值。

## 环境与配置
- 训练机 GPU：`待填（nvidia-smi）`
- 驱动 / CUDA：`待填`
- 依赖锁定文件：`STUDY/env-lock/linux-*.txt`
- wandb 项目：`rlpd_locomotion`

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