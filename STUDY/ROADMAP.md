# RLPD 学习与复现 ROADMAP

> 本分支（`study-rlpd`）用于：读源码 + 写笔记 + 打补丁/练习；正式训练在 Linux 游戏本上跑。
> 每个里程碑完成并 review 后打勾，并 commit 一次。

## Phase 0 — 基础设施（Mac，一次性）
- [x] 创建 `study-rlpd` 分支
- [x] fork rlpd 到个人 GitHub，配置 remote（origin=自己的 fork，upstream=上游）
- [x] 推送 `study-rlpd` 到 fork
- [ ] Linux 上 clone fork 并 checkout 同分支
- [ ] 注册 wandb（浏览器）
- [ ] 双机 `git rev-parse` 同 commit 验证

## Phase 1 — Linux 环境（与带读并行）
- [x] `nvidia-smi` 确认 GPU/驱动（RTX 5060 Laptop / sm_120 / 驱动 570.211 / CUDA 12.8）
- [x] 环境 + 系统包（patchelf、libosmesa6-dev）— 注：实际用 `python3.10 -m venv .venv`，非 conda py3.9
- [x] 依赖实证锁定（jax/flax/gym/dmcgym/d4rl 版本），`pip freeze` 存入 `STUDY/env-lock/linux-2026-09-13.txt`
- [x] import 冒烟通过（全模块可导入 + mujoco_py 能建 D4RL env）
- [x] `wandb login`

## Phase 2 — 带读里程碑（Mac）
- [ ] **M1 论文 + 入口**：读 arXiv 2302.02948，抽 Table 1 基线数字入 PAPER-NOTES；通读 `train_finetuning.py`；理解 offline/online 交错与 mask 语义；~~修复 `log_dir` 上游 bug~~ → **已证伪（2026-09-13）**：`log_dir` 由 `absl.logging` 自动注册（默认 `''`），不会崩，只是 checkpoint 落到相对路径；改进方向是显式定义 `--log_dir` 指向 `results/`
- [ ] **M2 SAC 核心**：`sac_learner.py` 的 `update()`（ensemble critic、min target、polyak tau、temperature 自调节、backup_entropy）
- [ ] **M3 数据与重放**：`rlpd/data/`（环形 buffer、D4RLDataset、双缓冲 iterator、像素 sliding window）
- [ ] **M4 configs 对照**：`sac_config.py` vs `rlpd_config.py` → 大 ensemble + LayerNorm 替代 CQL 保守项的动机
- [ ] **M5 像素线（读）**：DrQ 分支、随机裁剪增强、D4PG encoder、wrappers
- [ ] M6（可选）RedQ dropout / resnet 变体

## Phase 3 — 训练复现（Linux）
- [x] 玩具短跑验证管线：loss 下降、无 NaN、eval return 上升、跑完（冒烟 150–300 步验证 import 链与训练循环）
- [ ] 正式复现：`halfcheetah-expert-v2` + `utd_ratio=20` + 250k 步 + `rlpd_config`，seed=42
- [x] **R1 消融复现（超额完成）**：5 seeds × A/B/C/D，25k 步，结论见 `REPORT-R1.md`
- [ ] 结果按论文口径对照记入 `RESULTS.md`（正式 250k 复现后再填 Table 1 对照）
- [x] 按预算扩展：已完成 5 seeds（超出原计划 3 seeds）

## 验证清单
- [ ] 双机同 commit；`study-rlpd` vs `main` 的 diff 干净可读
- [ ] Linux import 冒烟通过；玩具 run 在 wandb 可见完整曲线
- [ ] 正式 run 的 eval return 达到论文量级；RESULTS.md 有对照表
- [ ] 每个里程碑有提交、代码/笔记先经 review
