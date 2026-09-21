# RLPD 复现验收要求（交付给 RLPD repo agent 的任务书 + 判定依据）

> **使用方式**：把本文 §0–§7 **整段复制**给在 `rlpd` 官方 repo 中工作的 agent。
> 该 agent 无法访问研究仓库，因此本文必须自包含。
>
> 创建日期：2026-09-21 ｜ 对应阶段：Phase B1（见 `docs/RLPD_REPRODUCTION.md`）

---

## §0 你的角色与边界

**工作对象**：官方 repo `ikostrikov/rlpd`（commit `c90fd4b`，JAX 实现，代码即论文 camera-ready 版）。

**它的定位**：**ground-truth 参考实现 + 环境/数据管线的验证器**，不是研究载体。研究主体在另一个仓库（PyTorch 栈），本 repo 只贡献两样东西：(1) 可信的对照曲线；(2) 移植时的代码对照物。

**你可以做**：
- 运行训练、解析日志 / wandb、统计与绘图
- 撰写报告文件
- 做不改动算法语义的**纯日志 instrumentation**（如额外打印指标）

**你不可以做**（越界会摧毁本 repo 作为 ground-truth 的价值）：
1. **修改算法代码**（`rlpd/agents/**`、`rlpd/networks/**`、`configs/**`、`train_finetuning.py` 的训练逻辑）。唯一例外是上述 instrumentation，且必须在报告中**单列声明改了哪个文件哪几行、为什么**
2. **通过调参拟合论文数字**。若使用论文配置却对不上，**如实报告差异并排查原因**（数据版本 / 评测协议 / seed 数 / 依赖版本），**不得**反复试超参直到"看起来对"
3. **把探索性结果（sandbox 模式的实验）表述为已验收结论**——两者必须在报告中分区呈现

---

## §1 复现的目的（为什么做，按优先级）

| 级别 | 目的 | 说明 |
|---|---|---|
| **P1** | 建立 ground-truth 锚点 | 本机环境 + D4RL 数据管线 + 论文配置能跑出**与论文一致的曲线**。这是未来 PyTorch 移植（Phase B2）的对照基准：移植正确性 = 两条曲线一致 |
| **P2** | 亲手理解 5 个组件 | 对称采样（50/50）/ 高 UTD (20) / LayerNorm / ensemble (E=10, Z=2) / 无显式 conservatism——在代码中的**确切实现位置与语义** |
| **P3** | 生成 owned questions | 记录"论文未说明但必须做决策"的点，以及观察到的异常——作为后续受控实验（Phase C）的素材 |
| **P4** | 环境可复现性 | 任何人按文档在本机可复现同样的 run |

**明确不作为目的的**（避免过度工作）：
- ❌ 做出新贡献、超越论文数字
- ❌ 复现论文全部 30 个任务（本阶段只做 §2 E1 指定的子集）
- ❌ 解释我们观察到的异常现象（如 medium vs expert crossover）——它属于 sandbox 记录，不属于验收范围
- ❌ PyTorch 移植（那是另一个仓库的 B2 工作）

---

## §2 验收标准（Exit Criteria）—— 逐条可核查

### E1 曲线对齐（核心指标）
- **范围**：3 envs × 2 datasets = **6 个组合**，全部 state-based D4RL：
  - `halfcheetah-medium-replay-v2`、`halfcheetah-medium-v2`
  - `hopper-medium-replay-v2`、`hopper-medium-v2`
  - `walker2d-medium-replay-v2`、`walker2d-medium-v2`
  - （依据论文 Appendix A.3 Fig 18 的 locomotion 组；**不含 ant**，ant 属 AntMaze/其他组）
- **协议**：每个组合 **≥5 seeds**（seed 0–4）；`utd_ratio=20`；`start_training=5000`；env steps 预算与论文一致（medium/medium-replay ≈ 200k–250k，以 README 示例为准并在报告中写明）；locomotion 组用 `configs/rlpd_config.py` 默认（不做 AntMaze 式覆盖）
- **达标判据**：与论文 Fig 18 对比 —— **曲线形态一致 + 曲线终点在 ±10% 以内**。论文用 10 seeds、我们用 5 seeds，此差异必须在报告中注明（不构成否决项）
- **证据**：`repro_e1_score_table.md`（或等效）含每个组合的终点 mean ± std，并附**叠加对比图**（我们的曲线 vs 论文曲线截图或人工读出的参考值）

### E2 每个 run 的元数据完整
每条记录必须包含：env_name、dataset 版本、seed、完整启动命令、env steps 预算、`utd_ratio`、所有 config 覆盖项、repo commit SHA、jax/jaxlib 版本、d4rl/mujoco 版本、eval 协议（`eval_interval`、`eval_episodes`）、GPU 小时数、起止时间。

### E3 组件映射表
5 个组件 × 三类信息：**(a) 代码位置**（文件:行号）、**(b) 对应 config 项**、**(c) 依据论文消融（Fig 7–12）移除后的预期变化**。

| 组件 | 代码位置 | config 项 | 移除后论文报告的变化 |
|---|---|---|---|
| 对称采样 50/50 | ? | `offline_ratio` | ? |
| 高 UTD=20 | ? | `utd_ratio` | ? |
| LayerNorm（critic） | ? | `critic_layer_norm` | ? |
| ensemble E=10 / min 子集 Z | ? | `num_qs` / `num_min_qs` | ? |
| 无显式 conservatism | ?（结构性缺失，需说明"没有哪个文件"） | — | ? |

> 已知线索（供核对，不必照抄）：LN 在 `rlpd/networks/mlp.py` 顺序为 `Dense→LayerNorm→ReLU` 且每层都有、critic 独有（`state_action_value.py` 末层 `Dense(1)`）；ensemble 在 `rlpd/networks/ensemble.py`（`subsample_ensemble`）；默认配置 `configs/rlpd_config.py`。
> **请逐条独立核实并给出准确行号**——若与上述线索不符，以你的实测为准并指出差异。

### E4 复现问题清单（≥10 条）
论文未指定、但实现必须做决策的点。每条写：**决策内容 + 我们选了什么 + 依据 + 风险**。
示例类型：LN 的 ε / elementwise_affine；actor `log_std` 的 clip 范围；target entropy 取值；`start_training` 的步数；offline buffer 的采样是否放回；`dones`/`masks` 的构造（尤其 D4RL 的 terminal 处理）；eval 用 `mode()` 还是采样；observation 归一化与否。

### E5 异常与观察记录（与 E1–E4 分区）
包括已知的探索性观察：**medium vs expert 数据集在训练中期与后期的优劣交叉（crossover）**，用户侧已有 5 seeds 且训练稳定。
- 每条标注 `status: exploratory`，**不得**与 E1 的验收结论混写
- 格式：观察（原始数字）与解释（推测机制）**分开写**
- **注意**：本项不阻塞验收；它是给研究仓库的素材，不是本 repo 的交付质量指标

### E6 环境配方与坑位
从零到跑通的完整命令序列 + 实际遇到的坑与解法（mujoco_py 编译、MuJoCo 二进制路径、jax CUDA 版本、显存预分配、wandb 离线模式等）。若某步与现有文档 `docs/RLPD_REPRODUCTION.md`（研究仓库）的描述不同，**以你的实测为准并明确指出差异**。

---

## §3 当前状态（2026-09-21，由用户提供，供你校对）

- ✅ 环境已在 RTX 5060（**Linux**）部署，训练可跑通并产出曲线
- ✅ 组件级代码定位已有初步结果（见 E3 线索）
- ✅ 已有探索性实验：medium vs expert 数据集对比（5 seeds，训练稳定）观察到 crossover
- ⬜ **尚未完成**：E1 的正式 sweep（6 组合 × 5 seeds × 与论文对比）与 E2–E6 的成文记录

**请以你的实测核对这些陈述；若有出入，在报告中明确纠正。**

---

## §4 交付物

1. **`REPRO_REPORT.md`**（写在 rlpd repo 根目录）：按 E1–E6 组织，逐条给出「达标 / 未达标 / 无法判定」+ 证据
2. **判定结论块**（模板见 §6）
3. **run 清单表**：env, dataset, seed, 命令, 起止时间, GPU 小时, 关键指标（一份可直接复制的表格）
4. **纯数字摘要**：供复制回研究仓库的 `docs/EXPERIMENT_TRACKER.md`（只含数字与元数据，不含解释）

---

## §5 纪律

- **≥5 seeds 才算证据**；不足的 run 必须标注 `preliminary`
- **计量单位必须是 env steps**。注意 `utd_ratio=20` 时 gradient steps = 20 × env steps——禁止把两者混用或互换陈述
- **不删除失败的 run**；标注 `failed` / `abandoned` 并记录原因
- 每次 run 记录 repo commit SHA；若中途改过代码（仅限 instrumentation），必须分段标注哪些 run 受影响
- **禁止静默跳过**：任何未完成项都要说明阻塞原因（缺数据 / 算力 / 依赖 / 时间）

---

## §6 判定结论模板

```
## RLPD 复现判定（日期）

复现可否结束：是 / 否

达标项：E1 ✅（6/6 组合，终点偏差 x%–y%）｜E2 ✅｜E3 ✅｜E4 ✅（n=__ 条）｜E5 ✅｜E6 ✅
未达标项：<逐条列出 + 证据缺失点>
无法判定项：<说明为何无法判定>

差距清单（按优先级）：
1. ...
2. ...

剩余算力估算：<GPU 小时 / 天数>
建议下一步：<收敛路径>
```

---

## §7 判定规则

- **全部 E1–E6 产出且 E1 达标 → 复现可结束**，进入 Phase B2（PyTorch 移植）
- **E1 未达标 → 给出差距清单 + 剩余算力估算 + 收敛建议**；**不要**自行调参挽救，先报告差异
- **E1 达标但 E2–E6 未齐 → 不可结束**（缺文档 = 复现不可交接，违反 P4）
- 若发现**论文与官方 repo 不一致**（如默认配置、评测协议），这是一项**有价值的发现**，请在报告中单列——它直接影响后续移植的目标值
