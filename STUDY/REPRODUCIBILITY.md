# 复现性缺陷报告：同一 seed 不可复现

> 发现日期：2026-09-15
> 严重程度：**高** —— 影响 R1 全部误差棒的解读
> 状态：已定位根因、已修复、已实测验证

## 一、症状

在把 R1（25k 步）延长到 100k 之前，先做了一次"同配置复现性冒烟校验"：
用**与 `r1_B_rlpd_expert` seed 0 逐字相同**的命令再跑一次，预期末点评估值应等于
R1 的 7797.5。

| 运行 | 日期 | 配置 | 25k 末点 `evaluation/return` |
|---|---|---|---|
| R1 B seed 0 | 2026-09-13 | expert, off=0.5, utd=20, eval=5000/10 | **7797.5** |
| 复现校验 | 2026-09-15 | 同上，逐字相同 | **7019.9** |

差距 777.6（10%）。随后又跑了 3 次完全相同的短 run（7500 步）：

| 运行 | `evaluation/return` |
|---|---|
| #1 | 1160.11 |
| #2 | **-386.33** |
| #3 | 1112.13 |

极差 **1546** —— 同一 seed、同一配置，结果能差到一正一负。

## 二、根因

`--seed` 只控制了 4 个随机源：训练 env（`env.seed`）、agent（`Agent.create`）、
在线 replay buffer（`replay_buffer.seed`）、评估 env（`seed+42`）。
**漏掉了两个**：

### 根因 1：离线数据集从未播种

`rlpd/data/dataset.py`：

```python
def __init__(self, dataset_dict, seed: Optional[int] = None):
    self._np_random = None
    self._seed = None
    if seed is not None:          # ← D4RLDataset 调用时不传 seed，永远进不来
        self.seed(seed)

@property
def np_random(self):
    if self._np_random is None:
        self.seed()               # ← 无参调用 → 用 os.urandom 取熵
    return self._np_random
```

`D4RLDataset.__init__` 与 `ReplayBuffer.__init__` 都调 `super().__init__(dataset_dict)`，
不传 seed。区别是 `train_finetuning.py` 上游**只给 replay_buffer 补了种**（第 203 行），
**漏了离线数据集 `ds`**。

后果：每步要抽 `batch_size × utd_ratio × offline_ratio = 256 × 20 × 0.5 = 2560`
条离线样本，这个抽样流完全不受控。

### 根因 2：外层 action_space 从未播种

`rlpd/wrappers/__init__.py` 的包装链：

```
gym.make(...) → SinglePrecision → UniversalSeed → RescaleAction → ClipAction
                                                       ↑ 在 __init__ 里新建了一个 spaces.Box
                                                         作为 self.action_space
```

- `gym.Wrapper.seed()` 只把 seed **往下**传给 inner env，从不碰 wrapper 自己的 action_space
- `UniversalSeed.seed()` 也只给 `self.env.action_space`（内层）播种
- ⇒ 最外层那个新建的 `spaces.Box` 永远没被播种

后果：`train_finetuning.py` 里

```python
if i < FLAGS.start_training:      # start_training = 5000
    action = env.action_space.sample()
```

**训练前 5000 步的随机探索动作每次都不同。** 这 5000 条 transition 进池后，
整条训练轨迹随之发散。

> 反直觉的一点：`env.reset()` 是**确定的**（诊断脚本里两次运行的初始观测逐位一致），
> 只有 `env.action_space.sample()` 不确定。所以肉眼查代码很容易漏掉。

## 三、诊断工具

`STUDY/scripts/check_seeding.py` —— 逐层拆开 Wrapper 链，打印每一层
action_space / observation_space 的 `_np_random` 是否存在。连续跑两次 `diff` 即可定位。

修复前的输出（关键部分）：

```
RecordEpisodeStatistics:  [未播种] action_space      ← env.action_space 就是它
ClipAction:               [未播种] action_space
RescaleAction:            [未播种] action_space
UniversalSeed:            [已播种] action_space      ← 只种到了这里
取 3 个 env.action_space.sample():
    sample[0] = [-0.116318754852, 0.919082343578, ...]     ← 两次运行不同
    sample[0] = [ 0.776469647884, -0.012249834836, ...]
取 3 个 env.reset():
    reset[0] = [-0.046042658389, -0.091805294156, ...]     ← 两次运行相同
```

## 四、修复

`train_finetuning.py` 中三处（均带 `[关键修复 2026-09-15]` 注释）：

```python
env.seed(FLAGS.seed)
env.action_space.seed(FLAGS.seed)              # 根因 2
env.observation_space.seed(FLAGS.seed)

eval_env.seed(FLAGS.seed + 42)
eval_env.action_space.seed(FLAGS.seed + 42)    # 根因 2（评估用确定性动作，影响小，但保持一致）
eval_env.observation_space.seed(FLAGS.seed + 42)

ds.seed(FLAGS.seed)                            # 根因 1
```

### 验证

修复后跑 3 次相同的 7500 步 run：

| 运行 | `evaluation/return` |
|---|---|
| #1 | 1021.9580078125 |
| #2 | 1021.9580078125 |

**逐位一致，极差 0.0 → PASS。**

## 五、走过的弯路

先怀疑 GPU/XLA 非确定性，试了 `XLA_FLAGS=--xla_gpu_deterministic_ops=true`：
极差仅从 1546 降到 127，**仍 FAIL**。

**教训：遇到"同 seed 不可复现"，先查 RNG 播种，不要一上来就怀疑 GPU。**
GPU 非确定性通常造成的是微小数值漂移；而一个完全未播种的 RNG 会造成
"同名运行结果差到一正一负"这种量级的差异。

## 六、对 R1 结论的影响

| R1 结论 | 是否仍成立 |
|---|---|
| 离线数据有用（B vs A，+61%） | ✅ 成立（差异远大于噪声） |
| 高 UTD 有用（B vs C，-63%） | ✅ 成立 |
| UTD 影响 > 离线数据（A vs C） | ✅ 成立 |
| **medium 与 expert 末点仅差 3.5%** | ❌ **不可靠** |
| **expert 组 seed 极差 3746** | ⚠️ 该数字里混着采样噪声，不能当作纯 seed 方差 |

原因：单次"同配置同 seed"的差异就达 10%（777.6/7797.5），
与"medium vs expert 差 3.5%"同量级甚至更大。
**R1 报告中所有小于 10% 的组间差异都无法与噪声区分。**

这正是 R2 要重做的部分。

## 七、预防规则

1. **长批任务开跑前，先花 5 分钟做一次"同配置重复跑是否一致"的冒烟校验。**
   本次若直接开跑 13 小时的批任务，25 个 run 会全部建立在"seed 不可复现"的沙地上。
2. 加新随机源时，把它接进 `FLAGS.seed`，并在 `STUDY/scripts/check_seeding.py` 里登记。
3. 报告误差棒时，区分"seed 间方差"与"运行间方差"——只有前者是科学上有意义的量。
