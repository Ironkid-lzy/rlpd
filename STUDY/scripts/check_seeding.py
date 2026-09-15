"""诊断脚本: 逐一定位 RLPD 各随机源的播种情况。

发现于 2026-09-15: 同一 seed、同一配置的多次 run 结果差异巨大
（25k 末点评估值 7797.5 / 7019.9 / 1160 / -386 / 1112 ...）。
本脚本用来定位哪些随机源实际受 --seed 控制、哪些被漏掉。

用法（连续跑两次，比较输出是否逐位一致）:
    cd /home/lzy/Projects/rlpd
    env -u PYTHONPATH LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia \
        .venv/bin/python STUDY/scripts/check_seeding.py
"""

from __future__ import annotations

import os
import sys

import gym
import numpy as np

import d4rl  # noqa: F401  ← 必须: D4RL 的 env 靠 import 这个包来注册

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlpd.wrappers import wrap_gym  # noqa: E402

SEED = 0


def rng_state(tag: str, space) -> None:
    """打印一个 space 的 np_random 是否已被播种。"""
    rng = getattr(space, "_np_random", None)
    if rng is None:
        print(f"    [未播种] {tag}: _np_random is None → 首次 sample 会用 os.urandom 取熵")
    else:
        print(f"    [已播种] {tag}: _np_random set, 下一个随机数 = {rng.random():.12f}")


def main() -> None:
    print("=" * 78)
    print("1) 训练环境（与 train_finetuning.py 的构造顺序完全一致）")
    print("=" * 78)
    env = gym.make("halfcheetah-expert-v2")
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(SEED)  # ← train_finetuning.py 第 175 行原本的做法
    # ↓↓↓ 本次新增的修复（train_finetuning.py 里紧跟 env.seed 的两行）↓↓↓
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    # 逐层拆开 Wrapper 链，看每层的 action_space / observation_space
    chain = []
    e = env
    while hasattr(e, "env"):
        chain.append(e)
        e = e.env
    chain.append(e)  # 最内层 raw env

    for w in chain:
        name = type(w).__name__
        print(f"  {name}:")
        rng_state("action_space", w.action_space)
        rng_state("observation_space", w.observation_space)

    print()
    print("  取 3 个 env.action_space.sample()（训练前 5000 步用的就是这个）:")
    samples = [env.action_space.sample() for _ in range(3)]
    for i, s in enumerate(samples):
        print(f"    sample[{i}] = [{s[0]:.12f}, {s[1]:.12f}, {s[2]:.12f}, ...]")

    print()
    print("  取 3 个 env.reset() 的初始观测前 3 维:")
    for i in range(3):
        obs = env.reset()
        print(f"    reset[{i}] = [{obs[0]:.12f}, {obs[1]:.12f}, {obs[2]:.12f}]")

    print()
    print("=" * 78)
    print("2) 离线数据集 / 在线 replay buffer 的采样流")
    print("=" * 78)

    from rlpd.data.d4rl_datasets import D4RLDataset
    from rlpd.data.replay_buffer import ReplayBuffer

    raw = gym.make("halfcheetah-expert-v2")
    raw = wrap_gym(raw, rescale_actions=True)

    ds = D4RLDataset(raw)
    print("  D4RLDataset:")
    print(f"    刚构造完 ds._np_random = {ds._np_random!r}   （None = 未播种）")
    ds.seed(SEED)  # ← 本次新增的修复（train_finetuning.py 里紧跟 D4RLDataset(env) 的那行）
    print(f"    ds.seed({SEED}) 后前 5 个采样下标 = {ds.np_random.randint(10 ** 6, size=5).tolist()}")

    rb = ReplayBuffer(raw.observation_space, raw.action_space, 1000)
    print("  ReplayBuffer:")
    print(f"    刚构造完 rb._np_random = {rb._np_random!r}   （None = 未播种）")
    rb.seed(SEED)  # train_finetuning.py 第 203 行原本就有这一步
    print(f"    rb.seed({SEED}) 后前 5 个采样下标 = {rb.np_random.randint(10 ** 6, size=5).tolist()}")

    print()
    print("判读方法: 连续跑两次本脚本，逐行 diff。")
    print("  - 若样本/观测/下标都一致，而只有标注 [未播种] 的行对应的输出不同 → 该随机源是漏网的。")
    print("  - 若全部一致，则说明非确定性来自 GPU/XLA（需 --xla_gpu_deterministic_ops）。")


if __name__ == "__main__":
    main()
