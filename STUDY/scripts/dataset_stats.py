"""D4RL halfcheetah 数据集描述性统计 —— 用于"认识数据集"。

只读 HDF5 + numpy，不走 d4rl 包的 import（避免其 noisy 副作用和 mujoco_py 初始化）。

用法:
    cd /home/lzy/Projects/rlpd
    env -u PYTHONPATH .venv/bin/python STUDY/scripts/dataset_stats.py
"""

from __future__ import annotations

import os

import h5py
import numpy as np

DATA_DIR = os.path.expanduser("~/.d4rl/datasets")

# 数据质量阶梯：由差到好。第三列是"数据是怎么造出来的"，这是理解 D4RL 的关键。
DATASETS = [
    ("random", "halfcheetah_random-v2.hdf5", "纯随机策略 rollout"),
    ("medium-replay", "halfcheetah_medium_replay-v2.hdf5", "medium 策略训练过程中的整个 replay buffer"),
    ("medium", "halfcheetah_medium-v2.hdf5", "medium 策略（早停的 SAC）的完整 episode"),
    ("expert", "halfcheetah_expert-v2.hdf5", "专家策略（训练充分的 SAC）的完整 episode"),
]

# D4RL halfcheetah 归一化常量（来自 d4rl/infos.py 的 REF_MIN_SCORE / REF_MAX_SCORE）
REF_MIN = -280.178953
REF_MAX = 12135.0


def episode_slices(timeouts: np.ndarray, terminals: np.ndarray) -> list[tuple[int, int]]:
    """把扁平 transition 数组切成 episode。

    D4RL 的 hdf5 是"整段拼接"的扁平数组，没有 episode 索引。
    v2 数据集用 timeouts（1000 步到点）和 terminals（真实终止）标记边界。
    """
    ends = np.where((timeouts > 0) | (terminals > 0))[0]
    if len(ends) == 0:
        return [(0, len(terminals))]
    starts = np.concatenate(([0], ends[:-1] + 1))
    slices = list(zip(starts.tolist(), (ends + 1).tolist()))
    # 最后一小段没被 timeout 收尾就补上，避免丢数据
    if slices[-1][1] < len(terminals):
        slices.append((slices[-1][1], len(terminals)))
    return slices


def norm_score(ret: float) -> float:
    """D4RL 归一化分数：0 = random 水平，100 = expert 水平。"""
    return (ret - REF_MIN) / (REF_MAX - REF_MIN) * 100.0


def collect(name: str, fname: str, desc: str) -> dict | None:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[跳过] 找不到 {path}")
        return None

    with h5py.File(path, "r") as f:
        obs = f["observations"][:]
        act = f["actions"][:]
        rew = f["rewards"][:]
        term = f["terminals"][:]
        tout = f["timeouts"][:]

    slices = episode_slices(tout, term)
    rets = np.array([rew[s:e].sum() for s, e in slices])
    lens = np.array([e - s for s, e in slices])

    act_span = (act.max(axis=0) - act.min(axis=0)) / 2.0  # 数据在 [-1,1] 上用到的比例

    return {
        "name": name,
        "desc": desc,
        "size_mb": os.path.getsize(path) / 1e6,
        "n_trans": len(rew),
        "n_ep": len(slices),
        "obs_dim": obs.shape[1],
        "act_dim": act.shape[1],
        "ret_mean": rets.mean(),
        "ret_std": rets.std(),
        "ret_min": rets.min(),
        "ret_max": rets.max(),
        "len_mean": lens.mean(),
        "len_min": lens.min(),
        "len_max": lens.max(),
        "abs_a_mean": np.abs(act).mean(),
        "a_std_mean": act.std(axis=0).mean(),
        "a_min": act.min(),
        "a_max": act.max(),
        "coverage": act_span.mean(),
        "o_std_mean": obs.std(axis=0).mean(),
    }


def main() -> None:
    rows = [r for r in (collect(*d) for d in DATASETS) if r is not None]
    if not rows:
        print("没有找到任何数据集")
        return

    print("=" * 108)
    print("D4RL halfcheetah-v2 数据集总览")
    print("=" * 108)
    hdr = f"{'数据集':<15}{'transitions':>13}{'episodes':>10}{'序列长':>8}{'obs/act':>10}{'文件':>9}"
    print(hdr)
    print("-" * 108)
    for r in rows:
        print(
            f"{r['name']:<15}{r['n_trans']:>13,}{r['n_ep']:>10,}"
            f"{r['len_mean']:>8.0f}{str(r['obs_dim']) + '/' + str(r['act_dim']):>10}"
            f"{r['size_mb']:>7.0f}MB"
        )

    print()
    print("=" * 108)
    print("数据质量：episode 回报（未折扣累加奖励）")
    print("=" * 108)
    print(f"{'数据集':<15}{'均值':>12}{'标准差':>11}{'最小':>11}{'最大':>11}{'归一化分数':>14}")
    print("-" * 108)
    for r in rows:
        print(
            f"{r['name']:<15}{r['ret_mean']:>12.0f}{r['ret_std']:>11.0f}"
            f"{r['ret_min']:>11.0f}{r['ret_max']:>11.0f}"
            f"{norm_score(r['ret_mean']):>14.1f}"
        )
    print()
    print("归一化分数 = (回报 - REF_MIN) / (REF_MAX - REF_MIN) * 100  "
          f"(REF_MIN={REF_MIN:.1f}, REF_MAX={REF_MAX:.0f})")
    print("解读：0 分 ≈ random 水平，100 分 = expert 水平。这是 D4RL 论文里报数的标准口径。")

    print()
    print("=" * 108)
    print("动作分布（action 是 6 维，取值范围 [-1, 1]）")
    print("=" * 108)
    print(f"{'数据集':<15}{'|a|均值':>11}{'各维std均值':>13}{'最小':>10}{'最大':>10}{'空间覆盖率':>13}")
    print("-" * 108)
    for r in rows:
        print(
            f"{r['name']:<15}{r['abs_a_mean']:>11.3f}{r['a_std_mean']:>13.3f}"
            f"{r['a_min']:>10.3f}{r['a_max']:>10.3f}{r['coverage']:>12.0%}"
        )
    print()
    print("空间覆盖率 = 数据在每个动作维度上实际用到的区间宽度 / 2（动作空间总宽度）。")
    print("动作幅度 |a| 越大代表策略越'激进'；std 越大代表策略越多样/越接近随机。")

    print()
    print("=" * 108)
    print("数据集是怎么造出来的（理解质量差异的关键）")
    print("=" * 108)
    for r in rows:
        print(f"  {r['name']:<15} {r['desc']}")


if __name__ == "__main__":
    main()
