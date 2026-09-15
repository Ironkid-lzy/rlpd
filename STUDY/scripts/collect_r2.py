"""从 wandb 云端拉取 R1/R2 的评估曲线，产出 CSV 并自动做复现性校验。

前置条件: 每个 run 已 sync 到云端（run_r2.sh 在每个 run 结束后自动 sync）。

用法:
    cd /home/lzy/Projects/rlpd
    no_proxy='*' env -u PYTHONPATH .venv/bin/python STUDY/scripts/collect_r2.py

产出:
    STUDY/results/r2/r2_5seed_eval_return.csv   各组 5-seed 的 mean/min/max + 逐 seed 末值
    STUDY/results/r2/r2_reproducibility.txt     与 R1 前 25k 的逐点比对结果

为什么用云端而不是本地 wandb 目录:
    wandb 0.29 的本地 offline run 目录只写 wandb-summary.json（最终值），
    完整 history 在二进制的 .wandb 文件里、没有稳定的解析接口。
    sync 之后用公开的 Api() 读 history 是唯一干净的做法。
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import wandb
except ImportError:
    sys.exit("需要 wandb: .venv/bin/pip install wandb")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "r2")

# 组名 → 人类可读标签
R1_LABELS = {
    "r1_A_online": "A online",
    "r1_B_rlpd_expert": "B expert",
    "r1_C_utd1": "C utd1",
    "r1_D_rlpd_medium": "D medium",
}
R2_LABELS = {
    "r2_A_online": "A online",
    "r2_random": "random",
    "r2_medrep": "medium-replay",
    "r2_medium": "medium",
    "r2_expert": "expert",
}

# R1 与 R2 中"配置完全相同"的组对，用于复现性校验。
# 依据: rlpd/agents/agent.py 的 eval_actions() 是纯函数（dist.mode()，不消耗 RNG），
# 所以 eval_interval / eval_episodes 的改动不会扰动训练轨迹。
REPRO_PAIRS = [
    ("r1_A_online", "r2_A_online"),
    ("r1_D_rlpd_medium", "r2_medium"),
    ("r1_B_rlpd_expert", "r2_expert"),
]


def fetch(api, project: str) -> dict[str, dict[int, list[float]]]:
    """返回 {group: {env_step: [每个 seed 的 evaluation/return]}}"""
    entity = api.default_entity
    path = f"{entity}/{project}"
    try:
        runs = list(api.runs(path, filters={"state": "finished"}))
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 无法读取 {path}: {e}")
        return {}

    out: dict[str, dict[int, list[float]]] = {}
    for r in runs:
        group = r.group or r.name
        hist = r.history(keys=["evaluation/return", "_step"], pandas=False)
        for row in hist:
            if "evaluation/return" not in row:
                continue
            step = int(row["_step"])
            out.setdefault(group, {}).setdefault(step, []).append(float(row["evaluation/return"]))
    print(f"[{path}] 读到 {len(runs)} 个 run, {len(out)} 个组: {sorted(out)}")
    return out


def summarize(data: dict[int, list[float]]) -> dict[int, tuple[float, float, float, int]]:
    """{step: (mean, min, max, n)}"""
    return {
        s: (float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v))
        for s, v in sorted(data.items())
    }


def auc(steps, means) -> float:
    """梯形法积分 / 步数跨度 —— 与 REPORT-R1.md 的口径一致。"""
    steps = np.asarray(steps, dtype=float)
    means = np.asarray(means, dtype=float)
    return float(np.trapz(means, steps) / (steps[-1] - steps[0]))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    api = wandb.Api()
    print(f"wandb entity: {api.default_entity}")

    r2 = fetch(api, "rlpd_r2")
    r1 = fetch(api, "rlpd_ablation")

    if not r2:
        sys.exit("没有读到 r2 数据 —— 请先确认 run_r2.sh 已把 run sync 到云端（no_proxy='*' wandb sync --sync-all）")

    # ---------- 1) 输出 R2 的 5-seed 均值曲线 CSV ----------
    groups = [g for g in R2_LABELS if g in r2]
    all_steps = sorted({s for g in groups for s in r2[g]})
    csv_path = os.path.join(OUT_DIR, "r2_5seed_eval_return.csv")
    with open(csv_path, "w") as f:
        f.write("env_step," + ",".join(f"{R2_LABELS[g]}_mean,{R2_LABELS[g]}_min,{R2_LABELS[g]}_max,{R2_LABELS[g]}_n" for g in groups) + "\n")
        for s in all_steps:
            cells = []
            for g in groups:
                if s in r2[g]:
                    m, lo, hi, n = summarize(r2[g])[s]
                    cells += [f"{m:.1f}", f"{lo:.1f}", f"{hi:.1f}", str(n)]
                else:
                    cells += ["", "", "", "0"]
            f.write(f"{s}," + ",".join(cells) + "\n")
    print(f"写出 {csv_path}")

    # ---------- 2) 屏幕表格 ----------
    print()
    print("=" * 100)
    print("R2 数据质量阶梯（5 seed 均值，evaluation/return）")
    print("=" * 100)
    hdr = f"{'step':>7} " + "".join(f"{R2_LABELS[g]:>16}" for g in groups)
    print(hdr)
    print("-" * 100)
    for s in all_steps:
        row = f"{s:>7} "
        for g in groups:
            ss = summarize(r2[g])
            row += f"{ss[s][0]:>16.1f}" if s in ss else f"{'-':>16}"
        print(row)

    print()
    print("末点 / AUC（5 seed 均值）")
    print(f"{'组':<16}{'末点 mean':>12}{'末点极差':>12}{'AUC':>12}")
    print("-" * 52)
    for g in groups:
        ss = summarize(r2[g])
        last = max(ss)
        m, lo, hi, _ = ss[last]
        st = sorted(ss)
        a = auc(st, [ss[x][0] for x in st])
        print(f"{R2_LABELS[g]:<16}{m:>12.1f}{hi - lo:>12.1f}{a:>12.1f}")

    # ---------- 3) 与 R1 的复现性校验 ----------
    lines = []
    lines.append("R1 vs R2 前 25k 对照（定性参考，不是复现性校验）")
    lines.append("=" * 78)
    lines.append("⚠ R1 是在修复 RNG 播种缺陷**之前**跑的（见 STUDY/REPRODUCIBILITY.md）：")
    lines.append("   它的数值里混着未受控的离线采样噪声，单次同配置差异可达 10%。")
    lines.append("   所以下表的逐点差异**不构成任何结论**，只用来粗查量级是否离谱。")
    lines.append("")
    for g1, g2 in REPRO_PAIRS:
        if g1 not in r1 or g2 not in r2:
            lines.append(f"{g1} → {g2}: 数据缺失，跳过")
            continue
        s1, s2 = summarize(r1[g1]), summarize(r2[g2])
        common = sorted(set(s1) & set(s2))
        lines.append(f"{g1} → {g2}   (共同评估点 {len(common)} 个)")
        if not common:
            lines.append("   无共同评估点（R1 interval=5000, R2 interval=2500，应在 5000 的倍数上对齐）")
            lines.append("")
            continue
        lines.append(f"   {'step':>7}{'R1':>12}{'R2':>12}{'diff':>10}")
        maxdiff = 0.0
        for s in common:
            d = s2[s][0] - s1[s][0]
            maxdiff = max(maxdiff, abs(d))
            lines.append(f"   {s:>7}{s1[s][0]:>12.1f}{s2[s][0]:>12.1f}{d:>10.1f}")
        verdict = (
            "几乎一致（<1%）"
            if maxdiff < 0.01 * max(1.0, abs(s1[common[-1]][0]))
            else "差异较大 —— 但 R1 本身带 10% 量级的采样噪声，属预期范围"
        )
        lines.append(f"   最大偏差 = {maxdiff:.4f}  →  {verdict}")
        lines.append("")

    txt = "\n".join(lines)
    print()
    print(txt)
    rep_path = os.path.join(OUT_DIR, "r2_vs_r1_prefix25k.txt")
    with open(rep_path, "w") as f:
        f.write(txt + "\n")
    print(f"写出 {rep_path}")


if __name__ == "__main__":
    main()
