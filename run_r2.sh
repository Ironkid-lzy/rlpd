#!/usr/bin/env bash
# =============================================================================
# r2 数据质量阶梯实验：100k 步 × 5 组 × 5 seeds = 25 runs
#
# ---------------------------------------------------------------------------
# 为什么做这个实验（承接 r1 的两个未解问题）
# ---------------------------------------------------------------------------
# r1 发现："medium 数据前期(10k-20k)明显优于 expert（约 2.3×），25k 时基本持平"。
# 但 r1 有两个硬伤，导致无法下结论：
#   (a) 只有 2 个数据点(medium/expert) —— 无法判断是否存在"质量阶梯"趋势，
#       还是恰好这两个数据集特殊；
#   (b) 只跑了 25k 步（论文官方设置是 250k）—— "末点持平"可能只是"expert 还没到
#       收益拐点"的假象。
#
# r2 的对策：
#   (a) 把数据点从 2 个扩到 4 个，形成质量阶梯：
#         random  <  medium-replay  <  medium  <  expert
#       其中 medium-replay(medium 策略训练时的整个 replay buffer) 是关键对照：
#       它"水平中等但多样性高"，而 medium(早停 SAC 的完整 episode)"水平中等但多样性低"。
#       这一对可以把"数据水平"和"数据多样性"两个混淆因素拆开。
#   (b) 步数 25k → 100k，看 medium 的领先是否会被 expert 反超。
#
# ---------------------------------------------------------------------------
# 控制变量（与 r1 完全一致的纪律：一次只动一个变量）
# ---------------------------------------------------------------------------
#   变的量：
#     - env_name（离线数据集）    → 数据质量阶梯
#     - offline_ratio             → 只有 A 组是 0（纯在线对照），其余均 0.5
#   不动的量：
#     - utd_ratio = 20（全部 5 组）
#     - config = configs/rlpd_config.py（num_qs=10, num_min_qs=2, LayerNorm）
#     - batch_size=256, start_training=5000, seed 集合 [0,1,2,3,4]
#
# 关于 r1 的 C 组（utd_ratio=1）：已在 r1 证明是四组里最差的（末点 -63%，AUC -85%），
# 没有延伸价值，r2 不再重复。
#
# ---------------------------------------------------------------------------
# 相对 r1 的唯一改动：评估密度（不影响训练轨迹）
# ---------------------------------------------------------------------------
#   eval_interval 5000 → 2500  (100k 得到 41 个评估点，看清交叉位置)
#   eval_episodes   10 → 20    (5 seeds × 20 episodes = 100 个 episode 求均值，
#                               把每个评估点的标准误从 ~76 压到 ~54)
#
#   评估开销实测（用两个"只差评估次数"的 run 做差得出；⚠ 别再拿纯仿真 env.step
#   的 0.036 ms/步 去估评估成本 —— 会低估约 40 倍，因为真实评估每步要调一次
#   单样本 JAX 前向，瓶颈在 Python/JAX dispatch）:
#     每个 episode ≈ 0.3 s（确定性策略、无梯度更新）
#     100k 步下 interval=2500/episodes=20 → 41 × 20 = 820 episodes ≈ 4 min/run
#     相对 ~28 min 的训练时长只占约 13% → 加密评估完全划算
#
#   ★ 注意：原先这里写着"r2_expert 前 25k 应与 r1_B 逐点吻合，是本批自带的校验和"。
#     **该说法已被实测证伪并作废** —— 见下方"复现性修复"一节。
#     r1 是在修复前跑的，两轮之间没有可比性，不要拿 r1 的数字当 r2 的基准。
#
# ---------------------------------------------------------------------------
# 复现性修复（2026-09-15，本批开跑前的阻塞性前置工作）
# ---------------------------------------------------------------------------
# 发现过程: 打算用"配置与 r1_B 完全相同"的一次 run 当复现性校验，预测末点应为 7797.5，
#   实测 7019.9（差 10%，FAIL）。随后又跑 3 次完全相同的短 run:
#     1160.1 / -386.3 / 1112.1  —— 同一 seed 竟能差 1500。
#
# 根因: `--seed` 只控制了 env / agent / ReplayBuffer / eval_env 四个 RNG，
#   漏掉了两处（详见 train_finetuning.py 中带 "[关键修复 2026-09-15]" 的注释）:
#     1) 离线数据集 D4RLDataset 从未 seed（Dataset 的 seed 默认 None，
#        np_random 首次访问时用 os.urandom 取熵）→ 每步 2560 条离线样本的抽样流不受控
#     2) wrap_gym 链尾 RescaleAction 新建的 action_space 从未 seed
#        （gym.Wrapper.seed() 只往下传给 inner env，不碰 wrapper 自己的 space）
#        → 训练前 start_training=5000 步的随机探索动作每次都不同
#   诊断脚本: STUDY/scripts/check_seeding.py（连续跑两次 diff 即可定位）
#   排错弯路: 先试了 XLA_FLAGS=--xla_gpu_deterministic_ops=true，极差仅从 1546 降到 127，
#     仍 FAIL。**遇到"同 seed 不可复现"应先查 RNG 播种，别一上来怀疑 GPU。**
#
# 修复后验证: 3 次相同的 7500 步 run → 末点评估值逐位一致（极差 0.0），PASS。
#
# 对解读的影响:
#   - r1 的 A/B/C/D 大差异（数十个百分点）仍然成立；
#   - 但 r1 报告的"seed 极差"里混着采样噪声，**<10% 的组间差异无法与噪声区分**。
#     r1 那条"medium 与 expert 末点仅差 3.5%"的结论不可靠，本批正是要重做它。
#
# ---------------------------------------------------------------------------
# 耗时预估（基于 results/ablation_r1.log 的实测时间戳）
# ---------------------------------------------------------------------------
#   r1 实测 25k ≈ 4.8 min/run（旧评估配置，utd=20）
#   拆解后的单 run 成本模型（实测拟合 + 两个独立数据点交叉验证）:
#     ~15 s      固定启动开销（import / JAX 初始化 / 数据集加载）
#     ~1.65 ms   每个 rollout 步（env.step）
#     ~7.2 ms    每个训练步（utd=20 → 20 次梯度更新）
#     ~1.43 s    每个评估 episode
#   ⇒ 100k 步单 run: 15 + 165 + 679 + 586 ≈ 24 min
#   ⇒ 25 runs ≈ 10 h
#
# ---------------------------------------------------------------------------
# 设计要点（沿用 r1）
# ---------------------------------------------------------------------------
#   - 外层循环 seed、内层循环组 → 中断时留下的是"完整的 seed 组"，仍可做配对比较
#     （反过来会出现 A 全跑完但 B 缺一半，无法配对）
#   - run.sh 默认 WANDB_MODE=offline → 训练绝不因断网失败；每个 run 结束后立即 sync
#     （no_proxy='*' 直连，绕开代理 6 秒超时导致 wandb 5 秒超时失败的问题）
#   - set -uo pipefail 但**不加 -e**：单个 run 失败不中断整批
#   - nohup + setsid 启动：脱离控制终端，避免 VS Code 终端被回收时 SIGHUP 杀掉
#
# 用法（约 9 小时，务必用 nohup setsid）:
#   nohup setsid bash run_r2.sh > results/r2_batch.log 2>&1 &
#   监控:  tail -f results/r2_batch.log | grep -E "^===|^####|^---|^!!!"
#   完成:  no_proxy='*' .venv/bin/wandb sync --sync-all
#
# 可用环境变量覆盖: SEEDS PROJECT STEPS
#   SEEDS="0" STEPS=5000 PROJECT=rlpd_r2_smoke bash run_r2.sh   # smoke test
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results

SEEDS="${SEEDS:-0 1 2 3 4}"
PROJECT="${PROJECT:-rlpd_r2}"
STEPS="${STEPS:-100000}"
# 可只跑部分组（用于重跑失败组 / 计时校验）:
#   EXP_GROUPS="expert" SEEDS=0 STEPS=25000 PROJECT=rlpd_r2_smoke bash run_r2.sh
# ⚠ 变量名不能叫 GROUPS —— 那是 bash 的内置只读数组（用户组 ID），
#   赋值会被 bash 覆盖成 uid（如 1000），取值永远不是你要的组名。
EXP_GROUPS="${EXP_GROUPS:-A random medrep medium expert}"
# 评估配置（可用环境变量覆盖，用于复现性对照实验）
EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"

# 评估配置显式写在命令行里（不依赖代码默认值），让脚本自身就是完整可复现的实验记录
COMMON="--start_training 5000 --max_steps ${STEPS} --eval_interval ${EVAL_INTERVAL} --eval_episodes ${EVAL_EPISODES} --utd_ratio 20 --config=configs/rlpd_config.py --project_name=${PROJECT}"

run_one() {
  local exp="$1" env="$2" off="$3" seed="$4"
  local t0=${SECONDS}

  echo ""
  echo "=== [$(date '+%F %T')] exp=${exp} seed=${seed} env=${env} offline_ratio=${off} ==="
  echo "    utd=20 start_training=5000 max_steps=${STEPS} eval_interval=${EVAL_INTERVAL} eval_episodes=${EVAL_EPISODES}"

  bash run.sh \
    --exp_name="${exp}" \
    --env_name="${env}" \
    --offline_ratio="${off}" \
    --seed="${seed}" \
    ${COMMON}
  rc=$?
  local dt=$((SECONDS - t0))

  if [ "${rc}" -ne 0 ]; then
    echo "!!! ${exp}_s${seed} 退出码 ${rc} (耗时 ${dt}s) —— 跳过 sync, 继续下一个"
    return "${rc}"
  fi

  echo "--- ${exp}_s${seed} 训练完成, 耗时 ${dt}s, 开始 sync ---"
  no_proxy='*' .venv/bin/wandb sync wandb/latest-run 2>&1 | tail -1 || true
  return 0
}

# 组名 → (exp_name, env_name, offline_ratio) 的映射
run_group() {
  case "$1" in
    A)      run_one r2_A_online halfcheetah-expert-v2        0    "$2" ;;
    random) run_one r2_random   halfcheetah-random-v2        0.5  "$2" ;;
    medrep) run_one r2_medrep   halfcheetah-medium-replay-v2 0.5  "$2" ;;
    medium) run_one r2_medium   halfcheetah-medium-v2        0.5  "$2" ;;
    expert) run_one r2_expert   halfcheetah-expert-v2        0.5  "$2" ;;
    *)      echo "!!! 未知组名: $1（可选: A random medrep medium expert）"; return 1 ;;
  esac
}

echo "########## R2 数据质量阶梯: 5 组 × ${STEPS} steps ##########"
echo "开始时间: $(date '+%F %T')"
echo "seeds: ${SEEDS}"
echo "groups: ${EXP_GROUPS}"
echo "wandb project: ${PROJECT}"
echo "预估总耗时: 约 13 小时（25 runs × ~32 min）"

for s in ${SEEDS}; do
  echo ""
  echo "########## SEED ${s} ##########"
  for g in ${EXP_GROUPS}; do
    run_group "${g}" "${s}"
  done
done

echo ""
echo "########## 全部完成 $(date '+%F %T') ##########"
echo "全部同步到云端:  no_proxy='*' .venv/bin/wandb sync --sync-all"
