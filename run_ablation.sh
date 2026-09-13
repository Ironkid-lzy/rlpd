#!/usr/bin/env bash
# =============================================================================
# r1 多 seed 消融批跑：A/B/C/D × 5 seeds = 20 runs（约 80-100 分钟）
#
# 设计要点:
#   - 外层循环 seed、内层循环实验 → 中断时留下的是"完整的 seed 组"，
#     仍可做配对比较。（反过来会出现 A 全跑完但 B 缺一半，无法对比）
#   - run.sh 内部默认 WANDB_MODE=offline → 训练绝不因断网失败；
#     每个 run 结束后立即 sync（no_proxy='*' 直连，绕开代理 6 秒超时）
#   - set -uo pipefail 但**不加 -e**：单个 run 失败不中断整批
#
# 用法:
#   前台:  bash run_ablation.sh 2>&1 | tee results/ablation_r1.log
#   后台:  nohup setsid bash run_ablation.sh > results/ablation_r1.log 2>&1 &
#
# 可用环境变量覆盖: SEEDS PROJECT STEPS
#   SEEDS="0 1" bash run_ablation.sh     # 只跑 2 个 seed
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results

SEEDS="${SEEDS:-0 1 2 3 4}"
PROJECT="${PROJECT:-rlpd_ablation}"
STEPS="${STEPS:-25000}"
COMMON="--start_training 5000 --max_steps ${STEPS} --config=configs/rlpd_config.py --project_name=${PROJECT}"

run_one() {
  local exp="$1" env="$2" off="$3" utd="$4" seed="$5"
  echo ""
  echo "=== [$(date '+%F %T')] exp=${exp} seed=${seed} env=${env} off=${off} utd=${utd} ==="

  bash run.sh \
    --exp_name="${exp}" \
    --env_name="${env}" \
    --offline_ratio="${off}" \
    --utd_ratio="${utd}" \
    --seed="${seed}" \
    ${COMMON}
  rc=$?

  if [ "${rc}" -ne 0 ]; then
    echo "!!! ${exp}_s${seed} 退出码 ${rc} —— 跳过 sync，继续下一个"
    return "${rc}"
  fi

  echo "--- sync ${exp}_s${seed} ---"
  no_proxy='*' .venv/bin/wandb sync wandb/latest-run 2>&1 | tail -1 || true
  return 0
}

for s in ${SEEDS}; do
  echo ""
  echo "########## SEED ${s} ##########"
  run_one r1_A_online       halfcheetah-expert-v2 0    20 "${s}"
  run_one r1_B_rlpd_expert  halfcheetah-expert-v2 0.5  20 "${s}"
  run_one r1_C_utd1         halfcheetah-expert-v2 0.5  1  "${s}"
  run_one r1_D_rlpd_medium  halfcheetah-medium-v2 0.5  20 "${s}"
done

echo ""
echo "=== ALL DONE $(date '+%F %T') ==="
