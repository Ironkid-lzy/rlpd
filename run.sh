#!/usr/bin/env bash
# =============================================================================
# RLPD 一键运行脚本
#
# 封装了本机运行必需的几个环境细节，避免每次敲一大串：
#   1. 清除 ROS2 的 PYTHONPATH 污染（否则会 import 到 /opt/ros 的旧包）
#   2. 提供 mujoco_py 需要的 LD_LIBRARY_PATH（mujoco210 二进制 + nvidia GL）
#   3. wandb 默认用 offline 模式（无需登录；想联网记录则去掉 WANDB_MODE=offline）
#   4. XLA_PYTHON_CLIENT_PREALLOCATE=false：训练前不预占全部 GPU 显存
#   5. 使用项目自己的 .venv/bin/python（避免系统 python 缺包）
#
# 用法：
#   bash run.sh --env_name=halfcheetah-expert-v2 --max_steps=250000 ...
# 等价于 README 里的 python train_finetuning.py，只是环境已替你配好。
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 关键：清掉 ROS2 的 PYTHONPATH，用 venv 的干净 python
exec env -u PYTHONPATH .venv/bin/python train_finetuning.py "$@"
