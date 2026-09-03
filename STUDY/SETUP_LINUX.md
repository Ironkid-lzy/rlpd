# Linux 训练机环境搭建指南

> 目标：在游戏本（Linux）上复现 RLPD。依赖是 2022 年老栈，**版本必须实证锁定**，不要无脑 `pip install -r requirements.txt`。

## 0. 前置检查
```bash
nvidia-smi          # 确认 GPU 型号、驱动、CUDA 版本
cat /etc/os-release # 发行版
conda --version     # 是否已有 conda
```

## 1. 系统包
```bash
sudo apt update
sudo apt install -y patchelf libgl1 libglib2.0-0 libglfw3 libosmesa6
# 若用 mujoco_py 还需要：build-essential python3-dev
```

## 2. conda 环境
```bash
conda create -n rlpd python=3.9 -y
conda activate rlpd
```

## 3. 依赖锁定（实证流程）
1. 先按 README 时代组合装：`jax[cuda11]==0.3.25`、`flax==0.6.x`、`gym[mujoco]==0.23.1`、`tensorflow-probability==0.17.0`、`optax==0.1.3`、`ml_collections`、`absl-py`、`wandb`、`scipy`、`moviepy`、`imageio`
2. `dmcgym` 与 `d4rl` 是 git 依赖且无 commit pin，若与上述版本冲突，pin 到可用 commit
3. 全部装好后：`pip freeze > STUDY/env-lock/linux-<日期>.txt` 并提交
4. 记录实际可用的版本组合到本文件下方

### 实测可用组合（待填）
| 包 | 版本 |
|---|---|
| jax / jaxlib | |
| flax | |
| gym | |
| dm_control | |
| tensorflow-probability | |
| optax | |
| dmcgym commit | |
| d4rl commit | |

## 4. 冒烟测试
```bash
# 1) 全模块可导入
python -c "import rlpd; from rlpd.agents.sac.sac_learner import SACLearner; from rlpd.data.d4rl_datasets import D4RLDataset; print('imports ok')"
# 2) mujoco_py 能建 D4RL env（会触发数据集下载到 ~/.d4rl）
python -c "import gym; e=gym.make('halfcheetah-expert-v2'); print(e.reset().shape)"
```

## 5. wandb
```bash
wandb login   # 浏览器粘贴 token
```

## 6. 训练（玩具短跑 → 正式）
```bash
# 玩具短跑：验证管线
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
  --env_name=halfcheetah-expert-v2 --utd_ratio=1 --start_training=5000 \
  --max_steps=50000 --config=configs/rlpd_config.py --project_name=rlpd_locomotion

# 正式复现（论文命令）
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
  --env_name=halfcheetah-expert-v2 --utd_ratio=20 --start_training=5000 \
  --max_steps=250000 --config=configs/rlpd_config.py --project_name=rlpd_locomotion
```

## 已知坑
- `train_finetuning.py` 引用未注册的 `log_dir` flag → 启动即崩。修复见分支上的补丁（M1 作业），或先 `git stash` 后跑。
- 无头机渲染：像素线需要 `MUJOCO_GL=egl`（或 osmesa）；state 线 D4RL 不依赖渲染。
- 笔记本长训注意散热：插电 + 独显直连模式。
