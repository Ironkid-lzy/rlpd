#! /usr/bin/env python
# =============================================================================
# RLPD (Reinforcement Learning with Prior Data) — state-based fine-tuning 入口。
# 论文: "Efficient Online Reinforcement Learning with Offline Data" (arXiv:2302.02948)
#
# Mental Model: 这个脚本是"总指挥"——它拥有环境交互循环、offline/online 数据混合、
# 以及日志记录；真正的学习数学（critic/actor 更新）在 SACLearner 里。
# 核心洞见: 不发明新算法，只是把离线数据混进在线 off-policy 训练，配合几个最小改动。
# =============================================================================
import os
import pickle

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

# --- 命令行参数体系（两层） ---
# 1) absl flags: 顶层运行旋钮（env、seed、batch、步数、utd_ratio...）
# 2) ml_collections config_flags: 超参文件（--config=...），lock_config=False 允许
#    --config.xxx=yyy 命令行覆盖文件里的值。
# 注意默认值: offline_ratio=0.5（每步 batch 一半来自离线数据）、
# start_training=1e4（前 1 万步纯随机探索不更新）、utd_ratio=1（论文复现用 20）。
flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# combine(): RLPD 数据混合的"心脏"。
# 把 offline batch 与 online batch 逐元素交错: tmp[0::2]=offline, tmp[1::2]=online，
# 得到 [off, on, off, on, ...] 的单个 batch。
# Mental Model: 因为 update() 会把 batch 均分成 utd_ratio 份依次更新，交错能保证
# 每一份小 batch 都同时含 offline 和 online 数据——每次梯度步都看到两种分布，
# 而不是前几次全 offline、后几次全 online。这是 RLPD 稳定工作的关键细节。
# 递归分支处理 Dict 观测（像素版 obs 是 dict），普通数组直接交错。
def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def main(_):
    # offline_ratio 必须是 [0,1] 内的比例（offline 数据在每步 batch 中的占比）。
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    # wandb 是唯一的日志后端（硬依赖）。所有指标（loss、return）只走它。
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    # exp_prefix: 本次运行的标识（seed + pretrain 步数 + 是否 LayerNorm），
    # 用于 checkpoint 目录命名，方便区分多次运行。
    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    # ⚠️ 上游 bug: FLAGS.log_dir 从未被注册为 flag（absl 对未定义 flag 直接抛
    # AttributeError）→ 任何运行都会在这里启动即崩。修复见 M1 作业。
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    # --- 训练环境 ---
    # wrap_gym 包装链: SinglePrecision(float64→float32, JAX 必须) →
    # UniversalSeed(连 obs/action space 一起 seed, 保证可复现) →
    # RescaleAction(-1,1)(与 D4RL 数据动作范围对齐) → ClipAction(防越界)。
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    # 离线数据集: D4RLDataset 包装 d4rl.qlearning_dataset(env)，首次调用自动下载到 ~/.d4rl。
    # "binary" 任务（Adroit）走 BinaryDataset（AWAC 格式，需手动下载）。
    # not ideal, but works for now:
    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    # 评估环境单独建一个（seed+42），与训练环境解耦，避免评估污染训练状态。
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    # --- Agent 创建 ---
    # model_cls 从 config 里取（rlpd_config → "SACLearner"），用 globals() 动态实例化，
    # 因此同一入口可跑不同算法（SAC / DrQ 等），只需换 config 文件。
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    # 在线经验池: 预分配 numpy 环形数组，容量 = max_steps（即整个训练过程都存得下）。
    # Mental Model: 重放缓冲把 RL 从"在线序列问题"变成"近似 i.i.d. 的监督学习问题"。
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    # --- 可选: 离线预训练阶段（pretrain_steps > 0 时启用） ---
    # 只吃 offline batch 做纯离线更新（不与环境交互），相当于"先学离线数据再上线"。
    # 论文默认 0（不预训练），RLPD 的卖点就是"不需要预训练，直接在线混合"。
    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            # antmaze 稀疏奖励整形: 所有 reward 减 1（成功 +1 → 0，失败 0 → -1），
            # 让稀疏任务变成"惩罚步数"的稠密化信号，是 D4RL antmaze 的常见 trick。
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    # ==================== 在线训练主循环（每 env step 一次） ====================
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        # --- 动作选择: 探索期 vs 学习期 ---
        # start_training 之前用随机动作（纯探索，填充经验池）；
        # 之后用 agent.sample_actions()（随机策略采样，SAC 的探索来自策略随机性）。
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        # --- mask 语义（关键细节） ---
        # mask 是折扣项: target = r + γ·mask·Q_next。区分两种"结束":
        # - 真终止(terminal, 如摔倒): mask=0，Q 目标不含未来
        # - 时间截断(TimeLimit.truncated, 如 1000 步到点): mask=1，环境其实还能继续
        # 把截断误当终止会导致价值估计偏差（低估），这是 RL 里常见的隐性 bug 来源。
        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        # 把这条 transition (s, a, r, mask, done, s') 写入环形缓冲。
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        # episode 结束时记录统计: info["episode"] 里 r/l/t 是 gym 的缩写键，
        # decode 成 return/length/time 后以 training/ 前缀记入 wandb。
        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        # --- 学习更新（RLPD 的核心数据混合） ---
        # 每步各抽一个半 batch:
        # - online: 从刚积累的在线经验池采样，数量 ∝ (1 - offline_ratio)
        # - offline: 从整个 D4RL 数据集 i.i.d. 采样，数量 ∝ offline_ratio
        # 注意 offline 是"每次从全量数据集重抽"，不是固定子集——保证多样性。
        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            # combine() 交错成单个 batch，交给 update() 内部按 utd_ratio 切份依次更新。
            batch = combine(offline_batch, online_batch)

            # antmaze 在线数据同样做 reward 整形（与离线侧一致）。
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            # update() 返回 (新 agent, 指标 dict)；agent 是不可变对象（flax PyTree），
            # 每次更新返回新实例——这是 JAX 函数式风格的体现。
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        # --- 周期性评估 + 可选 checkpoint ---
        # 每 eval_interval 步用确定性策略（eval_actions，取分布 mode）跑 10 个 episode，
        # 记录 evaluation/return。评估用确定性、训练用随机采样是 RL 标准做法。
        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            # checkpoint 默认关闭；开启时保存 flax TrainState（agent）或整个经验池。
            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
