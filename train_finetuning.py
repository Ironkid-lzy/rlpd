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
# exp_name: 实验标签，例如 r1_A_online / r1_B_rlpd_expert。
# 作用（多 seed 对照实验防混淆的关键）:
#   - wandb run name = "{exp_name}_s{seed}"，一眼看出属于哪个实验、哪个 seed
#   - wandb group    = exp_name，网页上按 group 聚合：同一实验的 5 个 seed
#                      自动合成 1 条 mean 曲线 + 阴影，而不是 5 条散线
#   - 写入 wandb config，可按条件筛选
#   - 带 r1_ 前缀用于区分"第几轮"实验，下轮改 r2_ 即可
flags.DEFINE_string("exp_name", "exp", "Experiment label (wandb run name/group).")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
# --- R2 起相对上游的改动: 加密评估 ---
# eval_interval: 5000 → 2500（eval_episodes 保持 10）
#
# 实测评估开销模型（RTX 5060 / halfcheetah / utd=20）:
#   每个 episode ≈ 1.43 s（1000 步 × ~1.4 ms/步）
#   慢的根源: 每步要调一次 agent.eval_actions()，即一次单样本 JAX 前向；
#   这 1.4 ms 里绝大部分是 Python/JAX dispatch 开销而非 GPU 计算。
#   ⚠ 不要用"纯仿真 env.step"的 0.036 ms/步 去估评估开销 —— 会低估约 40 倍。
#
# 于是总评估耗时 ∝ 评估次数 × episodes。100k 步下三档实测外推:
#   interval=5000, episodes=10 → 21 个点,  5.0 min/run,  整批 ~8.1 h  (r1 的配置)
#   interval=2500, episodes=10 → 41 个点,  9.8 min/run,  整批 ~10.0 h ← r2 采用
#   interval=2500, episodes=20 → 41 个点, 19.6 min/run,  整批 ~14.1 h
#
# 为什么 episodes 保持 10 而不是翻倍:
#   5 seeds × 10 episodes = 50 个 episode，单个评估点的均值标准误 ≈ 76；
#   翻倍到 20 只能压到 ≈ 54 —— 而我们要分辨的组间差异在千量级，精度早已过剩。
#   同样多出来的时间花在"加密评估点"上更有价值: r1 留下的核心问题是
#   "medium 的领先在第几步被 expert 反超"，这需要时间分辨率，不是精度。
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 2500, "Eval interval.")
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
    # name/group 是多 seed 对照实验的命名机制：
    #   name  = "r1_B_rlpd_expert_s0"  唯一可辨（哪个实验 + 哪个 seed）
    #   group = "r1_B_rlpd_expert"     网页按组聚合 → 1 条 mean±std 曲线
    wandb.init(
        project=FLAGS.project_name,
        name=f"{FLAGS.exp_name}_s{FLAGS.seed}",
        group=FLAGS.exp_name,
    )
    wandb.config.update(FLAGS)

    # exp_prefix: 本次运行的标识（seed + pretrain 步数 + 是否 LayerNorm），
    # 用于 checkpoint 目录命名，方便区分多次运行。
    exp_prefix = f"{FLAGS.exp_name}_s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    # 澄清（2026-09-13 实证）: 这里并非"上游 bug"。log_dir 由 absl.logging 自动注册:
    #   absl/logging/__init__.py: flags.DEFINE_string('log_dir',
    #       os.getenv('TEST_TMPDIR', ''), 'directory to write logfiles into', ...)
    # 默认值为 ''，故 os.path.join('', exp_prefix) == exp_prefix：
    # → 不会崩，但 checkpoint/buffer 会落在【当前工作目录】下的相对路径。
    # 若要集中管理，应显式定义自己的 --log_dir 并指向 results/ 等目录。
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
    # --- [关键修复 2026-09-15 之二] 补种外层 action/observation space ---
    # 上游缺陷: wrap_gym 链尾是 RescaleAction，它在 __init__ 里**新建**了一个
    #   spaces.Box 作为 self.action_space；而 gym.Wrapper.seed() 只把 seed 往下传给
    #   inner env，从不碰 wrapper 自己的 action_space。
    #   ⇒ 最外层的 env.action_space（那个新建的 Box）从未被播种，其 np_random
    #     在首次 .sample() 时由 gym.utils.seeding.np_random(None) 用 os.urandom 取熵。
    # 实测证据: STUDY/scripts/check_seeding.py 连续两次运行输出不同 ——
    #     sample[0] = [-0.116318754852, 0.919082343578, ...]
    #     sample[0] = [ 0.776469647884, -0.012249834836, ...]
    #   （对比: env.reset() 的初始观测两次完全一致，说明只有 action_space 漏了种。）
    # 后果: 训练前 start_training=5000 步用的是随机动作
    #   （`if i < FLAGS.start_training: action = env.action_space.sample()`），
    #   这 5000 条 transition 进池后整条训练轨迹随之发散。
    #   ── 与"离线数据集未播种"并列的两个根因之一。
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    # 离线数据集: D4RLDataset 包装 d4rl.qlearning_dataset(env)，首次调用自动下载到 ~/.d4rl。
    # "binary" 任务（Adroit）走 BinaryDataset（AWAC 格式，需手动下载）。
    # not ideal, but works for now:
    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    # --- [关键修复 2026-09-15] 给离线数据集补种 ---
    # 上游缺陷: Dataset.__init__(dataset_dict, seed=None) 的默认 seed 是 None，
    #   而 D4RLDataset/BinaryDataset 调 super().__init__(dataset_dict) 时没传 seed，
    #   于是 self._np_random 保持 None；首次访问 np_random 属性时走 self.seed()（无参）
    #   → gym.utils.seeding.np_random(None) → create_seed(None) 用 os.urandom 取熵。
    # 实测后果（这是本轮最重的发现）: 同一 seed、同一配置的两次 run，25k 末点评估值
    #   7797.5 vs 7019.9，差 10%。因为 --seed 只控制了 env(第175行) / agent(第195行) /
    #   replay_buffer(第203行) 的 RNG，**离线数据的采样流完全不受控**；
    #   而每步要抽 batch_size×utd_ratio×offline_ratio = 256 × 20 × 0.5 = 2560 条离线样本，
    #   这个不受控的流足以让整条训练轨迹在几步之内发散。
    # 为什么之前没暴露: 三个 RNG 里只漏了这一个，其余全部正确播种，
    #   所以现象是"看起来有种子"但"同种子跑不出同结果"，很容易被误判成 GPU 非确定性。
    # 影响: ① 任何"固定 seed 即可复现"的假设都不成立；
    #       ② 报告的 seed 极差里混着采样噪声，组间小差异（<10%）无法与噪声区分；
    #       ③ R1 的 A/B/C/D 大差异（数十个百分点）仍成立，只是误差棒被低估。
    # 修复: 显式补种，使 (seed) 完全决定一次 run。
    ds.seed(FLAGS.seed)

    # 评估环境单独建一个（seed+42），与训练环境解耦，避免评估污染训练状态。
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)
    # 同理补种外层 space（评估用确定性动作，影响较小，但保持一致以免留下隐患）
    eval_env.action_space.seed(FLAGS.seed + 42)
    eval_env.observation_space.seed(FLAGS.seed + 42)

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
            # 边界保护（实验A需要）: offline_ratio=0/1 时某一侧是空 batch（0 条），
            # combine 的交错赋值 tmp[0::2]=空数组 会抛 ValueError（broadcast 失败）。
            # 退化规则：某侧为空 → 只用非空侧，与比值取极限的语义一致
            # （0 → 纯 online，1 → 纯 offline）。
            n_offline = int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            n_online = int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            if n_offline == 0:
                # 浅解冻 FrozenDict → 普通 dict，兼容下方 antmaze 分支的原地修改
                batch = dict(online_batch)
            elif n_online == 0:
                batch = dict(offline_batch)
            else:
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
        # 每 eval_interval 步用确定性策略（eval_actions，取分布 mode）跑 eval_episodes 个
        # episode（R2 起: interval=2500, episodes=20），记录 evaluation/return。
        # 评估用确定性、训练用随机采样是 RL 标准做法。
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
