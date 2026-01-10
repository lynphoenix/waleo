"""PPO 训练 - 支持并行环境

基于 ManiSkill 原版 ppo_rgb.py
"""

from collections import defaultdict
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "WaleoManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    render_mode: str = "cameras"

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    robot_uids: str = "panda"
    include_state: bool = True
    total_timesteps: int = 25000000
    learning_rate: float = 1e-4
    num_envs: int = 512
    num_eval_envs: int = 8
    num_steps: int = 100
    num_eval_steps: int = 100
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.2
    reward_scale: float = 1.0
    eval_freq: int = 25
    image_size: int = 64

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray(object):
    """字典数组，用于存储批次观察数据"""
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            # Handle gymnasium Dict spaces
            from gymnasium import spaces as gym_spaces
            if isinstance(element_space, gym_spaces.Dict):
                self.data = {}
                for k, v in element_space.items():
                    if isinstance(v, dict):
                        self.data[k] = DictArray(buffer_shape, v, device=device)
                    else:
                        dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                                torch.uint8 if v.dtype == np.uint8 else v.dtype)
                        self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)
            else:
                raise ValueError(f"Expected dict space, got {type(element_space)}", flush=True)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class NatureCNN(nn.Module):
    """Nature CNN 架构用于图像特征提取"""
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # CNN 层
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255.0  # 归一化
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    """PPO 智能体"""
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=25000000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--test", action="store_true", flush=True)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # 使用默认参数
    args_obj = Args()
    args_obj.total_timesteps = 1000000 if args.test else args.total_steps
    args_obj.num_envs = args.num_envs
    args_obj.learning_rate = args.learning_rate

    # 计算批次参数
    args_obj.batch_size = int(args_obj.num_envs * args_obj.num_steps)
    args_obj.minibatch_size = int(args_obj.batch_size // args_obj.num_minibatches)
    args_obj.num_iterations = args_obj.total_timesteps // args_obj.batch_size

    if args_obj.exp_name is None:
        args_obj.exp_name = "train_ppo_visual_vectorized"
        run_name = f"{args_obj.env_id}_{args_obj.robot_uids}_{args_obj.exp_name}_{args_obj.seed}_{int(time.time())}"
    else:
        run_name = args_obj.exp_name

    print("=" * 80)
    print("PPO 训练 - 向量化版本", flush=True)
    print("=" * 80)
    print(f"环境: {args_obj.env_id}", flush=True)
    print(f"机器人: {args_obj.robot_uids}", flush=True)
    print(f"并行环境数: {args_obj.num_envs}", flush=True)
    print(f"批次大小: {args_obj.batch_size}", flush=True)
    print(f"小批次大小: {args_obj.minibatch_size}", flush=True)
    print(f"总迭代数: {args_obj.num_iterations}", flush=True)
    print(f"学习率: {args_obj.learning_rate}", flush=True)
    print("=" * 80)
    print()

    # Seeding
    random.seed(args_obj.seed)
    np.random.seed(args_obj.seed)
    torch.manual_seed(args_obj.seed)
    torch.backends.cudnn.deterministic = args_obj.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args_obj.cuda else "cpu", flush=True)
    print(f"使用设备: {device}", flush=True)
    print()

    # Env setup
    env_kwargs = dict(obs_mode="rgb", reward_mode="normalized_dense", render_mode=args_obj.render_mode)
    if args_obj.control_mode is not None:
        env_kwargs["control_mode"] = args_obj.control_mode

    print("创建环境...", flush=True)
    envs = gym.make(args_obj.env_id, num_envs=args_obj.num_envs, robot_uids=args_obj.robot_uids, **env_kwargs)

    # 应用观察包装器 - 展平 RGBD 观察
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args_obj.include_state)

    # 向量化环境
    envs = ManiSkillVectorEnv(envs, args_obj.num_envs, record_metrics=False)

    # 评估环境
    eval_envs = gym.make(args_obj.env_id, num_envs=args_obj.num_eval_envs, robot_uids=args_obj.robot_uids, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args_obj.include_state)
    eval_envs = ManiSkillVectorEnv(eval_envs, args_obj.num_eval_envs, record_metrics=False)

    print(f"✓ 训练环境: {args_obj.num_envs} 个并行环境", flush=True)
    print(f"✓ 评估环境: {args_obj.num_eval_envs} 个并行环境", flush=True)
    print()

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Create agent
    next_obs, _ = envs.reset(seed=args_obj.seed)

    print(f"观察空间: {envs.single_observation_space}", flush=True)
    print(f"动作空间: {envs.single_action_space}", flush=True)
    print()

    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args_obj.learning_rate, eps=1e-5)

    if args_obj.checkpoint:
        agent.load_state_dict(torch.load(args_obj.checkpoint))

    print("开始训练...", flush=True)
    print()

    # Storage setup
    obs = DictArray((args_obj.num_steps, args_obj.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args_obj.num_steps, args_obj.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args_obj.num_steps, args_obj.num_envs)).to(device)
    rewards = torch.zeros((args_obj.num_steps, args_obj.num_envs)).to(device)
    dones = torch.zeros((args_obj.num_steps, args_obj.num_envs)).to(device)
    values = torch.zeros((args_obj.num_steps, args_obj.num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args_obj.seed)
    eval_obs, _ = eval_envs.reset(seed=args_obj.seed)
    next_done = torch.zeros(args_obj.num_envs, device=device)

    cumulative_times = defaultdict(float)

    for iteration in range(1, args_obj.num_iterations + 1):
        print(f"迭代: {iteration}/{args_obj.num_iterations}, global_step={global_step}", flush=True)

        # ==================== 评估 ====================
        if iteration % args_obj.eval_freq == 1:
            print("评估中...", flush=True)
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0

            for _ in range(args_obj.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        agent.get_action(eval_obs, deterministic=True)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos.get("final_info", {}).get("episode", {}).items():
                            eval_metrics[k].append(v)

            print(f"评估完成: {args_obj.num_eval_steps * args_obj.num_eval_envs} 步, {num_episodes} 回合", flush=True)

            if eval_metrics:
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    print(f"  eval/{k}: {mean:.4f}", flush=True)

            eval_time = time.perf_counter() - stime
            cumulative_times["eval_time"] += eval_time
            print(f"  评估时间: {eval_time:.2f}s", flush=True)

        # 保存检查点
        if args_obj.save_model and iteration % args_obj.eval_freq == 1:
            Path("checkpoints").mkdir(exist_ok=True)
            model_path = f"checkpoints/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"✓ 保存模型: {model_path}", flush=True)

        # Annealing learning rate
        if args_obj.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args_obj.num_iterations
            lrnow = frac * args_obj.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ==================== Rollout ====================
        agent.eval()
        rollout_time = time.perf_counter()

        for step in range(args_obj.num_steps):
            global_step += args_obj.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args_obj.reward_scale

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # ==================== 计算 GAE ====================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args_obj.num_steps)):
                if t == args_obj.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args_obj.gamma * nextvalues * next_not_done - values[t]
                advantages[t] = lastgaelam = delta + args_obj.gamma * args_obj.gae_lambda * next_not_done * lastgaelam

            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ==================== PPO 更新 ====================
        agent.train()
        b_inds = np.arange(args_obj.batch_size)
        clipfracs = []
        update_time = time.perf_counter()

        for epoch in range(args_obj.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args_obj.batch_size, args_obj.minibatch_size):
                end = start + args_obj.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args_obj.clip_coef).float().mean().item()]

                if args_obj.target_kl is not None and approx_kl > args_obj.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args_obj.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args_obj.clip_coef, 1 + args_obj.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args_obj.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args_obj.clip_coef, args_obj.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args_obj.ent_coef * entropy_loss + v_loss * args_obj.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args_obj.max_grad_norm)
                optimizer.step()

            if args_obj.target_kl is not None and approx_kl > args_obj.target_kl:
                break

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 打印进度
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        eta = (args_obj.total_timesteps - global_step) / sps if sps > 0 else 0

        print(f"  SPS: {sps} | ETA: {eta/60:.1f}min", flush=True)
        print(f"  policy_loss: {pg_loss.item():.4f}, value_loss: {v_loss.item():.4f}, entropy: {entropy_loss.item():.4f}", flush=True)
        print(f"  approx_kl: {approx_kl.item():.4f}, clipfrac: {np.mean(clipfracs):.4f}, explained_var: {explained_var:.4f}", flush=True)
        print(f"  rollout_time: {rollout_time:.2f}s, update_time: {update_time:.2f}s", flush=True)
        print()

    # 保存最终模型
    if args_obj.save_model:
        Path("checkpoints").mkdir(exist_ok=True)
        model_path = f"checkpoints/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"✓ 保存最终模型: {model_path}", flush=True)

    print()
    print("=" * 80)
    print("训练完成!", flush=True)
    print(f"总步数: {global_step}", flush=True)
    print(f"总时间: {elapsed:.2f}s ({elapsed/60:.1f}min)", flush=True)
    print("=" * 80)

    envs.close()
    eval_envs.close()
