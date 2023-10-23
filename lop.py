import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-ids", type=Iterable[str], default=[
        "ALE/Alien-v5",
        "ALE/Atlantis-v5",
        "ALE/Boxing-v5",
        "ALE/Breakout-v5",
        "ALE/Centipede-v5",
    ], help="the ids of the environment")
    parser.add_argument("--game-timesteps", type=int, default=4000000,
        help="total timesteps of a gameplay")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=400000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--epsilon", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--learning-starts", type=int, default=20000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--visits", type=int, default=4,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", full_action_space=True)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, full_action_space=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 18),  # 18 is the number of atari actions
        )

    def forward(self, x):
        return self.network(x / 255.0)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = parse_args()
    run_name = f"LoP-{args.seed}"

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup

    q_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),  # env.observation_space,
        gym.spaces.Discrete(18),  # env.action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game

    global_steps = 0
    for visit in range(args.visits):
        for env_id in args.env_ids:
            envs = gym.vector.SyncVectorEnv(
                [make_env(env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

            obs, _ = envs.reset(seed=args.seed)
            for _ in range(args.game_timesteps):
                # ALGO LOGIC: put action logic here
                epsilon = args.epsilon
                if random.random() < epsilon:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    q_values = q_network(torch.Tensor(obs).to(device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        # Skip the envs that are not done
                        if "episode" not in info:
                            continue
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(f"V{visit}/{env_id}/episodic_return", info["episode"]["r"], global_steps)
                        writer.add_scalar(f"V{visit}/{env_id}/episodic_length", info["episode"]["l"], global_steps)
                        writer.add_scalar(f"V{visit}/{env_id}/epsilon", epsilon, global_steps)
                        break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                # ALGO LOGIC: training.
                if global_steps > args.learning_starts:
                    if global_steps % args.train_frequency == 0:
                        data = rb.sample(args.batch_size)
                        with torch.no_grad():
                            target_max, _ = target_network(data.next_observations).max(dim=1)
                            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                        loss = F.mse_loss(td_target, old_val)

                        if global_steps % 100 == 0:
                            writer.add_scalar(f"V{visit}/{env_id}/td_loss", loss, global_steps)
                            writer.add_scalar(f"V{visit}/{env_id}/q_values", old_val.mean().item(), global_steps)
                            # print("SPS:", int(global_step / (time.time() - start_time)))
                            writer.add_scalar(
                                f"V{visit}/{env_id}/SPS", int(global_steps / (time.time() - start_time)), global_steps
                            )

                        # optimize the model
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # update target network
                    if global_steps % args.target_network_frequency == 0:
                        for target_network_param, q_network_param in zip(
                            target_network.parameters(), q_network.parameters()
                        ):
                            target_network_param.data.copy_(
                                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                            )
                global_steps += 1

            envs.close()
    writer.close()
