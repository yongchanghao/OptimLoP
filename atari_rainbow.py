import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

from atari_utils import Rainbow, make_atari_env


class MultiVisitWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_visit = None
        self.name = None

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    f"train/episode/{self.name}/v{self.num_visit}": collect_result["n/ep"],
                    f"train/reward/{self.name}/v{self.num_visit}": collect_result["rew"],
                    f"train/length/{self.name}/v{self.num_visit}": collect_result["len"],
                }
                self.write(f"train/env_step/{self.name}/v{self.num_visit}", step, log_data)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                f"test/env_step/{self.name}/v{self.num_visit}": step,
                f"test/reward/{self.name}/v{self.num_visit}": collect_result["rew"],
                f"test/length/{self.name}/v{self.num_visit}": collect_result["len"],
                f"test/reward_std/{self.name}/v{self.num_visit}": collect_result["rew_std"],
                f"test/length_std/{self.name}/v{self.num_visit}": collect_result["len_std"],
            }
            self.write(f"test/env_step/{self.name}/v{self.num_visit}", step, log_data)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = {f"update/{k}/{self.name}/v{self.num_visit}": v for k, v in update_result.items()}
            self.write(f"update/gradient_step/{self.name}/v{self.num_visit}", step, log_data)
            self.last_log_update_step = step


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "Alien-v5",
            "Atlantis-v5",
            "Boxing-v5",
            "Breakout-v5",
            "Centipede-v5",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--num-visit", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument("--save-buffer-name", type=str, default=None)
    return parser.parse_args()


def rainbow(task, net, optim, policy, buffer, logger, log_path, num_visit, args=get_args()):
    env, train_envs, test_envs = make_atari_env(
        task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    assert args.state_shape == (env.observation_space.shape or env.observation_space.n)
    assert args.action_shape == (env.action_space.shape or env.action_space.n)

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write(f"train/env_step/v{num_visit}", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write(f"train/env_step/v{num_visit}", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()

    pprint.pprint(result)


def test_rainbow(args=get_args()):
    dummy_env, _, _ = make_atari_env(
        args.tasks[0],
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = dummy_env.observation_space.shape or dummy_env.observation_space.n
    args.action_shape = dummy_env.action_space.shape or dummy_env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = Rainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        action_space=dummy_env.action_space,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.training_num,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.training_num,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm,
        )

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "rainbow"
    log_name = os.path.join(args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger = MultiVisitWandbLogger(
        save_interval=1,
        name=log_name.replace(os.path.sep, "__"),
        run_id=args.resume_id,
        config=args,
        project=args.wandb_project,
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    for visit in range(args.num_visit):
        for i, task in enumerate(args.tasks):
            print(f"Task {i}: {task}, visit {visit}")
            logger.num_visit = visit
            logger.name = task
            rainbow(task, net, optim, policy, buffer, logger, log_path, i, args=args)


if __name__ == "__main__":
    test_rainbow(get_args())
