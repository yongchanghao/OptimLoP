import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy
from torch.utils.tensorboard import SummaryWriter

from src.optimizers.cadam_noise import CAdamNoise
from src.optimizers.cadam import CAdam
from src.optimizers.csgd import CSGD
from src.optimizers.lion import Lion
from src.optimizers.csgd_noise import CSGDNoise
from src.utils import (
    MultiVisitWandbLogger,
    OurOffpolicyTrainer,
    Rainbow,
    make_atari_env,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks",
        type=str,
        nargs="+",
        default=[
            "Alien-v5",
            "Boxing-v5",
            "Breakout-v5",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=50000)
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
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
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
    parser.add_argument("--wandb-entity", type=str, default="clean-rl")
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--beta0", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser.parse_args()


def rainbow(task, policy, buffer, logger, log_path, args=get_args()):
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
        def norm(model, ord):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                if ord == 0:
                    param_norm = torch.count_nonzero(p.grad.data)
                elif ord > 0.0:
                    param_norm = torch.linalg.vector_norm(p.grad.data, ord) ** ord
                total_norm += param_norm.item()
            if ord > 0.0:
                total_norm = total_norm ** (1.0 / ord)
            return total_norm

        # nature DQN setting, linear decay in the first 1M steps
        env_step += logger.global_base_env_step
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        # logger.write(
        #     "train/env_step",
        #     env_step,
        #     {
        #         "train/grad_norm_l0": norm(policy.model, 0),
        #         "train/grad_norm_l1": norm(policy.model, 1),
        #         "train/grad_norm_l2": norm(policy.model, 2),
        #     },
        # )
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    trainer = OurOffpolicyTrainer(
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
    )
    result = trainer.run()
    pprint.pprint(result)
    return trainer.env_step, trainer.gradient_step


def test_rainbow(args=get_args()):
    print("Environments:", args.tasks)
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
    if args.optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    elif args.optimizer == "csgd":
        optim = CSGD(net.parameters(), lr=args.lr, betas=(args.beta0, args.momentum))
    elif args.optimizer == "cadam":
        optim = CAdam(net.parameters(), lr=args.lr, betas=(args.beta0, args.momentum, 0.999))
    elif args.optimizer == "lion":
        optim = Lion(net.parameters(), lr=args.lr, betas=(args.beta0, args.momentum))
    elif args.optimizer == "cadam_noise":
        optim = CAdamNoise(net.parameters(), lr=args.lr, betas=(args.beta0, args.momentum, 0.999))
    elif args.optimizer == "csgd_noise":
        optim = CSGDNoise(net.parameters(), lr=args.lr, betas=(args.beta0, args.momentum))
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
    log_name = os.path.join(args.algo_name, args.optimizer, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger = MultiVisitWandbLogger(
        names=args.tasks,
        save_interval=1,
        name=log_name.replace(os.path.sep, "-"),
        run_id=args.resume_id,
        config=args,
        project=args.wandb_project,
        entity=args.wandb_entity,
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)
    for visit in range(args.num_visit):
        for i, task in enumerate(args.tasks):
            print(f"Visit ({visit + 1}/{args.num_visit}), {task} ({i+1}/{len(args.tasks)})")
            logger.current_name = task
            new_env_steps, new_grad_steps = rainbow(task, policy, buffer, logger, log_path, args=args)
            logger.add_to_base_env_step(task, new_env_steps)
            logger.add_to_base_grad_step(task, new_grad_steps)


if __name__ == "__main__":
    test_rainbow(get_args())
