import warnings
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Dict

import cv2
import gymnasium as gym
import numpy as np
import torch
from tianshou.env import ShmemVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import WandbLogger
from tianshou.utils.net.discrete import NoisyLinear
from torch import Tensor, nn
from torch.nn import functional as F

try:
    import envpool
except ImportError:
    envpool = None


def _parse_reset_result(reset_result):
    contains_info = isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict)
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.

    No-op is assumed to be action 0.

    :param gym.Env env: the environment to wrap.
    :param int noop_max: the maximum value of no-ops to run.
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        _, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        if hasattr(self.unwrapped.np_random, "integers"):
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, rew, done, info = step_result
            else:
                obs, rew, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw observations
      (for max pooling across time steps).

    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action.

        Repeat action, sum reward, and max over last observations.
        """
        obs_list, total_reward = [], 0.0
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return max_frame, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    It helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True

        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
            term = True
        self.lives = lives
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Related discussion: https://github.com/openai/baselines/issues/240.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        obs = self.env.step(1)[0]
        return (obs, {}) if return_info else obs


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return (observation - self.bias) / self.scale


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (n_frames, *env.observation_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else self._get_ob()

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)


def wrap_deepmind(
    env_id,
    episode_life=True,
    clip_rewards=True,
    frame_stack=4,
    scale=False,
    warp_frame=True,
):
    """Configure environment for DeepMind-style Atari.

    The observation is channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


def make_atari_env(task, seed, training_num, test_num, **kwargs):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        if kwargs.get("scale", 0):
            warnings.warn(
                "EnvPool does not include ScaledFloatFrame wrapper, "
                "please set `x = x / 255.0` inside CNN network's forward function.",
            )
        # parameters convertion
        train_envs = env = envpool.make_gymnasium(
            task.replace("NoFrameskip-v4", "-v5"),
            num_envs=training_num,
            seed=seed,
            episodic_life=True,
            reward_clip=True,
            full_action_space=True,
            stack_num=kwargs.get("frame_stack", 4),
        )
        test_envs = envpool.make_gymnasium(
            task.replace("NoFrameskip-v4", "-v5"),
            num_envs=test_num,
            seed=seed,
            episodic_life=False,
            reward_clip=False,
            full_action_space=True,
            stack_num=kwargs.get("frame_stack", 4),
        )
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) to run Atari games more efficiently.",
        )
        env = wrap_deepmind(task, **kwargs)
        train_envs = ShmemVectorEnv(
            [lambda: wrap_deepmind(task, episode_life=True, clip_rewards=True, **kwargs) for _ in range(training_num)],
        )
        test_envs = ShmemVectorEnv(
            [lambda: wrap_deepmind(task, episode_life=False, clip_rewards=False, **kwargs) for _ in range(test_num)],
        )
        env.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    return env, train_envs, test_envs


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def scale_obs(module: type[nn.Module], denom: float = 255.0) -> type[nn.Module]:
    class scaled_module(module):
        def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: Any | None = None,
            info: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, Any]:
            if info is None:
                info = {}
            return super().forward(obs / denom, state, info)

    return scaled_module


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, int(np.prod(action_shape)))),
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if info is None:
            info = {}
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_atoms], device)
        self.num_atoms = num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        if info is None:
            info = {}
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: str | int | torch.device = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w, action_shape, device, features_only=True)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512),
            nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms),
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512),
                nn.ReLU(inplace=True),
                linear(512, self.num_atoms),
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        if info is None:
            info = {}
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state


class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_quantiles: int = 200,
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(c, h, w, [self.action_num * num_quantiles], device)
        self.num_quantiles = num_quantiles

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        if info is None:
            info = {}
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state


class CReLU(torch.nn.ReLU):
    """CReLU activation function"""

    dim: int = -1

    def forward(self, input: Tensor) -> Tensor:
        positive = F.relu(input, inplace=self.inplace)
        negative = F.relu(-input, inplace=self.inplace)
        return torch.cat((positive, negative), dim=self.dim)


class MultiVisitWandbLogger(WandbLogger):
    def __init__(self, names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = names
        self.current_name = None
        self.base_env_steps = {name: 0 for name in names}
        self.base_grad_steps = {name: 0 for name in names}

    @property
    def global_base_env_step(self):
        return sum(self.base_env_steps.values())

    @property
    def task_base_env_step(self):
        return self.base_env_steps[self.current_name]

    @property
    def global_base_grad_step(self):
        return sum(self.base_grad_steps.values())

    @property
    def task_base_grad_step(self):
        return self.base_grad_steps[self.current_name]

    def add_to_base_env_step(self, name, step):
        self.base_env_steps[name] = step + self.base_env_steps.get(name, 0)

    def add_to_base_grad_step(self, name, step):
        self.base_grad_steps[name] = step + self.base_grad_steps.get(name, 0)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        global_step = self.global_base_env_step + step
        task_step = self.task_base_env_step + step

        if collect_result["n/ep"] > 0:
            if global_step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    f"train/task_step/{self.current_name}": task_step,
                    f"train/episode/{self.current_name}": collect_result["n/ep"],
                    f"train/reward/{self.current_name}": collect_result["rew"],
                    f"train/length/{self.current_name}": collect_result["len"],
                }
                self.write("train/global_step", global_step, log_data)
                self.last_log_train_step = global_step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        global_step = self.global_base_env_step + step
        task_step = self.task_base_env_step + step

        assert collect_result["n/ep"] > 0
        if global_step - self.last_log_test_step >= self.test_interval:
            log_data = {
                f"test/task_step/{self.current_name}": task_step,
                f"test/reward/{self.current_name}": collect_result["rew"],
                f"test/length/{self.current_name}": collect_result["len"],
                f"test/reward_std/{self.current_name}": collect_result["rew_std"],
                f"test/length_std/{self.current_name}": collect_result["len_std"],
            }
            self.write("test/global_step", global_step, log_data)
            self.last_log_test_step = global_step

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        global_step = self.global_base_grad_step + step
        task_step = self.task_base_grad_step + step
        if global_step - self.last_log_update_step >= self.update_interval:
            log_data = {f"update/{k}/{self.current_name}": v for k, v in update_result.items()}
            log_data[f"update/task_step/{self.current_name}"] = task_step
            self.write("update/gradient_step", global_step, log_data)
            self.last_log_update_step = global_step


class CReLUDQN(DQN):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            CReLU(dim=1, inplace=True),
            layer_init(nn.Conv2d(32 * 2, 64, kernel_size=4, stride=2)),
            CReLU(dim=1, inplace=True),
            layer_init(nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1)),
            CReLU(dim=1, inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, 512)),
                CReLU(dim=-1, inplace=True),
                layer_init(nn.Linear(512 * 2, int(np.prod(action_shape)))),
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(self.output_dim, output_dim)),
                CReLU(inplace=True),
            )
            self.output_dim = output_dim * 2

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if info is None:
            info = {}
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class CReLURainbow(Rainbow):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: str | int | torch.device = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w, action_shape, num_atoms, noisy_std, device, is_dueling, is_noisy)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512),
            CReLU(dim=-1, inplace=True),
            linear(512 * 2, self.action_num * self.num_atoms),
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512),
                CReLU(dim=-1, inplace=True),
                linear(512 * 2, self.num_atoms),
            )
        self.output_dim = self.action_num * self.num_atoms * 2


class OurOffpolicyTrainer(OffpolicyTrainer):
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        def norm(last_params, model, ord):
            total_norm = 0.0
            for last_p, p in zip(last_params, model.parameters()):
                if ord == 0:
                    param_norm = torch.count_nonzero(p - last_p)
                elif ord > 0.0:
                    param_norm = torch.linalg.vector_norm(p - last_p, ord) ** ord
                total_norm += param_norm.item()
            if ord > 0.0:
                total_norm = total_norm ** (1.0 / ord)
            return total_norm

        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            # last_params = [p.clone() for p in self.policy.model.parameters()]
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)

            # losses.update(
                # {
                    # "update_norm_l0": norm(last_params, self.policy.model, 0.0),
                    # "update_norm_l1": norm(last_params, self.policy.model, 1.0),
                    # "update_norm_l2": norm(last_params, self.policy.model, 2.0),
                # }
            # )

            self.log_update_data(data, losses)
