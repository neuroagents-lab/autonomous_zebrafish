import ray
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from stable_baselines3.common.env_util import is_wrapped
#from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

@ray.remote
class Actor:
    def __init__(self, env: gym.Env) -> None:
        self.env = env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_info: Optional[Dict[str, Any]] = {}

    def step(self, action: Any) -> Any:
        observation, reward, terminated, truncated, info = self.env.step(action)
        # convert to SB3 VecEnv api
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation, self.reset_info = self.env.reset()
        return (observation, reward, done, info, self.reset_info)

    def reset(self, seed, option) -> Any:
        maybe_options = {"options": option} if option else {}
        observation, self.reset_info = self.env.reset(seed=seed, **maybe_options)
        return (observation, self.reset_info)
    
    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def get_spaces(self) -> Tuple[Any, Any]:
        return (self.observation_space, self.action_space)

    def env_method(self, method_name, args) -> Any:
        return getattr(self.env, method_name)(*args[0], **args[1])
    
    def get_attr(self, attr_name: str) -> Any:
        return getattr(self.env, attr_name)

    def set_attr(self, attr_name: str, value: Any) -> None:
        setattr(self.env, attr_name, value)

    def is_wrapped(self, wrapper_class: Type[gym.Wrapper]) -> bool:
        return is_wrapped(self.env, wrapper_class)


class DistVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    :param env_fns: Environments to run in subprocesses

    Notes
    ------
    `Actor` is a support class for `DistVecEnv` that controls remote environments, similar to `_worker` for
    `SubprocEnv`.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.waiting = False
        self.closed = False
        self.ref_steps = []
        self.actors: List[Actor] = [Actor.remote(env=env_fn) for env_fn in env_fns]
        observation_space, action_space = ray.get(self.actors[0].get_spaces.remote())
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.ref_results = [actor.step.remote(action) for actor, action in zip(self.actors, actions)]
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = ray.get(self.ref_results)
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        self.ref_results = []
        for env_idx, actor in enumerate(self.actors):
            self.ref_results.append(actor.reset.remote(self._seeds[env_idx], self._options[env_idx]))
        results = ray.get(self.ref_results)

        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            ray.get(self.ref_results)
        ray.get([actor.close.remote() for actor in self.actors])
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        return ray.get([actor.render.remote("rgb_array") for actor in self.actors])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = self._get_indices(indices)
        return ray.get([self.actors[index].get_attr.remote(attr_name) for index in indices])

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        indices = self._get_indices(indices)
        ray.get([self.actors[index].set_attr.remote(attr_name, value) for index in indices])

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(indices)
        refs = [self.actors[index].env_method.remote(method_name, (method_args, method_kwargs)) for index in indices]
        return ray.get(refs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        indices = self._get_indices(indices)
        return ray.get([self.actors[index].is_wrapped.remote(wrapper_class) for index in indices])


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        #(space, type(space), space.spaces, type(space.spaces))
        #assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return dict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]
