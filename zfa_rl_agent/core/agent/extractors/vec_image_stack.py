from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

class VecImageStack(VecEnvWrapper):
    """
    Frame stacking wrapper that only stacks image observations in Dict observation spaces.
    Other observation types remain unstacked.
    
    :param venv: Vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param image_key: Key in the observation dict that contains the image to be stacked
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
    """

    def __init__(
        self, 
        venv: VecEnv, 
        n_stack: int, 
        image_key: str,
        channels_order: Optional[str] = None
    ) -> None:
        assert isinstance(
            venv.observation_space, spaces.Dict
        ), "ImageOnlyVecFrameStack requires Dict observation spaces"
        
        self.image_key = image_key
        self.n_stack = n_stack
        
        # Create a modified observation space
        modified_spaces = venv.observation_space.spaces.copy()
        
        # Create stacked observation only for the image key
        assert image_key in modified_spaces, f"Key '{image_key}' not found in observation space"
        
        # Create a stacked observation for the image key
        self.stacked_observation = StackedObservations(
            venv.num_envs, 
            n_stack, 
            venv.observation_space.spaces[image_key],
            channels_order
        )
        
        # Update the space for the image key
        modified_spaces[image_key] = self.stacked_observation.stacked_observation_space
            
        # Create the new observation space
        observation_space = spaces.Dict(modified_spaces)
        super().__init__(venv, observation_space=observation_space)
        
    def step_wait(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        observations, rewards, dones, infos = self.venv.step_wait()
        
        # Only stack the image observation
        if self.image_key in observations:
            # Create sub_infos for the image key's terminal observations
            sub_infos = [
                {"terminal_observation": info["terminal_observation"][self.image_key]} 
                if "terminal_observation" in info else {}
                for info in infos
            ]
            
            # Update the stacked observation for the image key
            observations[self.image_key], sub_infos = self.stacked_observation.update(
                observations[self.image_key], dones, sub_infos
            )
            
            # Update the terminal observations in the infos
            for i, info in enumerate(infos):
                if "terminal_observation" in info:
                    info["terminal_observation"][self.image_key] = sub_infos[i].get(
                        "terminal_observation", info["terminal_observation"][self.image_key]
                    )
                
        return observations, rewards, dones, infos
        
    def reset(self) -> dict[str, np.ndarray]:
        """
        Reset all environments
        """
        observations = self.venv.reset()
        
        # Only stack the image observation
        if self.image_key in observations:
            observations[self.image_key] = self.stacked_observation.reset(observations[self.image_key])
                
        return observations



