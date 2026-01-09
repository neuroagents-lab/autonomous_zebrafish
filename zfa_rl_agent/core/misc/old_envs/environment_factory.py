#import os
#os.environ["MUJOCO_GL"] = "egl"
#os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import numpy as np
from gymnasium.envs.registration import register
import dm_control.suite.swimmer as swimmer
import zfa_rl_agent.core.environments.zebrafish as zebrafish
from dm_control.rl import control
from dm_control.suite.wrappers import pixels
from shimmy import DmControlCompatibilityV0
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from zfa_rl_agent.core.environments.kelp_forest import generate_xml


def make_kelp_env(n_links=6, 
                     time_limit=swimmer._DEFAULT_TIME_LIMIT,
                     return_frame_wrapped = False,
                     render_args = {'height': 64, 'width': 64, 'camera_id': 2},
                     **environment_kwargs
                     ):
    """Returns the Swim task for a n-link swimmer.
    Args:
        n_links: int, number of links in the swimmer.
        time_limit: float, time limit of the episode. Default is 30.
        return_frame_wrapped: bool, if True, returns a VecFrameStack wrapped environment.
        render_args: dict, parameters of the vision sensor.
        environment_kwargs: dict, additional arguments to pass to the environment."""
    
    # additional args are density and obstacle_size
    model_string, assets = generate_xml(n_links)
    physics = zebrafish.Physics.from_xml_string(model_string, assets)
    task = zebrafish.Swimmer() # this is the base swimmer task for now

    env = control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
    )

    env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_args)
    env = DmControlCompatibilityV0(env)  #, render_mode = 'rgb_array')

    if return_frame_wrapped:
        dummy_env = DummyVecEnv([lambda: env])
        stacked_env = VecFrameStack(dummy_env, n_stack=2)
        return stacked_env
    
    return env

def make_stream_env(n_links=6, 
                     time_limit=swimmer._DEFAULT_TIME_LIMIT,
                     return_frame_wrapped = False,
                     render_args = {'height': 64, 'width': 64, 'camera_id': 2},
                     **environment_kwargs
                     ):
    """Returns the Swim task for a n-link swimmer.
    Args:
        n_links: int, number of links in the swimmer.
        time_limit: float, time limit of the episode. Default is 30.
        return_frame_wrapped: bool, if True, returns a VecFrameStack wrapped environment.
        render_args: dict, parameters of the vision sensor.
        environment_kwargs: dict, additional arguments to pass to the environment."""
    
    # additional args are density and obstacle_size
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets)
    task = swimmer.Swimmer() # this is the base swimmer task for now

    env = control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
    )

    env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_args)
    env = DmControlCompatibilityV0(env)  #, render_mode = 'rgb_array')

    if return_frame_wrapped:
        dummy_env = DummyVecEnv([lambda: env])
        stacked_env = VecFrameStack(dummy_env, n_stack=2)
        return stacked_env
    
    return env

    # def get_observation(self, physics):
    
    #     # Get the original observations
    #     obs = super().get_observation(physics)
    #     # Add pixel-based observations
    #     pixel_data = physics.render(
    #         camera_id=self._camera_id, 
    #         height=self._image_height, 
    #         width=self._image_width
    #     )
    #     obs['pixels'] = pixel_data
    #     return obs

