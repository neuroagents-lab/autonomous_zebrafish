import torch
import torch.nn as nn
from zfa_rl_agent.core.agent.extractors.vision_net import ConvReluNet, Net_3_layers
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import numpy as np

class MultiModalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, n_frames: int = 2, **kwargs):

        """
        Parameters:
        - observation_space: gym.spaces.Dict, observation space containing image and proprioception keys.
        - conv_output_size: int, size of the output from ConvReluNet.
        - proprio_input_size: int, size of the proprioceptive input vector.
        - mlp_arch: list[int], list defining the architecture of the MLP.
        """
        self.concat = kwargs.get('concat_proprio', False)
        self.mlp_arch = kwargs.get('mlp_arch', [64, 64])
        self.n_frames = n_frames
        self.proprio_pos_size = observation_space['joints'].shape[0] // self.n_frames
        self.proprio_vel_size = observation_space['body_velocities'].shape[0] // self.n_frames
        self.proprio_input_size = self.proprio_pos_size + self.proprio_vel_size

        # patch this
        combined_output_size = 64*64*2 + 2*self.mlp_arch[-1]

        self.test = observation_space
        super().__init__(observation_space = observation_space, features_dim=combined_output_size, **kwargs)
        self._features_dim = combined_output_size
        self.keys = list(self._observation_space.keys())
       
        vision_encoder = Net_3_layers()
        proprio_encoder = self._build_mlp(self.proprio_input_size, self.mlp_arch)
        proprio_pos_encoder = self._build_mlp(self.proprio_pos_size, self.mlp_arch)
        proprio_vel_encoder = self._build_mlp(self.proprio_vel_size, self.mlp_arch)

        if self.concat:
            extractor = {'pixels': vision_encoder, 'proprio': proprio_encoder}
        else:
            extractor = {'pixels': vision_encoder, 'joints': proprio_pos_encoder, 'body_velocities': proprio_vel_encoder}
       
        self.extractors = nn.ModuleDict(extractor)
        
    def forward(self, obs) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Parameters:
        - observations: dict, contains 'image' and 'proprioception' keys.

        Returns:
        - torch.Tensor, combined feature representation of shape (batch_size, combined_output_size).
        """
        encoded_tensor_list = []
        if self.concat:
            proprio_keys = ['joints', 'body_velocities']
            concatenated_observations = torch.cat([obs[key] for key in proprio_keys], dim=1)
            obs = {'proprio': concatenated_observations, 'pixels': obs['pixels']}
        
        processed_obs, processed_obs_next = self.process_obs(obs)
        #for key in self.keys: print(f'{key}', obs[key].shape, processed_obs[key].shape)

        for key, extractor in self.extractors.items():
            if key == 'pixels':
                encoded_tensor_list.append(extractor([processed_obs[key], processed_obs_next[key]])) # frame stacked input
            else:
                #print(processed_obs[key])
                encoded_tensor_list.append(extractor(processed_obs[key]))

        #for tensor in encoded_tensor_list: print(tensor.shape)
        combined_features = torch.cat(encoded_tensor_list, dim=-1)

        assert self.features_dim == combined_features.shape[-1], "Feature size mismatch"
        
        return combined_features

    def _build_mlp(self, input_size: int, layer_sizes: list[int]) -> nn.Sequential:
        """
        Build an MLP with ReLU activations based on a list of layer sizes.

        Parameters:
        - input_size: int, size of the input vector.
        - layer_sizes: list of int, sizes of the MLP hidden layers.

        Returns:
        - nn.Sequential, the constructed MLP.
        """
        layers = []
        for size in layer_sizes[:-1]:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers.append(nn.Linear(input_size, layer_sizes[-1]))
        return nn.Sequential(*layers)   
    
    def get_feature_dim(self):
        return self._features_dim
    
    def process_obs(self, obs_in):
        obs = self._to_tensor(obs_in)
        obs_t = {}
        obs_tp1 = {}

        for key in self.keys:
            #print("PRINTING", "\n", obs, "\n", type(obs), "\n", type(obs[key]))

            t_idx = self._observation_space[key].shape[0] // self.n_frames
            #print(t_idx, key)
            if key == 'pixels':
                #print(self._observation_space[key].shape[0])
                if self._observation_space[key].shape[0] == 64:
                    t_idx = self._observation_space[key].shape[-1] // self.n_frames
                if obs[key].dim() == 5:
                    o_t = obs[key][:, :, :t_idx, ...]
                    o_tp1 = obs[key][:, :, t_idx:2*t_idx, ...]
                elif obs[key].dim() == 4:
                    o_t = obs[key][:, :t_idx, ...]
                    o_tp1 = obs[key][:, t_idx:2*t_idx, ...]
                #print(o_t.shape, o_tp1.shape)

            else:        
                o_t = obs[key][..., :t_idx]
                o_tp1 = obs[key][..., t_idx:]

            obs_t[key] = o_t
            obs_tp1[key] = o_tp1
        return obs_t, obs_tp1 
    
    def _to_tensor(self, obs):
        """
        Moves the observation to the given device.

        :param obs:
        :param device: PyTorch device
        :return: PyTorch tensor of the observation on a desired device.
        """
        #print(type(obs), obs,)
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, dict):
            return {key: torch.as_tensor(obs[key], dtype=torch.float32) for key in obs.keys()}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")
        
    
#######################################################################################################################################
class MultiModalFeatureExtractor_V0(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        """
        Parameters:
        - observation_space: gym.spaces.Dict, observation space containing image and proprioception keys.
        - conv_output_size: int, size of the output from ConvReluNet.
        - proprio_input_size: int, size of the proprioceptive input vector.
        - mlp_arch: list[int], list defining the architecture of the MLP.
        """
        self.concat = kwargs.get('concat', False)

        self.conv_output_size = kwargs.get('conv_output_size', 32)
        self.mlp_arch = kwargs.get('mlp_arch', [64, 64])

        self.proprio_pos_size = observation_space['joints'].shape[0]
        self.proprio_vel_size = observation_space['body_velocities'].shape[0]
        self.proprio_input_size = self.proprio_pos_size + self.proprio_vel_size

        combined_output_size = self.conv_output_size + 2*self.mlp_arch[-1]

        # Initialize the BaseFeaturesExtractor
        super().__init__(observation_space, features_dim=combined_output_size)
        self._features_dim = combined_output_size
        
        self.vision_encoder = ConvReluNet(output_size=self.conv_output_size, training=True)        
        self.proprio_encoder = self._build_mlp(self.proprio_input_size, self.mlp_arch)
        self.proprio_pos_encoder = self._build_mlp(self.proprio_pos_size, self.mlp_arch)
        self.proprio_vel_encoder = self._build_mlp(self.proprio_vel_size, self.mlp_arch)

        if self.concat:
            extractor = {'pixels': self.vision_encoder, 'proprio': self.proprio_encoder}
        else:
            extractor = {'pixels': self.vision_encoder, 'joints': self.proprio_pos_encoder, 'body_velocities': self.proprio_vel_encoder}
       
        self.extractors = nn.ModuleDict(extractor)

    def forward(self, observations) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Parameters:
        - observations: dict, contains 'image' and 'proprioception' keys.

        Returns:
        - torch.Tensor, combined feature representation of shape (batch_size, combined_output_size).
        """
        encoded_tensor_list = []
        if self.concat:
            proprio_keys = ['joints', 'body_velocities']
            concatenated_observations = torch.cat([observations[key] for key in proprio_keys], dim=1)
            obs = {'proprio': concatenated_observations, 'pixels': observations['pixels']}

        obs = observations.copy()
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(obs[key]))

        combined_features = torch.cat(encoded_tensor_list, dim=1)
        return combined_features

    def _build_mlp(self, input_size: int, layer_sizes: list[int]) -> nn.Sequential:
        """
        Build an MLP with ReLU activations based on a list of layer sizes.

        Parameters:
        - input_size: int, size of the input vector.
        - layer_sizes: list of int, sizes of the MLP hidden layers.

        Returns:
        - nn.Sequential, the constructed MLP.
        """
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        return nn.Sequential(*layers)   
    
    def get_feature_dim(self):
        return self._features_dim


