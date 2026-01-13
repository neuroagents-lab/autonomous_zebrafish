from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
import torch.nn as nn
import torch
from stable_baselines3.common.type_aliases import TensorDict
from zfa_rl_agent.core.agent.extractors.vision_net import Net_3_layers

class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        mlp_output_dim: int = 64,
        net_arch: list[int] = [64, 64], 
        normalized_image: bool = True,
        flow: bool = True,
        pass_state: bool = True,
    ) -> None:
    
        super().__init__(observation_space, features_dim=1)
        
        self.pass_state = pass_state
        self._observation_space = observation_space
        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if flow: 
                    print('Using FlowNet')
                    extractors[key] = FlowNet(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)    
                else:
                    extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
               extractors[key] = MLP(subspace, features_dim=mlp_output_dim, layer_sizes=net_arch)
               total_concat_size += mlp_output_dim

        if pass_state:
            total_concat_size += observation_space.spaces['proprio'].shape[0]
        self.extractors = nn.ModuleDict(extractors)
        print(total_concat_size)
        # Update the features dim manually
        self._features_dim = total_concat_size
        self.features = None
    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        
        if self.pass_state:
            #print("using privelege")
            #proprio_state = torch.tensor(observations['proprio'][:, :25]).clone().detach()
            proprio_state = observations['proprio'].clone().detach()
            encoded_tensor_list.append(proprio_state)
        #print(encoded_tensor_list[0].shape, encoded_tensor_list[1].shape, encoded_tensor_list[2].shape)
        x = torch.cat(encoded_tensor_list, dim=1) # should dim=-1? 
        self.features = x
        return x 
    
    
class FlowNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        pretrained_path: str = "/data/group_data/neuroagents_lab/models/zfa/optic_flow_64_lr0.01_wd0.0001_best.pth",
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        self.n_input_channels = observation_space.shape[0]

        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "Can't use FlowCNN with the given space of {observation_space}\n")

        self.cnn = Net_3_layers()

        # Load pretrained weights if path is provided
        if pretrained_path:
            # Load the state dict
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cnn.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

        
        # Freeze the CNN parameters
        for param in self.cnn.parameters():
            param.requires_grad = False

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            if sample.shape[1] == 3:
                sample = torch.cat([sample, sample], dim=1)
            n_flatten = self.cnn(sample).shape[1]

        
        # with torch.no_grad():
        #     n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[1] == 3:
            observations = torch.cat([observations, observations], dim=1)
        return self.linear(self.cnn(observations))
    
class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        self.n_input_channels = observation_space.shape[0] #// 2

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
    #     with torch.no_grad():
    #         n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[:self.n_input_channels][None]).float()).shape[1]

    #     self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    #     #(observations.shape)
    #     return self.linear(self.cnn(observations[:, :self.n_input_channels])) # only take the first frame
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class MLP(BaseFeaturesExtractor):
    """
    MLP for proprioceptive data. Processes one observation in the frame stack input. 
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 32,
        layer_sizes: list[int] = [64, 64],
        ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        self.input_size = observation_space.shape[0] #// 2

        layers = []
        input_size = self.input_size
        for size in layer_sizes[:-1]:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers.append(nn.Linear(input_size, features_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        #assert x.shape[-1] > self.input_size, "Forward method expects a frame stack input"
        #return self.mlp(x[:, :self.input_size]) # only take the first frame
        return self.mlp(x)
