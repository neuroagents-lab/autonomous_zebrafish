import unittest
import torch
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from feature_net import LstmFeatureExtractor
from vision_net import ConvReluNet  # Assuming ConvReluNet is in vision_net.py
import numpy as np

class TestLstmFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Define observation space
        image_space = Box(low=0, high=255, shape=(3, 25, 25), dtype=np.float32)  # Matches ConvReluNet input
        proprio_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = Dict({"image": image_space, "proprio": proprio_space})

        # Define model parameters
        self.proprio_input_size = proprio_space.shape[0]
        self.conv_output_size = 128  # Output size from ConvReluNet
        self.mlp_arch = [256, 128]
        self.lstm_hidden_size = 64

        # Initialize LstmFeatureExtractor
        self.feature_extractor = LstmFeatureExtractor(
            observation_space=self.observation_space,
            proprio_input_size=self.proprio_input_size,
            conv_output_size=self.conv_output_size,
            mlp_arch=self.mlp_arch,
            lstm_hidden_size=self.lstm_hidden_size,
        )

    def test_feature_extractor_initialization(self):
        # Check that the feature extractor initializes without errors
        self.assertIsInstance(self.feature_extractor, LstmFeatureExtractor)

    def test_forward_pass(self):
        # Generate mock inputs
        batch_size = 8
        image_input = torch.rand(batch_size, 3, 25, 25)  # Adjusted to match ConvReluNet's input size
        proprio_input = torch.rand(batch_size, self.proprio_input_size)
        observations = {"image": image_input, "proprio": proprio_input}

        # Forward pass
        output = self.feature_extractor(observations)

        # Check output shape
        expected_output_size = self.lstm_hidden_size
        self.assertEqual(output.shape, (batch_size, expected_output_size))

    def test_batch_size_mismatch(self):
        # Generate inputs with mismatched batch sizes
        image_input = torch.rand(5, 3, 25, 25)
        proprio_input = torch.rand(8, self.proprio_input_size)
        observations = {"image": image_input, "proprio": proprio_input}

        # Assert that a batch size mismatch raises an error
        with self.assertRaises(AssertionError):
            self.feature_extractor(observations)

    def test_combined_feature_size(self):
        # Check combined output size matches expectation
        self.assertEqual(
            self.feature_extractor.features_dim,
            self.conv_output_size + self.lstm_hidden_size,
        )


if __name__ == "__main__":
    unittest.main()