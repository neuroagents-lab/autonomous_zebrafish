import unittest
import torch
from vision_net import ConvReluNet  # Assuming ConvReluNet is in vision_net.py
from feature_net import FcFeatureExtractor  # Assuming FcFeatureExtractor is in feature_extractors.py


class TestFcFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Define parameters for FcFeatureExtractor
        self.conv_output_size = 128  # Output size from ConvReluNet
        self.proprio_input_size = 10
        self.mlp_arch = [64, 32]

        # Initialize the feature extractor
        self.feature_extractor = FcFeatureExtractor(
            conv_output_size=self.conv_output_size,
            proprio_input_size=self.proprio_input_size,
            mlp_arch=self.mlp_arch,
        )

    def test_initialization(self):
        # Ensure the feature extractor initializes without errors
        self.assertIsInstance(self.feature_extractor, FcFeatureExtractor)

        # Check the combined output size
        expected_combined_output_size = self.conv_output_size + self.mlp_arch[-1]
        self.assertEqual(self.feature_extractor.combined_output_size, expected_combined_output_size)

    def test_forward_pass(self):
        # Generate mock inputs
        batch_size = 8
        image_input = torch.rand(batch_size, 3, 25, 25)  # Matches ConvReluNet input size
        proprio_input = torch.rand(batch_size, self.proprio_input_size)

        # Forward pass
        conv_features, proprio_features, combined_features = self.feature_extractor(image_input, proprio_input)

        # Check individual feature shapes
        self.assertEqual(conv_features.shape, (batch_size, self.conv_output_size))
        self.assertEqual(proprio_features.shape, (batch_size, self.mlp_arch[-1]))

        # Check combined feature shape
        expected_combined_shape = (batch_size, self.feature_extractor.combined_output_size)
        self.assertEqual(combined_features.shape, expected_combined_shape)

    def test_batch_size_mismatch(self):
        # Generate inputs with mismatched batch sizes
        image_input = torch.rand(5, 3, 25, 25)
        proprio_input = torch.rand(8, self.proprio_input_size)

        # Assert that a batch size mismatch raises an error
        with self.assertRaises(AssertionError):
            self.feature_extractor(image_input, proprio_input)

    def test_feature_size_mismatch(self):
        # Generate inputs
        batch_size = 8
        image_input = torch.rand(batch_size, 3, 25, 25)
        proprio_input = torch.rand(batch_size, self.proprio_input_size)

        # Manually corrupt the combined_output_size
        self.feature_extractor.combined_output_size += 1

        # Assert that a feature size mismatch raises an error
        with self.assertRaises(AssertionError):
            self.feature_extractor(image_input, proprio_input)


if __name__ == "__main__":
    unittest.main()