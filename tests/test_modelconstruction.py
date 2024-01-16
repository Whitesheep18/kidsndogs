import unittest  # Built-in Python module for writing and running tests
import torch 
from knd.models.model import DummyNet  # Importing the DummyNet
from omegaconf import OmegaConf

# Defining a test class for the DummyNet model
class TestDummyNetModel(unittest.TestCase):
    # This class inherits from unittest.TestCase providing a framework for writing test cases

    def test_model_initialization(self):
        # Test method to check the initialization of the DummyNet model

        # Creating a mock configuration object using OmegaConf
        cfg = OmegaConf.create({
            'experiment': {
                'lr': 0.0001,  # Mock learning rate
                'n_hidden': 512,  # Mock number of hidden units
                'dropout': 0.2  # Mock dropout rate
            }
        })

        # Instantiating the model with parameters from the mock configuration
        model = DummyNet(lr=cfg.experiment.lr, n_hidden=cfg.experiment.n_hidden, dropout=cfg.experiment.dropout)

        # Asserting that the model's learning rate matches the mock configuration
        self.assertEqual(model.lr, cfg.experiment.lr, "Learning rate mismatch")
        # Asserting that the model's number of hidden units matches the mock configuration
        self.assertEqual(model.n_hidden, cfg.experiment.n_hidden, "Number of hidden units mismatch")
        # Note: Dropout is not directly tested as it's an internal detail of the model's architecture

    def test_model_output_shape(self):
        # Test method to check the output shape of the model

        # Instantiating the model with default parameters
        model = DummyNet()

        # Creating a mock input tensor with the expected shape
        spectorgrams = torch.randn(32, 1, 128, 563)  # Random data simulating the input
        #create a random tensor with a shape of (32, 1, 128, 563)
        spectorgrams = torch.cat([spectorgrams, spectorgrams, spectorgrams], dim=1)  # Adjusting to match expected input channels
        '''
        This concatenates the tensor along the channel dimension (dim=1).
        The original tensor had a single channel, and this operation replicates it two more times along the channel axis.
        After this concatenation, the tensor will have three channels, simulating a 3-channel input, similar to an RGB image.
        '''

        # Getting output from the model
        output = model(spectorgrams)

        # Asserting that the output shape is as expected (batch size, number of classes)
        self.assertEqual(output.shape, (32, 8), "Output shape mismatch")

# Executing the test case when the script is run directly
if __name__ == '__main__':
    unittest.main()
