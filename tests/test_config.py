"""Tests for the config module."""

import unittest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kotlin_completion import config


class TestConfig(unittest.TestCase):
    """Test cases for the config module."""
    
    def test_config_constants_exist(self):
        """Test that essential config constants are defined."""
        self.assertTrue(hasattr(config, 'DATA_FOLDER'))
        self.assertTrue(hasattr(config, 'KT_DS_PATH'))
        self.assertTrue(hasattr(config, 'PY_DS_PATH'))
        self.assertTrue(hasattr(config, 'DEFAULT_MODEL_NAME'))
        
    def test_config_values(self):
        """Test that config values are reasonable."""
        self.assertIsInstance(config.DEFAULT_LEARNING_RATE, float)
        self.assertGreater(config.DEFAULT_LEARNING_RATE, 0)
        self.assertIsInstance(config.DEFAULT_NUM_EPOCHS, int)
        self.assertGreater(config.DEFAULT_NUM_EPOCHS, 0)
        self.assertIsInstance(config.RANDOM_SEED, int)


if __name__ == '__main__':
    unittest.main() 