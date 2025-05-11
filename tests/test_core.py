"""
Unit tests for the core module.

Tests focus on validating the functional flow and interface compatibility
of the core components using minimal test data.
"""

import os
import sys
import tempfile
import unittest
import logging
from pathlib import Path
import numpy as np

from tests.base import BrainStemXBaseTest
from src.core import (
    Config, get_logger, check_file_dependencies, log_image_stats,
    write_img, resample_if_needed, check_mask, _mu_sigma
)

class TestConfig(BrainStemXBaseTest):
    """Tests for Config class in core.py."""

    def test_config_init(self):
        """Test Config initialization with default values."""
        cfg = Config()
        self.assertEqual(cfg.n_jobs, min(os.cpu_count(), 22))
        self.assertFalse(cfg.overwrite)
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.log_level, "INFO")
        self.assertEqual(cfg.primary_sd, 2.0)
        self.assertIsNotNone(cfg.thresholds_sd)

    def test_config_resolve(self):
        """Test Config.resolve() with environment variables."""
        # Set test environment variables
        os.environ["FSLDIR"] = str(self.test_output_dir)
        
        cfg = Config()
        # Change a template path to use the environment variable
        cfg.brain_template = "${FSLDIR}/test_template.nii.gz"
        
        # Resolve the path
        cfg.resolve()
        
        # Check that the path was correctly resolved
        expected_path = f"{self.test_output_dir}/test_template.nii.gz"
        self.assertEqual(cfg.brain_template, expected_path)

    def test_config_load_save(self):
        """Test Config save and load functionality."""
        cfg = Config()
        cfg.n_jobs = 4
        cfg.log_level = "DEBUG"
        
        # Save the config
        cfg_path = self.test_output_dir / "test_config.json"
        cfg.save(cfg_path)
        
        # Check that the file exists
        self.assertTrue(cfg_path.exists())
        
        # Load the config
        loaded_cfg = Config.load(cfg_path)
        
        # Check that the loaded config has the same values
        self.assertEqual(loaded_cfg.n_jobs, 4)
        self.assertEqual(loaded_cfg.log_level, "DEBUG")


class TestLoggers(BrainStemXBaseTest):
    """Tests for logging functions in core.py."""

    def test_get_logger(self):
        """Test get_logger function."""
        log_file = self.test_output_dir / "test.log"
        logger = get_logger("test_logger", log_file, "INFO")
        
        # Check that the logger is configured correctly
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, logging.INFO)
        
        # Check that the log file exists
        self.assertTrue(log_file.exists())
        
        # Write to the log
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check that the message is in the log file
        with open(log_file, "r") as f:
            log_content = f.read()
        self.assertIn(test_message, log_content)


class TestUtilities(BrainStemXBaseTest):
    """Tests for utility functions in core.py."""

    def test_check_file_dependencies(self):
        """Test check_file_dependencies with existing and missing files."""
        # Create test files
        file1 = self.test_output_dir / "file1.txt"
        file2 = self.test_output_dir / "file2.txt"
        file1.touch()
        
        # Test with all files existing
        result = check_file_dependencies([file1], self.logger, "test")
        self.assertTrue(result)
        
        # Test with missing files
        with self.assertRaises(FileNotFoundError):
            check_file_dependencies([file1, file2], self.logger, "test")

    def test_mu_sigma(self):
        """Test _mu_sigma function for robust mean and standard deviation."""
        # Create test data with outliers
        data = np.array([1, 2, 3, 4, 5, 100])
        
        # Calculate robust mean and std
        mu, sigma = _mu_sigma(data)
        
        # Check that the results are reasonable
        # The median should be 3.5
        self.assertAlmostEqual(mu, 3.5)
        
        # The MAD should be close to 1.5 * 1.4826 = ~2.22
        self.assertGreater(sigma, 2.0)
        self.assertLess(sigma, 3.0)


class TestImageProcessing(BrainStemXBaseTest):
    """Tests for image processing functions in core.py."""

    def test_log_image_stats(self):
        """Test log_image_stats with a simple test image."""
        self.skip_if_no_ants()
        
        # Create a test image
        test_image = self.create_test_ants_image()
        
        # Get image stats
        stats = log_image_stats(test_image, "test_image", self.logger)
        
        # Check that the stats are correct
        self.assertEqual(stats['shape'], [10, 10, 10])
        self.assertEqual(stats['dtype'], "float32")
        self.assertAlmostEqual(stats['min'], 0.0)
        self.assertAlmostEqual(stats['max'], 1.0)
        self.assertAlmostEqual(stats['mean'], 0.027, places=3)  # 27/1000 voxels = 0.027

    def test_check_mask(self):
        """Test check_mask function."""
        self.skip_if_no_ants()
        
        # Create test masks
        empty_mask = self.create_test_ants_image(np.zeros((10, 10, 10), dtype=np.uint8))
        small_mask = self.create_test_ants_image(np.zeros((10, 10, 10), dtype=np.uint8))
        small_mask.numpy()[5, 5, 5] = 1  # Single voxel
        valid_mask = self.create_test_ants_image(self.binary_mask_3d)
        
        # Create a test audit dictionary
        audit = {"checks": {}}
        
        # Test empty mask
        check_mask(empty_mask, 5, self.logger, "empty_mask", audit)
        self.assertEqual(audit["checks"]["empty_mask"]["status"], "EMPTY")
        self.assertEqual(audit["checks"]["empty_mask"]["vox"], 0)
        
        # Test small mask
        check_mask(small_mask, 5, self.logger, "small_mask", audit)
        self.assertEqual(audit["checks"]["small_mask"]["status"], "SMALL")
        self.assertEqual(audit["checks"]["small_mask"]["vox"], 1)
        
        # Test valid mask
        check_mask(valid_mask, 5, self.logger, "valid_mask", audit)
        self.assertEqual(audit["checks"]["valid_mask"]["status"], "OK")
        self.assertEqual(audit["checks"]["valid_mask"]["vox"], 27)  # 3x3x3 cube


if __name__ == "__main__":
    unittest.main()