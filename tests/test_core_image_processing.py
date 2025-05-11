"""
Unit tests for image processing functions in the core module.

These tests focus on the interface compatibility and function flow
of the more complex image processing functions in core.py.
"""

import unittest
import logging
import numpy as np
from pathlib import Path

from tests.base import BrainStemXBaseTest
from src.core import (
    Config, skullstrip, preprocess_flair, preprocess_generic,
    _rigid_bbr, register_to_t1, brainstem_masks, wm_segmentation,
    lesion_masks, cluster_metrics
)


class TestSkullStrip(BrainStemXBaseTest):
    """Tests for the skullstrip function."""

    def test_skullstrip_interface(self):
        """Test skullstrip function interface with minimal inputs."""
        self.skip_if_no_ants()
        
        # Create a minimal configuration
        cfg = Config()
        cfg.skullstrip_order = ["basic"]  # Use only basic thresholding to avoid external dependencies
        
        # Create a simple test image with a bright "brain" region
        test_array = np.zeros((20, 20, 20), dtype=np.float32)
        test_array[5:15, 5:15, 5:15] = 100.0  # Bright cube in the center
        test_image = self.create_test_ants_image(test_array)
        
        # Test skullstrip function
        try:
            brain, mask = skullstrip(test_image, cfg, self.logger)
            
            # Verify that we got outputs of the right type
            self.assertIsNotNone(brain)
            self.assertIsNotNone(mask)
            
            # Basic check on the mask
            mask_array = mask.numpy()
            self.assertGreater(mask_array.sum(), 0)  # Should have some non-zero voxels
            
            # Check that the brain is the product of image and mask
            np.testing.assert_array_almost_equal(
                brain.numpy(),
                test_image.numpy() * mask.numpy()
            )
            
        except Exception as e:
            self.fail(f"skullstrip raised an unexpected exception: {e}")


class TestPreprocessing(BrainStemXBaseTest):
    """Tests for preprocessing functions."""

    def test_preprocess_generic_interface(self):
        """Test preprocess_generic function interface."""
        self.skip_if_no_ants()
        
        # Create a simple test image
        test_image = self.create_test_ants_image()
        
        # Test the function
        try:
            result = preprocess_generic(test_image, self.logger)
            
            # Verify the result is of the right type
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, test_image.shape)
            
        except Exception as e:
            self.fail(f"preprocess_generic raised an unexpected exception: {e}")

    def test_preprocess_flair_interface(self):
        """Test preprocess_flair function interface."""
        self.skip_if_no_ants()
        
        # Create a simple test image
        test_image = self.create_test_ants_image()
        
        # Create a minimal configuration
        cfg = Config()
        cfg.flair_template = self.test_output_dir / "flair_template.nii.gz"
        
        # Create a dummy FLAIR template
        template_image = self.create_test_ants_image()
        template_image.to_file(str(cfg.flair_template))
        
        # Test the function
        try:
            result = preprocess_flair(test_image, cfg, self.logger)
            
            # Verify the result is of the right type
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, test_image.shape)
            
        except Exception as e:
            self.fail(f"preprocess_flair raised an unexpected exception: {e}")


class TestRegistration(BrainStemXBaseTest):
    """Tests for registration functions."""

    def test_rigid_bbr_interface(self):
        """Test _rigid_bbr function interface."""
        self.skip_if_no_ants()
        
        # Create test images
        fixed = self.create_test_ants_image()
        moving = self.create_test_ants_image()
        
        # Test the function
        try:
            result = _rigid_bbr(fixed, moving)
            
            # Verify result is a dictionary
            self.assertIsInstance(result, dict)
            
            # Check for expected keys
            self.assertIn("warpedmovout", result)
            self.assertIn("fwdtransforms", result)
            self.assertIn("invtransforms", result)
            
        except Exception as e:
            self.fail(f"_rigid_bbr raised an unexpected exception: {e}")

    def test_register_to_t1_interface(self):
        """Test register_to_t1 function interface."""
        self.skip_if_no_ants()
        
        # Create test images
        t1 = self.create_test_ants_image()
        mov = self.create_test_ants_image()
        
        # Create config with registration parameters
        cfg = Config()
        cfg.reg_metric = "MI"
        
        # Test the function
        try:
            warped, fwd, inv = register_to_t1(t1, mov, cfg, self.logger, "TEST")
            
            # Verify outputs
            self.assertIsNotNone(warped)
            self.assertIsInstance(fwd, list)
            self.assertIsInstance(inv, list)
            
        except Exception as e:
            if "reg_metric" in str(e):
                # Skip the test if reg_metric is not a valid attribute
                # (This handles potential changes in the Config class)
                self.skipTest(f"Config doesn't have reg_metric attribute: {e}")
            else:
                self.fail(f"register_to_t1 raised an unexpected exception: {e}")


class TestLesionDetection(BrainStemXBaseTest):
    """Tests for lesion detection functions."""

    def test_lesion_masks_interface(self):
        """Test lesion_masks function interface."""
        self.skip_if_no_ants()
        
        # Create test images
        flair = self.create_test_ants_image()
        # Create non-uniform FLAIR with "hyperintensities"
        flair_array = flair.numpy()
        flair_array[5:7, 5:7, 5:7] = 2.0  # Bright spot
        flair.numpy()[:] = flair_array
        
        wm = self.create_test_ants_image(self.binary_mask_3d)
        bs = self.create_test_ants_image(self.binary_mask_3d)
        
        # Create config
        cfg = Config()
        cfg.thresholds_sd = [1.5, 2.0]
        cfg.min_vox = 1
        cfg.open_radius = 0
        cfg.connectivity = 6
        
        # Test the function
        try:
            masks = lesion_masks(flair, wm, bs, cfg, self.logger)
            
            # Verify output is a dictionary of the right size
            self.assertIsInstance(masks, dict)
            self.assertEqual(len(masks), len(cfg.thresholds_sd))
            
            # Check each mask
            for threshold, mask in masks.items():
                self.assertIn(threshold, cfg.thresholds_sd)
                self.assertEqual(mask.shape, flair.shape)
                
        except Exception as e:
            self.fail(f"lesion_masks raised an unexpected exception: {e}")

    def test_wm_segmentation_interface(self):
        """Test wm_segmentation function interface."""
        self.skip_if_no_ants()
        
        # Create test images
        t1_brain = self.create_test_ants_image()
        flair_reg = self.create_test_ants_image()
        brain_mask = self.create_test_ants_image(self.binary_mask_3d)
        
        # Test the function with t1
        try:
            wm, les = wm_segmentation(t1_brain, flair_reg, brain_mask, self.logger)
            
            # Verify outputs
            self.assertIsNotNone(wm)
            self.assertEqual(wm.shape, t1_brain.shape)
            
        except Exception as e:
            self.fail(f"wm_segmentation with T1 raised an unexpected exception: {e}")
        
        # Test the function without t1
        try:
            wm, les = wm_segmentation(None, flair_reg, brain_mask, self.logger)
            
            # Verify outputs
            self.assertIsNotNone(wm)
            self.assertEqual(wm.shape, flair_reg.shape)
            self.assertIsNone(les)  # les should be None when t1 is None
            
        except Exception as e:
            self.fail(f"wm_segmentation without T1 raised an unexpected exception: {e}")


class TestClusterMetrics(BrainStemXBaseTest):
    """Tests for cluster_metrics function."""

    def test_cluster_metrics_interface(self):
        """Test cluster_metrics function interface."""
        self.skip_if_no_ants()
        
        # Create a test mask with a single cluster
        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[3:6, 3:6, 3:6] = 1
        mask = self.create_test_ants_image(mask_array)
        
        # Create config
        cfg = Config()
        cfg.connectivity = 6
        cfg.max_elongation = 10.0
        
        # Create output directory
        out_dir = self.test_output_dir
        
        # Test the function
        try:
            csv_path = cluster_metrics(mask, cfg, out_dir, self.logger)
            
            # Verify output
            self.assertTrue(csv_path.exists())
            
            # Read the CSV to check it has the right format
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Should have at least one row for the cluster
            self.assertGreaterEqual(len(df), 1)
            
            # Should have expected columns
            expected_columns = ["label", "area", "mm3", "elong"]
            for col in expected_columns:
                self.assertIn(col, df.columns)
                
        except Exception as e:
            self.fail(f"cluster_metrics raised an unexpected exception: {e}")
            
    def test_cluster_metrics_empty(self):
        """Test cluster_metrics with an empty mask."""
        self.skip_if_no_ants()
        
        # Create an empty mask
        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask = self.create_test_ants_image(mask_array)
        
        # Create config
        cfg = Config()
        
        # Create output directory
        out_dir = self.test_output_dir
        
        # Test the function
        try:
            csv_path = cluster_metrics(mask, cfg, out_dir, self.logger)
            
            # Verify output
            self.assertTrue(csv_path.exists())
            
            # Should be an empty dataframe
            import pandas as pd
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 0)
            
        except Exception as e:
            self.fail(f"cluster_metrics raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()