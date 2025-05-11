"""
Base test case for BrainStemX unit tests.

This module provides a base test case with common functionality:
- Path handling for test data input and output
- Setup and teardown methods
- Utility functions for creating test data
"""

import unittest
import os
import shutil
import tempfile
import logging
from pathlib import Path
import numpy as np

try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class BrainStemXBaseTest(unittest.TestCase):
    """Base test case for BrainStemX tests."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level test resources."""
        # Configure logging for tests
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        cls.logger = logging.getLogger(f"test.{cls.__name__}")
        
        # Define base paths
        cls.project_root = Path(__file__).parent.parent
        cls.test_data_dir = cls.project_root / "unit-test-data-local-only"
        
        # Skip tests that require test data if the directory doesn't exist
        cls.test_data_available = cls.test_data_dir.exists()
        if not cls.test_data_available:
            cls.logger.warning(f"Test data directory not found: {cls.test_data_dir}")

    def setUp(self):
        """Set up test case resources."""
        # Create a temporary directory for test outputs
        self.temp_output_dir = Path(self.project_root) / "tests" / "temp-test-data-output"
        self.temp_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create test case specific output directory
        self.test_output_dir = self.temp_output_dir / self._testMethodName
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Initialize minimal test data
        self.small_array_3d = np.zeros((10, 10, 10), dtype=np.float32)
        self.small_array_3d[4:7, 4:7, 4:7] = 1.0  # Simple cube in the center
        
        self.binary_mask_3d = np.zeros((10, 10, 10), dtype=np.uint8)
        self.binary_mask_3d[4:7, 4:7, 4:7] = 1    # Binary mask matching the cube

    def tearDown(self):
        """Clean up resources after test."""
        # Remove test case specific output directory
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def skip_if_no_test_data(self):
        """Skip the test if test data is not available."""
        if not self.test_data_available:
            self.skipTest("Test data directory not found")

    def skip_if_no_ants(self):
        """Skip the test if ANTs is not available."""
        if not ANTS_AVAILABLE:
            self.skipTest("ANTs is not available")
    
    def skip_if_no_nibabel(self):
        """Skip the test if nibabel is not available."""
        if not NIBABEL_AVAILABLE:
            self.skipTest("nibabel is not available")

    def create_test_nifti(self, array=None, affine=None, filename="test.nii.gz"):
        """Create a test NIfTI file.
        
        Args:
            array: Numpy array data (defaults to self.small_array_3d)
            affine: Affine transformation matrix (defaults to identity)
            filename: Name of the output file
            
        Returns:
            Path to the created NIfTI file
        """
        self.skip_if_no_nibabel()
        
        if array is None:
            array = self.small_array_3d
        
        if affine is None:
            affine = np.eye(4)
            
        output_path = self.test_output_dir / filename
        img = nib.Nifti1Image(array, affine)
        nib.save(img, output_path)
        
        return output_path
    
    def create_test_ants_image(self, array=None, spacing=(1.0, 1.0, 1.0)):
        """Create a test ANTs image.
        
        Args:
            array: Numpy array data (defaults to self.small_array_3d)
            spacing: Voxel spacing (defaults to 1mm isotropic)
            
        Returns:
            ANTs image object
        """
        self.skip_if_no_ants()
        
        if array is None:
            array = self.small_array_3d
            
        img = ants.from_numpy(
            array,
            origin=(0, 0, 0),
            spacing=spacing,
            direction=np.eye(3)
        )
        
        return img
    
    def get_test_data_path(self, filename):
        """Get the path to a test data file.
        
        Args:
            filename: Name of the test data file
            
        Returns:
            Path to the test data file
        """
        self.skip_if_no_test_data()
        
        path = self.test_data_dir / filename
        if not path.exists():
            self.skipTest(f"Test data file not found: {path}")
            
        return path