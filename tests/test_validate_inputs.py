"""
Unit tests for the validate_inputs module.

These tests focus on the input validation functionality
that verifies NIfTI files and their JSON side-car data.
"""

import unittest
import json
import nibabel as nib
import numpy as np
from pathlib import Path

from tests.base import BrainStemXBaseTest
from src.validate_inputs import (
    read_json, check_range, validate_pair, validate_inputs
)


class TestValidateInputs(BrainStemXBaseTest):
    """Tests for the validate_inputs module."""

    def setUp(self):
        """Set up test case with mock input files."""
        super().setUp()
        
        # Create test data directory
        self.test_data_dir = self.test_output_dir / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create minimal NIfTI files for testing
        self.flair_path = self.test_data_dir / "flair.nii.gz"
        self.t1_path = self.test_data_dir / "t1.nii.gz"
        self.dwi_path = self.test_data_dir / "dwi.nii.gz"
        self.swi_path = self.test_data_dir / "swi.nii.gz"
        
        # Create minimal arrays
        array_3d = np.zeros((10, 10, 10), dtype=np.float32)
        
        # Save as NIfTI
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm voxel size
        
        for path in [self.flair_path, self.t1_path, self.dwi_path, self.swi_path]:
            img = nib.Nifti1Image(array_3d, affine)
            nib.save(img, path)
            
        # Create valid JSON side-car files
        flair_json = {
            "RepetitionTime": 8.0,
            "EchoTime": 0.1,
            "MagneticFieldStrength": 3.0,
            "PixelSpacing": [1.0, 1.0],
            "SliceThickness": 1.0
        }
        
        t1_json = {
            "RepetitionTime": 2.0,
            "EchoTime": 0.003,
            "MagneticFieldStrength": 3.0,
            "PixelSpacing": [1.0, 1.0],
            "SliceThickness": 1.0
        }
        
        dwi_json = {
            "RepetitionTime": 5.0,
            "EchoTime": 0.05,
            "MagneticFieldStrength": 3.0,
            "PixelSpacing": [1.0, 1.0],
            "SliceThickness": 1.0
        }
        
        swi_json = {
            "RepetitionTime": 3.0,
            "EchoTime": 0.02,
            "MagneticFieldStrength": 3.0,
            "PixelSpacing": [1.0, 1.0],
            "SliceThickness": 1.0
        }
        
        # Write JSON files
        with open(self.flair_path.with_suffix(".json"), "w") as f:
            json.dump(flair_json, f)
            
        with open(self.t1_path.with_suffix(".json"), "w") as f:
            json.dump(t1_json, f)
            
        with open(self.dwi_path.with_suffix(".json"), "w") as f:
            json.dump(dwi_json, f)
            
        with open(self.swi_path.with_suffix(".json"), "w") as f:
            json.dump(swi_json, f)

    def test_read_json(self):
        """Test read_json function."""
        json_path = self.test_data_dir / "test.json"
        
        # Create a test JSON file
        test_data = {"key": "value", "number": 42}
        with open(json_path, "w") as f:
            json.dump(test_data, f)
        
        # Test valid JSON
        data = read_json(json_path)
        self.assertEqual(data, test_data)
        
        # Test invalid JSON
        invalid_path = self.test_data_dir / "invalid.json"
        with open(invalid_path, "w") as f:
            f.write("{invalid json")
        
        with self.assertRaises(RuntimeError):
            read_json(invalid_path)

    def test_check_range(self):
        """Test check_range function."""
        # Value in range
        self.assertTrue(check_range(5, 0, 10))
        
        # Value at lower bound
        self.assertTrue(check_range(0, 0, 10))
        
        # Value at upper bound
        self.assertTrue(check_range(10, 0, 10))
        
        # Value below range
        self.assertFalse(check_range(-1, 0, 10))
        
        # Value above range
        self.assertFalse(check_range(11, 0, 10))

    def test_validate_pair(self):
        """Test validate_pair function."""
        # Test valid pair
        report = validate_pair(self.flair_path, self.flair_path.with_suffix(".json"), "FLAIR", self.logger)
        self.assertEqual(report["status"], "PASS")
        
        # Test missing JSON
        non_existent = self.test_data_dir / "non_existent.json"
        report = validate_pair(self.flair_path, non_existent, "FLAIR", self.logger)
        self.assertEqual(report["warning"], "no_sidecar")
        
        # Test FLAIR with invalid TE
        bad_flair_json = self.test_data_dir / "bad_flair.json"
        with open(bad_flair_json, "w") as f:
            json.dump({
                "RepetitionTime": 8.0,
                "EchoTime": 0.01,  # Too short for FLAIR
                "MagneticFieldStrength": 3.0,
                "PixelSpacing": [1.0, 1.0],
                "SliceThickness": 1.0
            }, f)
            
        report = validate_pair(self.flair_path, bad_flair_json, "FLAIR", self.logger)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("EchoTime" in v for v in report.get("violations", [])))
        
        # Test mismatched voxel sizes
        mismatched_json = self.test_data_dir / "mismatched.json"
        with open(mismatched_json, "w") as f:
            json.dump({
                "RepetitionTime": 8.0,
                "EchoTime": 0.1,
                "MagneticFieldStrength": 3.0,
                "PixelSpacing": [2.0, 2.0],  # Different from NIfTI header
                "SliceThickness": 2.0         # Different from NIfTI header
            }, f)
            
        report = validate_pair(self.flair_path, mismatched_json, "FLAIR", self.logger)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("error", report)
        self.assertIn("Voxel size mismatch", report["error"])

    def test_validate_inputs(self):
        """Test validate_inputs function."""
        out_dir = self.test_output_dir / "validation_output"
        out_dir.mkdir(exist_ok=True)
        
        # Test with valid inputs
        validate_inputs(out_dir, self.flair_path, self.t1_path, self.logger, self.dwi_path, self.swi_path)
        
        # Check that the validation report was created
        report_path = out_dir / "inputs_valid.json"
        self.assertTrue(report_path.exists())
        
        # Check the report content
        with open(report_path, "r") as f:
            report = json.load(f)
            
        # Should have 4 entries (FLAIR, T1, DWI, SWI)
        self.assertEqual(len(report), 4)
        
        # All should have passed
        for entry in report:
            self.assertEqual(entry["status"], "PASS")
            
        # Test with invalid FLAIR
        bad_flair_json = self.flair_path.with_suffix(".json")
        with open(bad_flair_json, "w") as f:
            json.dump({
                # Missing RepetitionTime - should cause failure
                "EchoTime": 0.1,
                "MagneticFieldStrength": 3.0,
                "PixelSpacing": [1.0, 1.0],
                "SliceThickness": 1.0
            }, f)
            
        with self.assertRaises(RuntimeError):
            validate_inputs(out_dir, self.flair_path, self.t1_path, self.logger)


if __name__ == "__main__":
    unittest.main()