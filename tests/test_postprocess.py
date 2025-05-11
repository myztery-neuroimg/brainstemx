"""
Unit tests for the postprocess module.

These tests focus on the postprocessing functionality
that performs quantitative analysis of lesion clusters.
"""

import unittest
import json
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from tests.base import BrainStemXBaseTest
from src.postprocess import analyse, _log_img_stats


class TestPostprocess(BrainStemXBaseTest):
    """Tests for the postprocess module."""

    def setUp(self):
        """Set up test case with mock output files for postprocessing."""
        super().setUp()
        
        # Create subject directory
        self.subj_dir = self.test_output_dir / "subject"
        self.subj_dir.mkdir(exist_ok=True)
        
        # Create the minimal required files for analyse function
        array_3d = np.zeros((20, 20, 20), dtype=np.float32)
        
        # Create a "lesion" in the FLAIR image
        flair_array = array_3d.copy()
        flair_array[5:10, 5:10, 5:10] = 100.0  # Bright cube
        
        # Create a mask with one lesion
        mask_array = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_array[5:10, 5:10, 5:10] = 1  # Binary mask of the "lesion"
        
        # Create a brainstem mask that contains the lesion
        bs_array = np.zeros((20, 20, 20), dtype=np.uint8)
        bs_array[4:11, 4:11, 4:11] = 1  # Slightly larger than the lesion
        
        # Save as NIfTI files
        affine = np.eye(4)
        
        # Save FLAIR
        flair_img = nib.Nifti1Image(flair_array, affine)
        flair_path = self.subj_dir / "flair_to_t1.nii.gz"
        nib.save(flair_img, flair_path)
        
        # Save primary lesion mask
        mask_img = nib.Nifti1Image(mask_array, affine)
        mask_path = self.subj_dir / "lesion_sd2.0.nii.gz"
        nib.save(mask_img, mask_path)
        
        # Save brainstem mask
        bs_img = nib.Nifti1Image(bs_array, affine)
        bs_path = self.subj_dir / "brainstem_mask.nii.gz"
        nib.save(bs_img, bs_path)
        
        # Optionally, create additional modalities for testing
        # T1 (with hypointensity in the lesion)
        t1_array = np.ones((20, 20, 20), dtype=np.float32) * 50
        t1_array[5:10, 5:10, 5:10] = 20.0  # Dark cube (hypointense)
        t1_img = nib.Nifti1Image(t1_array, affine)
        t1_path = self.subj_dir / "t1_brain.nii.gz"
        nib.save(t1_img, t1_path)
        
        # DWI (with hyperintensity in the lesion)
        dwi_array = np.ones((20, 20, 20), dtype=np.float32) * 30
        dwi_array[5:10, 5:10, 5:10] = 80.0  # Bright cube
        dwi_img = nib.Nifti1Image(dwi_array, affine)
        dwi_path = self.subj_dir / "dwi_to_t1.nii.gz"
        nib.save(dwi_img, dwi_path)
        
        # SWI (with hypointensity in the lesion)
        swi_array = np.ones((20, 20, 20), dtype=np.float32) * 40
        swi_array[5:10, 5:10, 5:10] = 10.0  # Dark cube
        swi_img = nib.Nifti1Image(swi_array, affine)
        swi_path = self.subj_dir / "swi_to_t1.nii.gz"
        nib.save(swi_img, swi_path)

    def test_log_img_stats(self):
        """Test _log_img_stats function."""
        self.skip_if_no_nibabel()
        
        # Load image
        img = nib.load(self.subj_dir / "flair_to_t1.nii.gz")
        
        # Call function
        _log_img_stats(img, "FLAIR", self.logger)
        
        # Not much to verify here as it just logs stats
        # We're mainly testing that it doesn't raise exceptions

    def test_analyse_basic(self):
        """Test analyse function with minimal required inputs."""
        self.skip_if_no_nibabel()
        
        # Run analyse
        csv_path = analyse(self.subj_dir)
        
        # Check that the CSV file was created
        self.assertTrue(csv_path.exists())
        
        # Check that the overlap.json file was created
        overlap_path = self.subj_dir / "overlap.json"
        self.assertTrue(overlap_path.exists())
        
        # Check the CSV content
        df = pd.read_csv(csv_path)
        
        # Should have one row for the lesion
        self.assertEqual(len(df), 1)
        
        # Should have basic columns
        expected_columns = ["cluster", "voxels", "mm3", "flair_mean", "flair_p90"]
        for col in expected_columns:
            self.assertIn(col, df.columns)
            
        # Should have correct voxel count
        self.assertEqual(df.iloc[0]["voxels"], 125)  # 5x5x5 cube
        
        # FLAIR mean should be about 100
        self.assertAlmostEqual(df.iloc[0]["flair_mean"], 100.0, places=0)
        
        # Check the overlap.json content
        with open(overlap_path, "r") as f:
            overlap = json.load(f)
            
        # Should have an entry for primary_sd
        self.assertIn("2.0", overlap)
        
        # Should have overlap metrics
        if "T1_hypo_vox" in overlap["2.0"]:
            self.assertEqual(overlap["2.0"]["T1_hypo_vox"], 125)
        
        if "SWI_bloom_vox" in overlap["2.0"]:
            self.assertEqual(overlap["2.0"]["SWI_bloom_vox"], 125)
            
        if "DWI_high_vox" in overlap["2.0"]:
            self.assertEqual(overlap["2.0"]["DWI_high_vox"], 125)

    def test_analyse_missing_files(self):
        """Test analyse with missing files."""
        self.skip_if_no_nibabel()
        
        # Create a fresh subject directory without all required files
        empty_dir = self.test_output_dir / "empty_subject"
        empty_dir.mkdir(exist_ok=True)
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            analyse(empty_dir)
            
    def test_analyse_without_optional_modalities(self):
        """Test analyse function without optional modalities."""
        self.skip_if_no_nibabel()
        
        # Create a minimal subject directory with only required files
        min_dir = self.test_output_dir / "min_subject"
        min_dir.mkdir(exist_ok=True)
        
        # Copy just the required files
        for fname in ["flair_to_t1.nii.gz", "lesion_sd2.0.nii.gz", "brainstem_mask.nii.gz"]:
            src = self.subj_dir / fname
            dst = min_dir / fname
            dst.write_bytes(src.read_bytes())
        
        # Run analyse
        csv_path = analyse(min_dir)
        
        # Check that the CSV file was created
        self.assertTrue(csv_path.exists())
        
        # Check the CSV content
        df = pd.read_csv(csv_path)
        
        # Should have one row for the lesion
        self.assertEqual(len(df), 1)
        
        # Should have only basic columns, not T1, DWI, or SWI metrics
        self.assertIn("flair_mean", df.columns)
        self.assertNotIn("t1_mean", df.columns)
        self.assertNotIn("dwi_mean", df.columns)
        self.assertNotIn("swi_min", df.columns)


class TestPostprocessWithRealData(BrainStemXBaseTest):
    """Tests for postprocess with actual NIfTI files.
    
    These tests are skipped if test data is not available.
    """
    
    @unittest.skip("Requires real test data - enable when test data is available")
    def test_analyse_with_real_data(self):
        """Test analyse with real NIfTI files."""
        self.skip_if_no_test_data()
        self.skip_if_no_nibabel()
        
        # Paths to real processed data
        test_subj_dir = self.test_data_dir / "processed_subject"
        
        if not test_subj_dir.exists():
            self.skipTest(f"Processed subject directory not found: {test_subj_dir}")
            
        # Required files for analyse
        required_files = [
            test_subj_dir / "flair_to_t1.nii.gz",
            test_subj_dir / "lesion_sd2.0.nii.gz",
            test_subj_dir / "brainstem_mask.nii.gz"
        ]
        
        if not all(f.exists() for f in required_files):
            self.skipTest("Required files not found in processed subject directory")
            
        # Run analyse
        csv_path = analyse(test_subj_dir)
        
        # Check that the CSV file was created
        self.assertTrue(csv_path.exists())
        
        # Check that the overlap.json file was created
        overlap_path = test_subj_dir / "overlap.json"
        self.assertTrue(overlap_path.exists())


if __name__ == "__main__":
    unittest.main()