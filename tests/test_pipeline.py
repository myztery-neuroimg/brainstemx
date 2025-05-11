"""
Unit tests for the pipeline module.

These tests focus on the functional flow and integration points
of the pipeline module, which coordinates the overall processing workflow.
"""

import unittest
import logging
import tempfile
from pathlib import Path
import numpy as np
import json
from unittest.mock import patch, MagicMock

from tests.base import BrainStemXBaseTest
from src.core import Config
from src.pipeline import process_subject, process_batch


class TestPipeline(BrainStemXBaseTest):
    """Tests for the pipeline module."""

    def setUp(self):
        """Set up test case with mock data."""
        super().setUp()
        
        # Create test directories
        self.subj_id = "test_subject"
        self.out_dir = self.test_output_dir / self.subj_id
        self.out_dir.mkdir(exist_ok=True)
        
        # Create test files
        self.flair_path = self.test_output_dir / "flair.nii.gz"
        self.t1_path = self.test_output_dir / "t1.nii.gz"
        self.dwi_path = self.test_output_dir / "dwi.nii.gz"
        self.swi_path = self.test_output_dir / "swi.nii.gz"
        
        # Touch test files to create them
        self.flair_path.touch()
        self.t1_path.touch()
        self.dwi_path.touch()
        self.swi_path.touch()
        
        # Create a minimal config
        self.cfg = Config()
        self.cfg.n_jobs = 1
        
        # Set up mock paths for validating inputs
        # For real tests, we would use the existing generate_synthetic_data.py to create test data
        flair_json = self.test_output_dir / "flair.json"
        with open(flair_json, "w") as f:
            json.dump({
                "RepetitionTime": 8.0,
                "EchoTime": 0.1,
                "MagneticFieldStrength": 3.0,
                "PixelSpacing": [1.0, 1.0],
                "SliceThickness": 1.0
            }, f)
        
        t1_json = self.test_output_dir / "t1.json"
        with open(t1_json, "w") as f:
            json.dump({
                "RepetitionTime": 2.0,
                "EchoTime": 0.003,
                "MagneticFieldStrength": 3.0,
                "PixelSpacing": [1.0, 1.0],
                "SliceThickness": 1.0
            }, f)

    @patch('src.pipeline.validate_inputs')
    @patch('src.pipeline.ants.image_read')
    @patch('src.pipeline.preprocess_flair')
    @patch('src.pipeline.skullstrip')
    def test_process_subject_flow(self, mock_skullstrip, mock_preprocess_flair, 
                                 mock_image_read, mock_validate_inputs):
        """Test the process_subject function flow with mocked dependencies."""
        # Set up mock returns
        mock_image = MagicMock()
        mock_image.numpy.return_value = np.zeros((10, 10, 10))
        mock_image.shape = (10, 10, 10)
        mock_image_read.return_value = mock_image
        
        mock_preprocess_flair.return_value = mock_image
        
        brain_mask = MagicMock()
        brain_mask.numpy.return_value = np.ones((10, 10, 10))
        brain_mask.shape = (10, 10, 10)
        mock_skullstrip.return_value = (mock_image, brain_mask)
        
        # Test the process_subject function
        with patch('src.pipeline.register_to_t1') as mock_register:
            mock_register.return_value = (mock_image, ["fwd"], ["inv"])
            
            with patch('src.pipeline.brainstem_masks') as mock_bs_masks:
                bs_mask = MagicMock()
                bs_mask.numpy.return_value = np.ones((10, 10, 10))
                bs_mask.shape = (10, 10, 10)
                mock_bs_masks.return_value = (bs_mask, bs_mask, bs_mask)
                
                with patch('src.pipeline.wm_segmentation') as mock_wm_seg:
                    wm_mask = MagicMock()
                    wm_mask.numpy.return_value = np.ones((10, 10, 10))
                    wm_mask.shape = (10, 10, 10)
                    mock_wm_seg.return_value = (wm_mask, None)
                    
                    with patch('src.pipeline.lesion_masks') as mock_lesion_masks:
                        lesion = MagicMock()
                        lesion.numpy.return_value = np.zeros((10, 10, 10))
                        lesion.shape = (10, 10, 10)
                        mock_lesion_masks.return_value = {
                            1.5: lesion,
                            2.0: lesion,
                            2.5: lesion,
                            3.0: lesion
                        }
                        
                        with patch('src.pipeline.cluster_metrics') as mock_metrics:
                            mock_metrics.return_value = self.out_dir / "clusters.csv"
                            
                            with patch('src.pipeline.qc_overlay') as mock_qc:
                                # Try running the process_subject function
                                try:
                                    process_subject(
                                        self.subj_id,
                                        self.flair_path,
                                        self.t1_path,
                                        self.out_dir,
                                        self.cfg,
                                        self.dwi_path,
                                        self.swi_path
                                    )
                                    
                                    # Check that all the expected steps were called
                                    mock_validate_inputs.assert_called_once()
                                    mock_image_read.assert_called()
                                    mock_preprocess_flair.assert_called_once()
                                    mock_skullstrip.assert_called_once()
                                    mock_register.assert_called()
                                    mock_bs_masks.assert_called_once()
                                    mock_wm_seg.assert_called_once()
                                    mock_lesion_masks.assert_called_once()
                                    mock_metrics.assert_called_once()
                                    mock_qc.assert_called_once()
                                    
                                    # Check that the output JSON was created
                                    output_json = self.out_dir / "outputs.json"
                                    self.assertTrue(output_json.exists())
                                    
                                except Exception as e:
                                    self.fail(f"process_subject raised an unexpected exception: {e}")

    @patch('src.pipeline.process_subject')
    def test_process_batch(self, mock_process_subject):
        """Test process_batch function."""
        # Create a test batch file
        batch_file = self.test_output_dir / "batch.tsv"
        with open(batch_file, "w") as f:
            f.write(f"subj1\t{self.flair_path}\t{self.t1_path}\n")
            f.write(f"subj2\t{self.flair_path}\t{self.t1_path}\n")
        
        # Test the process_batch function
        process_batch(batch_file, self.cfg)
        
        # Check that process_subject was called twice
        self.assertEqual(mock_process_subject.call_count, 2)


class TestPipelineWithTestData(BrainStemXBaseTest):
    """Tests for pipeline with more realistic test data.
    
    These tests use the test data directory and are skipped if it's not available.
    """
    
    @unittest.skip("Requires real test data - enable when test data is available")
    def test_process_subject_with_test_data(self):
        """Test process_subject with real test data."""
        self.skip_if_no_test_data()
        self.skip_if_no_ants()
        
        # Get paths to test data
        flair_path = self.get_test_data_path("flair.nii.gz")
        t1_path = self.get_test_data_path("t1.nii.gz")
        
        # Create output directory
        out_dir = self.test_output_dir
        
        # Create config
        cfg = Config()
        cfg.n_jobs = 1
        
        # Run process_subject
        try:
            process_subject("test_subject", flair_path, t1_path, out_dir, cfg)
            
            # Check for expected outputs
            expected_files = [
                "outputs.json",
                "flair_pp.nii.gz",
                "t1_brain.nii.gz",
                "brain_mask.nii.gz",
                "flair_to_t1.nii.gz",
                "brainstem_mask.nii.gz",
                "wm_mask.nii.gz",
                "lesion_sd2.0.nii.gz",
                "QC_overlay.png",
                "lesion_clusters.csv"
            ]
            
            for filename in expected_files:
                file_path = out_dir / filename
                self.assertTrue(file_path.exists(), f"Expected file not found: {filename}")
                
        except Exception as e:
            self.fail(f"process_subject with test data raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()