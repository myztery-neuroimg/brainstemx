"""
Basic tests that don't rely on external dependencies.

These tests verify that the test framework itself is working properly.
"""

import unittest
from pathlib import Path

from tests.base import BrainStemXBaseTest


class TestFramework(BrainStemXBaseTest):
    """Basic tests to verify the test framework."""

    def test_directory_setup(self):
        """Test that the test directories are set up correctly."""
        # Check that output directory exists
        self.assertTrue(self.temp_output_dir.exists())
        self.assertTrue(self.test_output_dir.exists())
        
        # Check that test arrays are created
        self.assertEqual(self.small_array_3d.shape, (10, 10, 10))
        self.assertEqual(self.binary_mask_3d.shape, (10, 10, 10))

    def test_file_io(self):
        """Test basic file I/O operations."""
        # Create a test file
        test_file = self.test_output_dir / "test_file.txt"
        test_content = "Hello, world!"
        
        # Write to the file
        test_file.write_text(test_content)
        
        # Check that the file exists
        self.assertTrue(test_file.exists())
        
        # Read from the file
        read_content = test_file.read_text()
        
        # Check that the content matches
        self.assertEqual(read_content, test_content)

    def test_project_structure(self):
        """Test that the project structure is correct."""
        # Project root should exist
        self.assertTrue(self.project_root.exists())
        
        # Source directory should exist
        src_dir = self.project_root / "src"
        self.assertTrue(src_dir.exists())
        
        # Core module should exist
        core_py = src_dir / "core.py"
        self.assertTrue(core_py.exists())


if __name__ == "__main__":
    unittest.main()