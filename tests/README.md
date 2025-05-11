# BrainStemX Test Suite

This directory contains unit tests for the BrainStemX brain-stem MRI intensity-clustering pipeline. The tests are designed to validate program flow, interface compatibility, and error handling with minimal test data.

## Test Structure

The test suite is organized as follows:

- `base.py`: Base test case class with common utilities and fixtures
- `test_core.py`: Tests for basic utility functions in core.py
- `test_core_image_processing.py`: Tests for image processing functions in core.py
- `test_pipeline.py`: Tests for the main pipeline workflow
- `test_validate_inputs.py`: Tests for input validation
- `test_postprocess.py`: Tests for post-processing analysis
- `run_tests.py`: Test runner script

## Running Tests

### Running All Tests

To run all tests:

```bash
python -m tests.run_tests
```

Or use the executable directly:

```bash
./tests/run_tests.py
```

### Running Specific Test Modules

To run tests for specific modules:

```bash
# Run only core module tests
python -m tests.run_tests --core

# Run only pipeline tests
python -m tests.run_tests --pipeline

# Run only input validation tests
python -m tests.run_tests --validate

# Run only postprocessing tests
python -m tests.run_tests --postprocess
```

### Verbose Output

For more detailed test output:

```bash
python -m tests.run_tests --verbose
```

## Test Data

The tests are designed to work with minimal synthetic data by default. For more comprehensive testing:

1. Place real NIfTI test files in the `unit-test-data-local-only` directory in the project root
2. Tests that require real data will automatically use these files if available
3. Tests that require real data but don't find it will be skipped

The test data directory structure should be:

```
unit-test-data-local-only/
├── flair.nii.gz       # Raw FLAIR file
├── t1.nii.gz          # Raw T1 file
├── processed_subject/ # Directory with processed outputs
    ├── flair_to_t1.nii.gz
    ├── lesion_sd2.0.nii.gz
    ├── brainstem_mask.nii.gz
    └── ...
```

## Test Output

Test output files are stored in the `tests/temp-test-data-output` directory, which is automatically cleaned up after tests run.

## Adding New Tests

To add new tests:

1. Create a new test file following the naming convention `test_*.py`
2. Import the `BrainStemXBaseTest` class from `tests.base`
3. Create test classes that inherit from `BrainStemXBaseTest`
4. Add test methods with names starting with `test_`

Example:

```python
from tests.base import BrainStemXBaseTest

class TestMyFeature(BrainStemXBaseTest):
    def test_my_function(self):
        # Test code here
        self.assertEqual(1, 1)
```

## Dependencies

The tests require the same dependencies as the main BrainStemX package. Key testing dependencies include:

- unittest (standard library)
- numpy
- nibabel
- ants (optional, tests will be skipped if not available)