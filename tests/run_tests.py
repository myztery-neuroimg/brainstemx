#!/usr/bin/env python3
"""
BrainStemX Test Runner

This script discovers and runs all BrainStemX unit tests.
"""

import unittest
import sys
import argparse
import logging
from pathlib import Path


def setup_logging(verbose=False):
    """Set up logging for the test runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("test_runner")


def run_tests(pattern="test_*.py", test_dir=None, verbose=False):
    """Run all tests matching the pattern."""
    logger = setup_logging(verbose)
    
    # Determine test directory
    if test_dir is None:
        test_dir = Path(__file__).parent
    else:
        test_dir = Path(test_dir)
    
    logger.info("Discovering tests in %s with pattern %s", test_dir, pattern)
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern=pattern)
    
    # Count tests
    test_count = suite.countTestCases()
    logger.info("Found %d tests", test_count)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Report results
    logger.info(
        "Tests complete: %d passed, %d failed, %d errors, %d skipped",
        test_count - len(result.failures) - len(result.errors) - len(result.skipped),
        len(result.failures),
        len(result.errors),
        len(result.skipped)
    )
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run BrainStemX tests")
    parser.add_argument(
        "--pattern", "-p",
        default="test_*.py",
        help="Pattern to match test files (default: test_*.py)"
    )
    parser.add_argument(
        "--dir", "-d",
        default=None,
        help="Directory to search for tests (default: tests/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Increase verbosity"
    )
    parser.add_argument(
        "--core", 
        action="store_true",
        help="Run only core module tests"
    )
    parser.add_argument(
        "--pipeline", 
        action="store_true",
        help="Run only pipeline module tests"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Run only validation module tests"
    )
    parser.add_argument(
        "--postprocess", 
        action="store_true",
        help="Run only postprocessing module tests"
    )
    
    args = parser.parse_args()
    
    # Determine test pattern based on module flags
    pattern = args.pattern
    if args.core:
        pattern = "test_core*.py"
    elif args.pipeline:
        pattern = "test_pipeline.py"
    elif args.validate:
        pattern = "test_validate*.py"
    elif args.postprocess:
        pattern = "test_postprocess.py"
    
    # Run tests and return exit code
    return run_tests(pattern, args.dir, args.verbose)


if __name__ == "__main__":
    sys.exit(main())