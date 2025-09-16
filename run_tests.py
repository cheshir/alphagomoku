#!/usr/bin/env python3
"""Test runner script for AlphaGomoku with different test configurations."""

import sys
import subprocess
import argparse
import os


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def run_unit_tests(verbose=False, markers=None):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    if verbose:
        cmd.append("-v")
    if markers:
        cmd.extend(["-m", markers])
    cmd.extend(["--tb=short"])

    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False, markers=None):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    if verbose:
        cmd.append("-v")
    if markers:
        cmd.extend(["-m", markers])
    cmd.extend(["--tb=short"])

    return run_command(cmd, "Integration Tests")


def run_performance_tests(verbose=False):
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/", "-v", "--tb=short"]
    if verbose:
        cmd.append("-s")  # Don't capture output for performance tests

    return run_command(cmd, "Performance Tests")


def run_specific_component(component, verbose=False):
    """Run tests for a specific component."""
    component_map = {
        'env': 'test_env.py',
        'model': 'test_model.py',
        'mcts': 'test_mcts.py',
        'tss': 'test_tss',  # Multiple TSS test files
        'trainer': 'test_trainer.py',
        'buffer': 'test_data_buffer.py',
        'parallel': 'test_parallel_selfplay.py',
        'evaluator': 'test_evaluator.py',
        'errors': 'test_error_handling.py'
    }

    if component not in component_map:
        print(f"❌ Unknown component: {component}")
        print(f"Available components: {list(component_map.keys())}")
        return False

    test_pattern = component_map[component]
    cmd = ["python", "-m", "pytest", "-k", test_pattern]

    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short"])

    return run_command(cmd, f"{component.title()} Component Tests")


def run_fast_tests(verbose=False):
    """Run fast tests (exclude slow markers)."""
    cmd = ["python", "-m", "pytest", "-m", "not slow", "--tb=short"]
    if verbose:
        cmd.append("-v")

    return run_command(cmd, "Fast Tests")


def run_all_tests(verbose=False, skip_performance=False):
    """Run all tests."""
    success = True

    # Unit tests
    if not run_unit_tests(verbose):
        success = False

    # Integration tests
    if not run_integration_tests(verbose):
        success = False

    # Performance tests (optional)
    if not skip_performance:
        if not run_performance_tests(verbose):
            success = False
    else:
        print("\n⏩ Skipping performance tests")

    return success


def run_coverage_tests():
    """Run tests with coverage reporting."""
    if not check_coverage_available():
        print("❌ pytest-cov not available. Install with: pip install pytest-cov")
        return False

    cmd = [
        "python", "-m", "pytest",
        "--cov=alphagomoku",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "tests/unit/", "tests/integration/"
    ]

    return run_command(cmd, "Coverage Tests")


def check_dependencies():
    """Check if required testing dependencies are available."""
    required = ['pytest', 'numpy', 'torch']
    optional = ['pytest-cov', 'pytest-xdist', 'pytest-timeout']

    missing_required = []
    missing_optional = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)

    for package in optional:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)

    if missing_required:
        print(f"❌ Missing required packages: {missing_required}")
        return False

    if missing_optional:
        print(f"⚠️  Missing optional packages: {missing_optional}")
        print("Consider installing for enhanced testing features:")
        print(f"pip install {' '.join(missing_optional)}")

    print("✅ All required dependencies available")
    return True


def check_coverage_available():
    """Check if coverage is available."""
    try:
        import pytest_cov
        return True
    except ImportError:
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="AlphaGomoku Test Runner")

    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow)")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--component", type=str, help="Run tests for specific component")
    parser.add_argument("--markers", type=str, help="Run tests with specific pytest markers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests in full run")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps or len(sys.argv) == 1:
        if not check_dependencies():
            sys.exit(1)
        if args.check_deps:
            return

    # Run specific test categories
    success = True

    if args.unit:
        success = run_unit_tests(args.verbose, args.markers)
    elif args.integration:
        success = run_integration_tests(args.verbose, args.markers)
    elif args.performance:
        success = run_performance_tests(args.verbose)
    elif args.fast:
        success = run_fast_tests(args.verbose)
    elif args.coverage:
        success = run_coverage_tests()
    elif args.component:
        success = run_specific_component(args.component, args.verbose)
    else:
        # Run all tests
        success = run_all_tests(args.verbose, args.skip_performance)

    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check output above.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()