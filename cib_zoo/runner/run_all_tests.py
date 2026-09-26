"""Global Test Suite Runner for OASIS CIB Modular Engine.

Executes all hermetic, architectural, and systemic tests with 4-thread CPU enforcement,
suitable for local verification, CI workflows, and FEP cluster Apptainer environments.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import pytest

# Ensure oasis root is on sys.path
_oasis_root = Path(__file__).resolve().parents[2]
if str(_oasis_root) not in sys.path:
    sys.path.insert(0, str(_oasis_root))

# Enforce thread limit (4 max)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
for var in THREAD_VARS:
    os.environ[var] = "4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all OASIS CIB modular test suites with thread caps and structured reporting."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Run pytest in verbose mode (default: True).",
    )
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        default="",
        help="Pytest keyword expression to filter tests.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="cib_zoo/tests",
        help="Directory containing tests (default: cib_zoo/tests).",
    )
    parser.add_argument(
        "--junit-xml",
        type=str,
        default="",
        help="Path to export JUnit XML report (optional).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("  OASIS CIB MODULAR ENGINE — GLOBAL VERIFICATION TEST RUNNER")
    print(f"  Working Directory: {_oasis_root}")
    print(f"  Thread Limit:      4 threads (OMP, OpenBLAS, MKL, Torch)")
    print(f"  Target Test Suite: {args.test_dir}")
    print("=" * 80)

    pytest_args = [args.test_dir]
    if args.verbose:
        pytest_args.append("-v")
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    if args.junit_xml:
        pytest_args.extend([f"--junitxml={args.junit_xml}"])

    start_time = time.time()
    exit_code = pytest.main(pytest_args)
    duration = time.time() - start_time

    print("\n" + "=" * 80)
    if exit_code == 0:
        print(f"  [SUCCESS] All verification tests passed in {duration:.2f}s!")
    else:
        print(f"  [FAILURE] Tests finished with exit code {exit_code} in {duration:.2f}s.")
    print("=" * 80)

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
