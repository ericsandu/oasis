#!/usr/bin/env bash
# ==============================================================================
# Global Verification Test Suite Runner for OASIS CIB Modular Engine
# Safe for local workstations, CI workflows, and UPB Grid Apptainer nodes.
# ==============================================================================
set -eo pipefail

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

poetry run python cib_zoo/runner/run_all_tests.py "$@"
