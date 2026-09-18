#!/usr/bin/env bash
# ==============================================================================
# SLURM Apptainer Execution Launcher for UPB Grid (fep8.grid.pub.ro)
# ==============================================================================
#SBATCH --job-name=cib_propagation
#SBATCH --partition=grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cib_fep_%j.log
#SBATCH --error=cib_fep_%j.err

set -eo pipefail

# Cap thread utilization across mathematical runtime libraries to 4 threads
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/grid/share/images/pytorch_latest.sif}"
PRESET="${1:-s1}"
NUM_BOTS="${2:-8}"
MAX_STEPS="${3:-10}"
OUTPUT_JSON="${4:-cib_results_${SLURM_JOB_ID:-manual}.json}"

echo "[FEP Launcher] Starting job on node $(hostname) at $(date)"
echo "[FEP Launcher] Preset: ${PRESET} | Bots: ${NUM_BOTS} | Steps: ${MAX_STEPS}"
echo "[FEP Launcher] Threads capped at: ${OMP_NUM_THREADS}"

if command -v apptainer &> /dev/null && [ -f "${CONTAINER_IMAGE}" ]; then
    echo "[FEP Launcher] Executing inside Apptainer container: ${CONTAINER_IMAGE}"
    apptainer exec --nv \
        --bind "${SCRIPT_DIR}:/workspace" \
        --pwd /workspace \
        "${CONTAINER_IMAGE}" \
        poetry run python cib_zoo/runner/run_fep.py \
            --preset "${PRESET}" \
            --num-bots "${NUM_BOTS}" \
            --max-steps "${MAX_STEPS}" \
            --output-json "${OUTPUT_JSON}"
else
    echo "[FEP Launcher] Running directly via host environment with Poetry"
    cd "${SCRIPT_DIR}"
    poetry run python cib_zoo/runner/run_fep.py \
        --preset "${PRESET}" \
        --num-bots "${NUM_BOTS}" \
        --max-steps "${MAX_STEPS}" \
        --output-json "${OUTPUT_JSON}"
fi

echo "[FEP Launcher] Execution completed successfully at $(date)"
