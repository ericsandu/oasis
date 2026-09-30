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

# Resolve true OASIS project root on host
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    OASIS_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    OASIS_DIR="$(pwd)"
else
    OASIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${OASIS_DIR}"

SIF_CANDIDATES=(
    "${CONTAINER_IMAGE:-}"
    "${OASIS_DIR}/container/oasis_hermes.sif"
    "${OASIS_DIR}/container/oasis_base.sif"
    "${OASIS_DIR}/oasis.sif"
    "/grid/share/images/pytorch_latest.sif"
)
CONTAINER_IMAGE=""
for s in "${SIF_CANDIDATES[@]}"; do
    if [ -n "$s" ] && [ -f "$s" ]; then
        CONTAINER_IMAGE="$s"
        break
    fi
done
PRESET="${1:-s1}"
NUM_BOTS="${2:-8}"
MAX_STEPS="${3:-10}"
OUTPUT_JSON="${4:-cib_results_${SLURM_JOB_ID:-manual}.json}"

echo "[FEP Launcher] Starting job on node $(hostname) at $(date)"
echo "[FEP Launcher] Project Directory: ${OASIS_DIR}"
echo "[FEP Launcher] Preset: ${PRESET} | Bots: ${NUM_BOTS} | Steps: ${MAX_STEPS}"
echo "[FEP Launcher] Threads capped at: ${OMP_NUM_THREADS}"

if command -v apptainer &> /dev/null && [ -f "${CONTAINER_IMAGE}" ]; then
    echo "[FEP Launcher] Executing inside Apptainer container: ${CONTAINER_IMAGE}"
    apptainer exec --nv \
        --bind "${OASIS_DIR}:/app" \
        --bind "${OASIS_DIR}:/workspace" \
        --pwd /app \
        "${CONTAINER_IMAGE}" \
        python3 cib_zoo/runner/run_fep.py \
            --preset "${PRESET}" \
            --num-bots "${NUM_BOTS}" \
            --max-steps "${MAX_STEPS}" \
            --output-json "${OUTPUT_JSON}"
else
    echo "[FEP Launcher] Running directly via host environment with Poetry"
    cd "${OASIS_DIR}"
    poetry run python cib_zoo/runner/run_fep.py \
        --preset "${PRESET}" \
        --num-bots "${NUM_BOTS}" \
        --max-steps "${MAX_STEPS}" \
        --output-json "${OUTPUT_JSON}"
fi

# ==============================================================================
# Hermes Autonomous Diagnostic Gateway & Discord Webhook Notification
# ==============================================================================
SIM_EXIT_CODE=$?
if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    echo "[FEP Launcher] Triggering Hermes Autonomous Diagnostic Gateway..."
    LOG_FILE="cib_fep_${SLURM_JOB_ID:-manual}.log"
    python3 scripts/hermes_diagnostic_gateway.py \
        --results-json "${OUTPUT_JSON}" \
        --log-file "${LOG_FILE}" \
        --db-path "twitter_simulation.db" \
        --webhook-url "${DISCORD_WEBHOOK_URL}" \
        --vllm-url "${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
        --model "${HERMES_MODEL:-Qwen/Qwen3.8-27B}" || true
fi

echo "[FEP Launcher] Execution completed at $(date) (exit code: ${SIM_EXIT_CODE})"
exit ${SIM_EXIT_CODE}
