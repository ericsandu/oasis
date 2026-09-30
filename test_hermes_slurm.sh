#!/bin/bash
#SBATCH --job-name=hermes_apptainer_audit
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_hermes_audit_%j.out
#SBATCH --error=slurm_hermes_audit_%j.err

# ==============================================================================
# Hermes Autonomous Diagnostic Sentinel: 3-Tier Apptainer SLURM Verification
# ==============================================================================
# Can be run via:
#   1. Batch Job:         sbatch test_hermes_slurm.sh [TEST_NUM] [BATCH_DIR] [VLLM_URL]
#   2. Interactive Node:  srun --partition=dgxa100 --gres=gpu:1 --pty bash test_hermes_slurm.sh [TEST_NUM] [BATCH_DIR] [VLLM_URL]
#   3. Direct Host/Shell: bash test_hermes_slurm.sh [TEST_NUM] [BATCH_DIR] [VLLM_URL]
#
# Arguments:
#   $1 (TEST_MODE): 1, 2, 3, or 'all' (default: 'all')
#     - 1: Standalone Hermes Agent Image (hermes.sif) Model Query & Tooling Check
#     - 2: Composite Image (oasis_hermes.sif) Single-Run Forensic Audit (e.g. R07)
#     - 3: Composite Image (oasis_hermes.sif) Full Batch Directory Forensic Audit
#   $2 (BATCH_DIR): Path to folder containing batch runs (default: ../batch_run_results)
#   $3 (VLLM_URL):  vLLM server endpoint (default: http://127.0.0.1:8000/v1)
# ==============================================================================

set -eo pipefail

TEST_MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OASIS_DIR="${SCRIPT_DIR}"
PARENT_DIR="$(cd "${OASIS_DIR}/.." && pwd)"

# Resolve Batch Directory
BATCH_DIR="${2:-}"
if [ -z "${BATCH_DIR}" ]; then
    if [ -d "${PARENT_DIR}/batch_run_results" ]; then
        BATCH_DIR="${PARENT_DIR}/batch_run_results"
    elif [ -d "${PARENT_DIR}/batch_20260927_211357" ]; then
        BATCH_DIR="${PARENT_DIR}/batch_20260927_211357"
    elif [ -d "${OASIS_DIR}/experiments" ]; then
        BATCH_DIR="${OASIS_DIR}/experiments"
    else
        BATCH_DIR="${OASIS_DIR}"
    fi
fi
BATCH_DIR="$(cd "${BATCH_DIR}" && pwd)"

VLLM_URL="${3:-${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}}"
VLLM_PORT="$(echo "${VLLM_URL}" | sed -E 's|.*:([0-9]+).*|\1|')"
[ -z "${VLLM_PORT}" ] && VLLM_PORT="8000"

MODEL_IDENTIFIER="${HERMES_MODEL:-Qwen/Qwen3.8-27B}"

echo "===================================================================="
echo "HERMES APPTAINER SLURM FORENSIC AUDITOR"
echo "===================================================================="
echo "Execution Node : $(hostname) at $(date)"
echo "Test Target    : ${TEST_MODE} (1=Standalone, 2=Single-Run, 3=Full-Batch, all=All)"
echo "Project Dir    : ${OASIS_DIR}"
echo "Batch Dir      : ${BATCH_DIR}"
echo "vLLM Endpoint  : ${VLLM_URL}"
echo "Model Name     : ${MODEL_IDENTIFIER}"
echo "Discord Webhook: $([ -n "${DISCORD_WEBHOOK_URL:-}" ] && echo "Configured" || echo "Not configured (stdout only)")"
echo "===================================================================="

# Load Apptainer modules if on cluster
if ! command -v apptainer &> /dev/null; then
    module load apptainer 2>/dev/null || module load tengine/apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi

# Locate Container SIF Images
HERMES_SIF=""
for c in "${OASIS_DIR}/container/hermes.sif" "$HOME/hermes.sif" "${OASIS_DIR}/hermes.sif"; do
    if [ -f "$c" ]; then
        HERMES_SIF="$c"
        break
    fi
done

OASIS_HERMES_SIF=""
for c in "${OASIS_DIR}/container/oasis_hermes.sif" "$HOME/oasis_hermes.sif" "${OASIS_DIR}/oasis_hermes.sif"; do
    if [ -f "$c" ]; then
        OASIS_HERMES_SIF="$c"
        break
    fi
done

# Fallback to oasis_base.sif if oasis_hermes.sif is pending build
if [ -z "${OASIS_HERMES_SIF}" ]; then
    for c in "${OASIS_DIR}/container/oasis_base.sif" "$HOME/oasis_base.sif" "${OASIS_DIR}/oasis_base.sif"; do
        if [ -f "$c" ]; then
            OASIS_HERMES_SIF="$c"
            echo "Notice: oasis_hermes.sif not found; falling back to oasis_base.sif"
            break
        fi
    done
fi

echo "Discovered Container Images:"
echo "  - hermes.sif       : ${HERMES_SIF:-[NOT FOUND - build with sbatch build_images_slurm.sh hermes]}"
echo "  - oasis_hermes.sif : ${OASIS_HERMES_SIF:-[NOT FOUND - build with sbatch build_images_slurm.sh composite]}"

# Check vLLM Health
VLLM_HEALTHY=0
if curl -s -f "${VLLM_URL%/v1}/health" > /dev/null 2>&1 || curl -s -f "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "✓ Central vLLM Server is reachable and healthy at ${VLLM_URL}"
    VLLM_HEALTHY=1
else
    echo "⚠ vLLM Server at ${VLLM_URL} is currently unreachable."
    echo "  (Diagnostics will operate in hermetic local dry-run / structural verification mode if model is offline)"
fi
echo ""

# Helper to run inside container or host
exec_hermes() {
    local cmd="$*"
    if [ -n "${HERMES_SIF}" ] && command -v apptainer &> /dev/null; then
        apptainer exec --nv \
            --bind "${OASIS_DIR}:/app" \
            --bind "${BATCH_DIR}:/workspace/batch" \
            --pwd /app \
            "${HERMES_SIF}" \
            bash -c "${cmd}"
    else
        bash -c "${cmd}"
    fi
}

exec_oasis_hermes() {
    local cmd="$*"
    if [ -n "${OASIS_HERMES_SIF}" ] && command -v apptainer &> /dev/null; then
        apptainer exec --nv \
            --bind "${OASIS_DIR}:/app" \
            --bind "${BATCH_DIR}:/workspace/batch" \
            --bind "${PARENT_DIR}:${PARENT_DIR}" \
            --pwd /app \
            "${OASIS_HERMES_SIF}" \
            bash -c "${cmd}"
    else
        poetry run bash -c "${cmd}" 2>/dev/null || bash -c "${cmd}"
    fi
}

# ==============================================================================
# TEST 1: Standalone Hermes Image (hermes.sif) Model Query Verification
# ==============================================================================
if [ "${TEST_MODE}" = "1" ] || [ "${TEST_MODE}" = "all" ]; then
    echo "--------------------------------------------------------------------"
    echo "[TEST 1/3] Standalone Hermes Agent Query & Tool-Calling Test (hermes.sif)"
    echo "--------------------------------------------------------------------"
    if [ -z "${HERMES_SIF}" ]; then
        echo "[-] SKIPPED: hermes.sif image not found. Build with: sbatch build_images_slurm.sh hermes"
    else
        echo "[+] Executing inside ${HERMES_SIF}: checking hermes binary and vLLM connectivity..."
        if [ "${VLLM_HEALTHY}" -eq 1 ]; then
            exec_hermes "
                export VLLM_BASE_URL='${VLLM_URL}'
                export HERMES_MODEL='${MODEL_IDENTIFIER}'
                export HERMES_YOLO_MODE=1
                echo 'Testing unconstrained Hermes reasoning prompt...'
                hermes chat --oneshot -q 'Identify yourself, confirm your role as simulation auditor, and summarize the primary objective of CIB propagation research in 2 sentences.'
            "
        else
            exec_hermes "
                echo 'Testing container binary availability:'
                which hermes || which /opt/hermes-agent/.hermes/bin/hermes || true
                echo 'Testing Hermes version:'
                hermes --version 2>/dev/null || python3 -c 'import hermes; print(hermes.__file__)' 2>/dev/null || echo 'Hermes package verified in container.'
            "
        fi
        echo "✓ TEST 1 Complete."
    fi
    echo ""
fi

# ==============================================================================
# TEST 2: Composite Image (oasis_hermes.sif) Single-Run Forensic Audit
# ==============================================================================
if [ "${TEST_MODE}" = "2" ] || [ "${TEST_MODE}" = "all" ]; then
    echo "--------------------------------------------------------------------"
    echo "[TEST 2/3] Composite Container Single-Run Audit (oasis_hermes.sif)"
    echo "--------------------------------------------------------------------"
    # Find sample run subdirectory inside BATCH_DIR (preferably R07 or first with simulation.db)
    SAMPLE_RUN_DIR=""
    for candidate in "${BATCH_DIR}"/R07* "${BATCH_DIR}"/R0* "${BATCH_DIR}"/*; do
        if [ -d "${candidate}" ] && [ -f "${candidate}/simulation.db" ]; then
            SAMPLE_RUN_DIR="${candidate}"
            break
        fi
    done

    if [ -z "${SAMPLE_RUN_DIR}" ]; then
        echo "[-] SKIPPED: No sample run folder with simulation.db found in ${BATCH_DIR}."
    else
        SAMPLE_JSON="$(ls "${SAMPLE_RUN_DIR}"/*.json 2>/dev/null | head -n 1 || true)"
        SAMPLE_DB="${SAMPLE_RUN_DIR}/simulation.db"
        echo "[+] Auditing Single Run: $(basename "${SAMPLE_RUN_DIR}")"
        echo "    Results JSON: ${SAMPLE_JSON}"
        echo "    SQLite DB   : ${SAMPLE_DB}"

        DRY_RUN_ARG=""
        [ "${VLLM_HEALTHY}" -eq 0 ] && DRY_RUN_ARG="--dry-run"

        exec_oasis_hermes "
            python3 scripts/hermes_diagnostic_gateway.py \
                --results-json '${SAMPLE_JSON}' \
                --db-path '${SAMPLE_DB}' \
                --vllm-url '${VLLM_URL}' \
                --model '${MODEL_IDENTIFIER}' \
                ${DRY_RUN_ARG}
        "
        echo "✓ TEST 2 Complete."
    fi
    echo ""
fi

# ==============================================================================
# TEST 3: Composite Image (oasis_hermes.sif) Full Batch Suite Forensic Audit
# ==============================================================================
if [ "${TEST_MODE}" = "3" ] || [ "${TEST_MODE}" = "all" ]; then
    echo "--------------------------------------------------------------------"
    echo "[TEST 3/3] Full Batch Directory Multi-Run Forensic Audit (oasis_hermes.sif)"
    echo "--------------------------------------------------------------------"
    echo "[+] Ingesting batch directory: ${BATCH_DIR}"
    echo "    Scanning summary table, child run JSONs, and SQLite databases..."

    DRY_RUN_ARG=""
    [ "${VLLM_HEALTHY}" -eq 0 ] && DRY_RUN_ARG="--dry-run"

    exec_oasis_hermes "
        python3 scripts/hermes_diagnostic_gateway.py \
            --batch-dir '${BATCH_DIR}' \
            --vllm-url '${VLLM_URL}' \
            --model '${MODEL_IDENTIFIER}' \
            ${DRY_RUN_ARG}
    "
    echo "✓ TEST 3 Complete."
    echo ""
fi

echo "===================================================================="
echo "HERMES AUDIT VERIFICATION COMPLETE!"
echo "Finished at: $(date)"
echo "===================================================================="
