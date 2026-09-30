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
#   2. Interactive Node:  srun --account=phd --partition=dgxa100 --gres=gpu:1 --pty bash test_hermes_slurm.sh [TEST_NUM] [BATCH_DIR] [VLLM_URL]
#   3. Direct Host/Shell: bash test_hermes_slurm.sh [TEST_NUM] [BATCH_DIR] [VLLM_URL]
#
# Arguments:
#   $1 (TEST_MODE): 1, 2, 3, or 'all' (default: 'all')
#     - 1: Standalone Hermes Agent Image (hermes.sif) Model Query & Tooling Check
#     - 2: Composite Image (oasis_hermes.sif) Single-Run Forensic Audit (e.g. R07)
#     - 3: Composite Image (oasis_hermes.sif) Full Batch Directory Forensic Audit
#   $2 (BATCH_DIR): Path to folder containing batch runs or single run
#   $3 (VLLM_URL):  vLLM server endpoint (default: http://127.0.0.1:8000/v1)
# ==============================================================================

set -eo pipefail

TEST_MODE="${1:-all}"

# ==============================================================================
# Robust Project Root & Directory Resolution for SLURM Environments
# ==============================================================================
# When running via `sbatch`, SLURM copies the script to /var/spool/slurmd/job*/slurm_script.
# Using ${BASH_SOURCE[0]} in sbatch resolves to /var/spool/slurmd/, which breaks relative paths.
# We resolve the true project root by searching SLURM_SUBMIT_DIR, pwd, and marker files.
resolve_oasis_dir() {
    local candidates=(
        "${OASIS_DIR:-}"
        "${SLURM_SUBMIT_DIR:-}"
        "${SLURM_SUBMIT_DIR:-}/oasis"
        "$(pwd)"
        "$(pwd)/oasis"
        "$HOME/eric_sandu/oasis"
        "$HOME/oasis"
    )
    for c in "${candidates[@]}"; do
        if [ -n "$c" ] && [ -f "$c/scripts/hermes_diagnostic_gateway.py" ]; then
            echo "$(cd "$c" && pwd)"
            return 0
        fi
    done

    # Secondary check via readlink if not running inside /var/spool/slurmd/
    local script_parent
    script_parent="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
    if [ -f "${script_parent}/scripts/hermes_diagnostic_gateway.py" ]; then
        echo "${script_parent}"
        return 0
    fi

    # Fallback to current working directory
    pwd
}

OASIS_DIR="$(resolve_oasis_dir)"
PARENT_DIR="$(cd "${OASIS_DIR}/.." && pwd)"
cd "${OASIS_DIR}"

# ==============================================================================
# Robust Batch Directory Resolution (handles ~, relative paths, and fallbacks)
# ==============================================================================
RAW_BATCH_DIR="${2:-}"

# Expand leading tilde (~) if present in argument
if [[ "${RAW_BATCH_DIR}" == ~* ]]; then
    RAW_BATCH_DIR="${RAW_BATCH_DIR/#\~/$HOME}"
fi

resolve_batch_dir() {
    local raw="$1"
    if [ -n "$raw" ]; then
        # 1. Direct absolute path
        if [[ "$raw" == /* ]] && [ -d "$raw" ]; then
            echo "$(cd "$raw" && pwd)"
            return 0
        fi
        # 2. Relative to SLURM_SUBMIT_DIR (where user invoked sbatch/srun)
        if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/${raw}" ]; then
            echo "$(cd "${SLURM_SUBMIT_DIR}/${raw}" && pwd)"
            return 0
        fi
        # 3. Relative to OASIS_DIR
        if [ -d "${OASIS_DIR}/${raw}" ]; then
            echo "$(cd "${OASIS_DIR}/${raw}" && pwd)"
            return 0
        fi
        # 4. Relative to PARENT_DIR
        if [ -d "${PARENT_DIR}/${raw}" ]; then
            echo "$(cd "${PARENT_DIR}/${raw}" && pwd)"
            return 0
        fi
        # 5. Relative to current working directory
        if [ -d "$(pwd)/${raw}" ]; then
            echo "$(cd "$(pwd)/${raw}" && pwd)"
            return 0
        fi
    fi

    # Auto-discovery if empty or not found:
    local auto_candidates=(
        "${OASIS_DIR}/experiments"
        "${PARENT_DIR}/batch_run_results"
        "${PARENT_DIR}/batch_20260927_211357"
        "${SLURM_SUBMIT_DIR:-}/experiments"
        "${SLURM_SUBMIT_DIR:-}/../batch_run_results"
        "$HOME/eric_sandu/oasis/experiments"
    )
    for c in "${auto_candidates[@]}"; do
        if [ -d "$c" ]; then
            # If candidate contains batch_* subdirectories, pick the latest
            local latest_sub
            latest_sub="$(ls -td "$c"/batch_* 2>/dev/null | head -n 1 || true)"
            if [ -n "$latest_sub" ] && [ -d "$latest_sub" ]; then
                echo "$(cd "$latest_sub" && pwd)"
                return 0
            fi
            echo "$(cd "$c" && pwd)"
            return 0
        fi
    done

    echo "${OASIS_DIR}"
}

BATCH_DIR="$(resolve_batch_dir "${RAW_BATCH_DIR}")"

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
echo "SLURM Job ID   : ${SLURM_JOB_ID:-manual}"
echo "Submit Dir     : ${SLURM_SUBMIT_DIR:-none}"
echo "Discord Webhook: $([ -n "${DISCORD_WEBHOOK_URL:-}" ] && echo "Configured" || echo "Not configured (stdout only)")"
echo "===================================================================="

# Sanity check: Ensure hermes_diagnostic_gateway.py exists
GATEWAY_HOST_SCRIPT="${OASIS_DIR}/scripts/hermes_diagnostic_gateway.py"
if [ ! -f "${GATEWAY_HOST_SCRIPT}" ]; then
    echo "[-] ERROR: ${GATEWAY_HOST_SCRIPT} not found!"
    echo "    Check OASIS_DIR path resolution on node $(hostname)."
    exit 1
fi

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
echo "  - hermes.sif       : ${HERMES_SIF:-[NOT FOUND - build with: sbatch build_images_slurm.sh hermes]}"
echo "  - oasis_hermes.sif : ${OASIS_HERMES_SIF:-[NOT FOUND - build with: sbatch build_images_slurm.sh composite]}"

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

# Build Apptainer Bind Arguments (Mounts /app, /workspace/batch, and exact host paths)
CONTAINER_BINDS=("--bind" "${OASIS_DIR}:/app" "--bind" "${OASIS_DIR}:${OASIS_DIR}")

if [ -d "${BATCH_DIR}" ]; then
    CONTAINER_BINDS+=("--bind" "${BATCH_DIR}:/workspace/batch" "--bind" "${BATCH_DIR}:${BATCH_DIR}")
fi

if [ -d "${PARENT_DIR}" ] && [ "${PARENT_DIR}" != "/" ]; then
    CONTAINER_BINDS+=("--bind" "${PARENT_DIR}:${PARENT_DIR}")
fi

if [ -n "$HOME" ] && [ -d "$HOME" ] && [ "$HOME" != "/" ]; then
    CONTAINER_BINDS+=("--bind" "$HOME:$HOME")
fi

# Execution dispatcher: runs hermes_diagnostic_gateway inside container if SIF exists, or directly via python3
run_gateway() {
    if [ -n "${OASIS_HERMES_SIF}" ] && command -v apptainer &> /dev/null; then
        apptainer exec --nv \
            "${CONTAINER_BINDS[@]}" \
            --pwd /app \
            "${OASIS_HERMES_SIF}" \
            python3 /app/scripts/hermes_diagnostic_gateway.py "$@"
    else
        python3 "${GATEWAY_HOST_SCRIPT}" "$@"
    fi
}

run_hermes_cli() {
    if [ -n "${HERMES_SIF}" ] && command -v apptainer &> /dev/null; then
        apptainer exec --nv \
            "${CONTAINER_BINDS[@]}" \
            --pwd /app \
            "${HERMES_SIF}" \
            "$@"
    else
        "$@"
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
            run_hermes_cli env \
                VLLM_BASE_URL="${VLLM_URL}" \
                HERMES_MODEL="${MODEL_IDENTIFIER}" \
                HERMES_YOLO_MODE=1 \
                hermes chat --oneshot -q "Identify yourself, confirm your role as simulation auditor, and summarize the primary objective of CIB propagation research in 2 sentences."
        else
            echo "Testing container binary availability:"
            run_hermes_cli which hermes || true
            run_hermes_cli python3 -c "import sys; print('Python in container:', sys.executable)"
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
    SAMPLE_RUN_DIR=""
    # Check if BATCH_DIR itself is an individual run directory
    if [ -f "${BATCH_DIR}/simulation.db" ]; then
        SAMPLE_RUN_DIR="${BATCH_DIR}"
    else
        # Search inside BATCH_DIR for child run folders with simulation.db
        for candidate in "${BATCH_DIR}"/R07* "${BATCH_DIR}"/R0* "${BATCH_DIR}"/*; do
            if [ -d "${candidate}" ] && [ -f "${candidate}/simulation.db" ]; then
                SAMPLE_RUN_DIR="${candidate}"
                break
            fi
        done
    fi

    if [ -z "${SAMPLE_RUN_DIR}" ]; then
        echo "[-] SKIPPED: No run folder with simulation.db found in ${BATCH_DIR}."
    else
        SAMPLE_JSON="$(ls "${SAMPLE_RUN_DIR}"/*results*.json "${SAMPLE_RUN_DIR}"/*.json 2>/dev/null | head -n 1 || true)"
        SAMPLE_DB="${SAMPLE_RUN_DIR}/simulation.db"
        echo "[+] Auditing Single Run: $(basename "${SAMPLE_RUN_DIR}")"
        echo "    Results JSON: ${SAMPLE_JSON}"
        echo "    SQLite DB   : ${SAMPLE_DB}"

        GATEWAY_ARGS=(
            "--results-json" "${SAMPLE_JSON}"
            "--db-path" "${SAMPLE_DB}"
            "--vllm-url" "${VLLM_URL}"
            "--model" "${MODEL_IDENTIFIER}"
        )
        if [ "${VLLM_HEALTHY}" -eq 0 ]; then
            GATEWAY_ARGS+=("--dry-run")
        fi

        run_gateway "${GATEWAY_ARGS[@]}"
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

    GATEWAY_ARGS=(
        "--batch-dir" "${BATCH_DIR}"
        "--vllm-url" "${VLLM_URL}"
        "--model" "${MODEL_IDENTIFIER}"
    )
    if [ "${VLLM_HEALTHY}" -eq 0 ]; then
        GATEWAY_ARGS+=("--dry-run")
    fi

    run_gateway "${GATEWAY_ARGS[@]}"
    echo "✓ TEST 3 Complete."
    echo ""
fi

echo "===================================================================="
echo "HERMES AUDIT VERIFICATION COMPLETE!"
echo "Finished at: $(date)"
echo "===================================================================="
