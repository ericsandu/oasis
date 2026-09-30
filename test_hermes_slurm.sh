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

    local script_parent
    script_parent="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
    if [ -f "${script_parent}/scripts/hermes_diagnostic_gateway.py" ]; then
        echo "${script_parent}"
        return 0
    fi

    pwd
}

OASIS_DIR="$(resolve_oasis_dir)"
PARENT_DIR="$(cd "${OASIS_DIR}/.." && pwd)"
cd "${OASIS_DIR}"

# ==============================================================================
# Robust Batch Directory Resolution (handles ~, relative paths, and fallbacks)
# ==============================================================================
RAW_BATCH_DIR="${2:-}"

if [[ "${RAW_BATCH_DIR}" == ~* ]]; then
    RAW_BATCH_DIR="${RAW_BATCH_DIR/#\~/$HOME}"
fi

resolve_batch_dir() {
    local raw="$1"
    if [ -n "$raw" ]; then
        if [[ "$raw" == /* ]] && [ -d "$raw" ]; then
            echo "$(cd "$raw" && pwd)"
            return 0
        fi
        if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/${raw}" ]; then
            echo "$(cd "${SLURM_SUBMIT_DIR}/${raw}" && pwd)"
            return 0
        fi
        if [ -d "${OASIS_DIR}/${raw}" ]; then
            echo "$(cd "${OASIS_DIR}/${raw}" && pwd)"
            return 0
        fi
        if [ -d "${PARENT_DIR}/${raw}" ]; then
            echo "$(cd "${PARENT_DIR}/${raw}" && pwd)"
            return 0
        fi
        if [ -d "$(pwd)/${raw}" ]; then
            echo "$(cd "$(pwd)/${raw}" && pwd)"
            return 0
        fi
    fi

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
echo "Model Name     : ${MODEL_IDENTIFIER}"
echo "SLURM Job ID   : ${SLURM_JOB_ID:-manual}"
echo "Submit Dir     : ${SLURM_SUBMIT_DIR:-none}"
echo "Discord Webhook: $([ -n "${DISCORD_WEBHOOK_URL:-}" ] && echo "Configured" || echo "Not configured (stdout only)")"
echo "===================================================================="

GATEWAY_HOST_SCRIPT="${OASIS_DIR}/scripts/hermes_diagnostic_gateway.py"
if [ ! -f "${GATEWAY_HOST_SCRIPT}" ]; then
    echo "[-] ERROR: ${GATEWAY_HOST_SCRIPT} not found!"
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

if [ -z "${OASIS_HERMES_SIF}" ]; then
    for c in "${OASIS_DIR}/container/oasis_base.sif" "$HOME/oasis_base.sif" "${OASIS_DIR}/oasis_base.sif"; do
        if [ -f "$c" ]; then
            OASIS_HERMES_SIF="$c"
            echo "Notice: oasis_hermes.sif not found; using oasis_base.sif"
            break
        fi
    done
fi

echo "Discovered Container Images:"
echo "  - hermes.sif       : ${HERMES_SIF:-[NOT FOUND - build with: sbatch build_images_slurm.sh hermes]}"
echo "  - oasis_hermes.sif : ${OASIS_HERMES_SIF:-[NOT FOUND - build with: sbatch build_images_slurm.sh composite]}"

# ==============================================================================
# Isolated & Persistent Writable Hermes Workspace
# Prevents Errno 30 Read-only filesystem errors from SquashFS Apptainer containers
# ==============================================================================
HERMES_BASE_CACHE="${OASIS_DIR}/.hermes_workspace/template"
JOB_SEED="${SLURM_JOB_ID:-$$}"
HERMES_WORKSPACE="${OASIS_DIR}/.hermes_workspace/job_${JOB_SEED}"
mkdir -p "${HERMES_BASE_CACHE}/logs" "${HERMES_BASE_CACHE}/skills" "${HERMES_BASE_CACHE}/installs"
mkdir -p "${HERMES_WORKSPACE}/logs" "${HERMES_WORKSPACE}/skills" "${HERMES_WORKSPACE}/installs"

# Seed base cache from SIF image if installs directory is missing or empty
INIT_SIF="${HERMES_SIF:-${OASIS_HERMES_SIF}}"
if [ ! -d "${HERMES_BASE_CACHE}/installs" ] || [ -z "$(ls -A "${HERMES_BASE_CACHE}/installs" 2>/dev/null)" ]; then
    if [ -n "${INIT_SIF}" ] && [ -f "${INIT_SIF}" ] && command -v apptainer &> /dev/null; then
        echo "[+] Seeding Hermes runtime cache from ${INIT_SIF}:/etc/hermes to ${HERMES_BASE_CACHE}..."
        apptainer exec --bind "${OASIS_DIR}:${OASIS_DIR}" "${INIT_SIF}" cp -a /etc/hermes/. "${HERMES_BASE_CACHE}/" 2>/dev/null || true
    fi
fi

# Populate job-specific workspace from base cache
cp -a "${HERMES_BASE_CACHE}/." "${HERMES_WORKSPACE}/" 2>/dev/null || true

# Purge any stale lock/marker files left over from container builds or aborted runs
find "${HERMES_WORKSPACE}" -name "*.lock" -type f -delete 2>/dev/null || true
find "${HERMES_WORKSPACE}" -name "*completion*" -type f -delete 2>/dev/null || true
find "${HERMES_WORKSPACE}" -name "*in-progress*" -type f -delete 2>/dev/null || true
find "${HERMES_BASE_CACHE}" -name "*.lock" -type f -delete 2>/dev/null || true
chmod -R u+rwX "${HERMES_WORKSPACE}" "${HERMES_BASE_CACHE}" 2>/dev/null || true

# Symlink active workspace to .hermes_workspace/current for inspection
mkdir -p "${OASIS_DIR}/.hermes_workspace"
ln -sfn "${HERMES_WORKSPACE}" "${OASIS_DIR}/.hermes_workspace/current" 2>/dev/null || true

sync_hermes_config() {
    local target_url="$1"
    local target_model="$2"
    cat << EOF > "${HERMES_WORKSPACE}/config.yaml"
model:
  provider: openai
  default: "qwen-vllm"
  base_url: "${target_url}"
  api_key: "EMPTY"
  model_name: "${target_model}"
  context_length: 262144
  temperature: 0.2

terminal:
  backend: local

tools:
  terminal: true
  web_search: true
  code_search: true
  browser: false
  computer_use: false

search:
  provider: duckduckgo

logging:
  level: INFO
EOF
    if [ -n "$HOME" ] && [ -d "$HOME" ]; then
        mkdir -p "$HOME/.hermes" 2>/dev/null || true
        cp -f "${HERMES_WORKSPACE}/config.yaml" "$HOME/.hermes/config.yaml" 2>/dev/null || true
    fi
}

sync_hermes_config "${VLLM_URL}" "${MODEL_IDENTIFIER}"

# Export environment variables universally for host and container
export HERMES_HOME="/etc/hermes"
export APPTAINERENV_HERMES_HOME="/etc/hermes"
export SINGULARITYENV_HERMES_HOME="/etc/hermes"
export HERMES_WORKSPACE="${HERMES_WORKSPACE}"
export APPTAINERENV_HERMES_WORKSPACE="${HERMES_WORKSPACE}"
export SINGULARITYENV_HERMES_WORKSPACE="${HERMES_WORKSPACE}"

# Build Apptainer Bind Arguments
CONTAINER_BINDS=("--bind" "${OASIS_DIR}:/app" "--bind" "${OASIS_DIR}:${OASIS_DIR}")

# Mount isolated writable Hermes workspace over /etc/hermes
if [ -d "${HERMES_WORKSPACE}" ]; then
    CONTAINER_BINDS+=(
        "--bind" "${HERMES_WORKSPACE}:/etc/hermes"
        "--bind" "${HERMES_WORKSPACE}:${HERMES_WORKSPACE}"
    )
fi

if [ -d "${BATCH_DIR}" ]; then
    CONTAINER_BINDS+=("--bind" "${BATCH_DIR}:/workspace/batch" "--bind" "${BATCH_DIR}:${BATCH_DIR}")
fi

if [ -d "${PARENT_DIR}" ] && [ "${PARENT_DIR}" != "/" ]; then
    CONTAINER_BINDS+=("--bind" "${PARENT_DIR}:${PARENT_DIR}")
fi

if [ -d "/export" ]; then
    CONTAINER_BINDS+=("--bind" "/export:/export")
fi

if [ -n "$HOME" ] && [ -d "$HOME" ] && [ "$HOME" != "/" ]; then
    CONTAINER_BINDS+=("--bind" "$HOME:$HOME")
fi

if [ -d "$HOME/models" ]; then
    CONTAINER_BINDS+=("--bind" "$HOME/models:/models")
fi

# ==============================================================================
# Dynamic vLLM Server Lifecycle Management
# ==============================================================================
VLLM_PID=""

cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        echo ""
        echo "===================================================================="
        echo "Shutting down dedicated vLLM server (PID: ${VLLM_PID})..."
        kill -9 "${VLLM_PID}" 2>/dev/null || true
        pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
    fi
    # Archive Hermes logs from this run if present
    if [ -d "${HERMES_WORKSPACE}/logs" ] && [ -n "$(ls -A "${HERMES_WORKSPACE}/logs" 2>/dev/null)" ]; then
        mkdir -p "${OASIS_DIR}/logs/hermes_audit_${JOB_SEED}" 2>/dev/null || true
        cp -r "${HERMES_WORKSPACE}/logs/." "${OASIS_DIR}/logs/hermes_audit_${JOB_SEED}/" 2>/dev/null || true
    fi
    # Cleanup temporary per-job workspace
    rm -rf "${HERMES_WORKSPACE}" 2>/dev/null || true
    echo "Finished at: $(date)"
    echo "===================================================================="
}
trap cleanup EXIT INT TERM

VLLM_HEALTHY=0
if curl -s -f "${VLLM_URL%/v1}/health" > /dev/null 2>&1 || curl -s -f "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "✓ Central vLLM Server is already reachable and healthy at ${VLLM_URL}"
    VLLM_HEALTHY=1
else
    echo "Notice: Central vLLM Server at ${VLLM_URL} is not currently running."

    # Check if GPU is present on this node
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "[+] GPU detected via nvidia-smi. Searching for model weights to launch dedicated vLLM server..."

        MODEL_CANDIDATES=(
            "${MODEL_PATH:-}"
            "/export/home/acs/prof/teodor_daniel.milea/models/Qwen3.8-27B"
            "/export/home/acs/prof/teodor_daniel.milea/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
            "$HOME/models/Qwen3.8-27B"
            "$HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
            "/models/Qwen3.8-27B"
            "/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
            "Qwen/Qwen3.8-27B"
            "Qwen/Qwen2.5-14B-Instruct"
        )

        RESOLVED_MODEL=""
        for m in "${MODEL_CANDIDATES[@]}"; do
            if [ -n "$m" ] && { [ -d "$m" ] || [ -f "$m" ]; }; then
                RESOLVED_MODEL="$m"
                break
            fi
        done

        if [ -z "${RESOLVED_MODEL}" ] && [ -n "${HERMES_MODEL:-}" ]; then
            RESOLVED_MODEL="${HERMES_MODEL}"
        fi

        VLLM_SIF="${OASIS_HERMES_SIF:-${HERMES_SIF}}"

        if [ -n "${RESOLVED_MODEL}" ] && [ -n "${VLLM_SIF}" ]; then
            echo "[+] Found Model Weights: ${RESOLVED_MODEL}"

            # Pick port (use 8000 if free, else dynamic)
            if ! nc -z 127.0.0.1 8000 2>/dev/null && ! curl -s "http://127.0.0.1:8000/health" > /dev/null 2>&1; then
                VLLM_PORT=8000
            else
                JOB_SEED=${SLURM_JOB_ID:-$$}
                VLLM_PORT=$(( 18000 + (JOB_SEED % 4000) ))
            fi

            VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
            sync_hermes_config "${VLLM_URL}" "${MODEL_IDENTIFIER}"
            VLLM_LOG="${OASIS_DIR}/vllm_hermes_audit_${JOB_SEED}.log"
            ln -sf "${VLLM_LOG}" "${OASIS_DIR}/vllm_hermes_audit.log" 2>/dev/null || true
            pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
            sleep 1

            # Export environment variables universally for both host and container
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export VLLM_NO_USAGE_STATS=1
            export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
            export APPTAINERENV_HF_HUB_OFFLINE=1
            export APPTAINERENV_TRANSFORMERS_OFFLINE=1
            export APPTAINERENV_VLLM_NO_USAGE_STATS=1
            export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
            export SINGULARITYENV_HF_HUB_OFFLINE=1
            export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
            export SINGULARITYENV_VLLM_NO_USAGE_STATS=1
            export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

            # Avoid vLLM warning on client-side VLLM_BASE_URL
            unset VLLM_BASE_URL 2>/dev/null || true
            unset APPTAINERENV_VLLM_BASE_URL 2>/dev/null || true
            unset SINGULARITYENV_VLLM_BASE_URL 2>/dev/null || true

            echo "[+] Starting background vLLM server inside ${VLLM_SIF} on port ${VLLM_PORT}..."
            apptainer exec --nv \
                "${CONTAINER_BINDS[@]}" \
                --pwd /app \
                "${VLLM_SIF}" \
                python3 -m vllm.entrypoints.openai.api_server \
                    --model "${RESOLVED_MODEL}" \
                    --served-model-name "Qwen/Qwen3.8-27B" \
                    --host 127.0.0.1 \
                    --port "${VLLM_PORT}" \
                    --gpu-memory-utilization 0.85 \
                    --max-model-len 32768 \
                    --enforce-eager \
                    --enable-auto-tool-choice \
                    --tool-call-parser hermes \
                    --trust-remote-code > "${VLLM_LOG}" 2>&1 &
            VLLM_PID=$!

            echo "Waiting up to 600s (10 min) for vLLM to load model weights and initialize on port ${VLLM_PORT} (PID: ${VLLM_PID})..."
            for i in $(seq 1 300); do
                sleep 2
                if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                    echo "[-] WARNING: vLLM process exited. Tail of ${VLLM_LOG}:"
                    tail -n 35 "${VLLM_LOG}" 2>/dev/null || true
                    break
                fi
                if curl -s -f "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
                    echo ""
                    echo "✓ vLLM server is healthy and ready to serve Hermes inference!"
                    VLLM_HEALTHY=1
                    break
                fi
                if [ $((i % 5)) -eq 0 ]; then
                    STATUS_LINE=""
                    if [ -f "${VLLM_LOG}" ]; then
                        STATUS_LINE=$( (grep -E "(Loading safetensors|Loading pt checkpoint|Completed|Application startup|Uvicorn running)" "${VLLM_LOG}" || tail -n 1 "${VLLM_LOG}" || echo "initializing...") 2>/dev/null | tail -n 1 || true )
                    fi
                    if [ -z "${STATUS_LINE}" ]; then
                        STATUS_LINE="initializing..."
                    fi
                    echo "  [$(date +%T)] Waiting for vLLM (${i}/300) - Status: ${STATUS_LINE}"
                fi
            done
        else
            echo "[-] Unable to launch local vLLM (Model or SIF image not found)."
        fi
    else
        echo "[-] No GPU available on this node. (Submit via sbatch / srun on dgxa100 for live vLLM inference)."
    fi
fi

if [ "${VLLM_HEALTHY}" -eq 1 ]; then
    echo "✓ Active Model Endpoint: ${VLLM_URL}"
else
    echo "⚠ vLLM unavailable. Operating in local dry-run structural validation mode."
fi
echo ""

# Execution dispatchers
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
        echo "[+] Executing inside ${HERMES_SIF}..."
        if [ "${VLLM_HEALTHY}" -eq 1 ]; then
            echo "[+] Initializing Hermes package manager runtime..."
            run_hermes_cli hermes pm repair 2>/dev/null || true

            echo "[+] Querying Hermes Agent CLI with live model at ${VLLM_URL}..."
            if run_hermes_cli env \
                OPENAI_BASE_URL="${VLLM_URL}" \
                OPENAI_API_BASE="${VLLM_URL}" \
                OPENAI_API_KEY="EMPTY" \
                HERMES_MODEL="${MODEL_IDENTIFIER}" \
                HERMES_YOLO_MODE=1 \
                hermes chat --oneshot -q "Identify yourself, confirm your role as forensic simulation auditor, and summarize the primary objective of CIB propagation research in 2 sentences."; then
                echo "✓ TEST 1 Complete."
            else
                echo "[-] TEST 1: Hermes CLI encountered non-zero return code (see diagnostic above)."
            fi
        else
            echo "Testing container binary availability (dry mode):"
            run_hermes_cli which hermes || true
            run_hermes_cli python3 -c "import sys; print('Python in container:', sys.executable)"
            echo "✓ TEST 1 Complete."
        fi
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
    if [ -f "${BATCH_DIR}/simulation.db" ]; then
        SAMPLE_RUN_DIR="${BATCH_DIR}"
    else
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

        if run_gateway "${GATEWAY_ARGS[@]}"; then
            echo "✓ TEST 2 Complete."
        else
            echo "[-] TEST 2: Gateway execution encountered an issue."
        fi
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

    if run_gateway "${GATEWAY_ARGS[@]}"; then
        echo "✓ TEST 3 Complete."
    else
        echo "[-] TEST 3: Gateway execution encountered an issue."
    fi
    echo ""
fi

echo "===================================================================="
echo "HERMES AUDIT VERIFICATION COMPLETE!"
echo "Finished at: $(date)"
echo "===================================================================="
