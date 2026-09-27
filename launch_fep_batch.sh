#!/usr/bin/env bash
# ==============================================================================
# SLURM Apptainer Batch Execution Orchestrator for UPB Grid (fep8.grid.pub.ro)
# 
# Executes the 10-Run CIB Propagation Experiment Matrix (~1-2 hours)
# Target Hardware: 1x NVIDIA A100 GPU (Partition: dgxa100, ucsx, or grid)
# Engine: OASIS Social Simulation + vLLM Continuous Batching + Gorse Recommender
# ==============================================================================
#SBATCH --job-name=cib_fep_batch
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_cib_batch_%j.out
#SBATCH --error=slurm_cib_batch_%j.err

set -eo pipefail

# Cap thread utilization across math runtime libraries
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# Resolve true OASIS project root on host
# In SLURM batch jobs, ${BASH_SOURCE[0]} points to /var/spool/slurmd/job*/slurm_script (root-owned)
# We must use $SLURM_SUBMIT_DIR (where sbatch was invoked) so all files land in the oasis repo
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    OASIS_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    OASIS_DIR="$(pwd)"
else
    OASIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${OASIS_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${OASIS_DIR}/experiments/batch_${TIMESTAMP}"
mkdir -p "${BATCH_DIR}"
mkdir -p "${OASIS_DIR}/data"
mkdir -p "${OASIS_DIR}/experiments"

echo "===================================================================="
echo "Starting OASIS CIB Batch Experiment Matrix on Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual} | Partition: ${SLURM_JOB_PARTITION:-local}"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Project Directory: ${OASIS_DIR}"
echo "Output Directory: ${BATCH_DIR}"
echo "Start Time: $(date)"
echo "===================================================================="

# 1. Locate Apptainer SIF Image
SIF_CANDIDATES=(
    "${CONTAINER_IMAGE:-}"
    "$HOME/eric_sandu/oasis.sif"
    "$HOME/oasis.sif"
    "$HOME/camel-oasis.sif"
    "$HOME/pytorch.sif"
    "${OASIS_DIR}/oasis.sif"
)

CONTAINER_SIF=""
for s in "${SIF_CANDIDATES[@]}"; do
    if [ -n "$s" ] && [ -f "$s" ]; then
        CONTAINER_SIF="$s"
        break
    fi
done

if [ -n "$CONTAINER_SIF" ] && command -v apptainer &> /dev/null; then
    echo "✓ Using Apptainer Container: ${CONTAINER_SIF}"
    CONTAINER_RUN="apptainer exec --nv \
        --bind $HOME/models:/models \
        --bind ${OASIS_DIR}:/app \
        --bind ${OASIS_DIR}:/workspace \
        --bind ${OASIS_DIR}/data:/app/data \
        --bind ${OASIS_DIR}/experiments:/app/experiments \
        --bind ${BATCH_DIR}:/workspace/experiments/current_batch \
        --pwd /app \
        ${CONTAINER_SIF}"
    # Container has all packages pre-installed globally in python3
    PY_EXEC="${CONTAINER_RUN} python3"
else
    echo "Notice: No Apptainer container found or apptainer command unavailable. Running directly in host Poetry environment."
    CONTAINER_RUN=""
    PY_EXEC="poetry run python"
fi

# 2. Resolve Model Path
MODEL_CANDIDATES=(
    "${MODEL_PATH:-}"
    "/models/Qwen3.8-27B"
    "$HOME/models/Qwen3.8-27B"
    "/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
    "$HOME/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
    "/models/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

RESOLVED_MODEL="Qwen/Qwen2.5-14B-Instruct"
for m in "${MODEL_CANDIDATES[@]}"; do
    if [ -n "$m" ]; then
        if [ -d "$m" ] || [ -f "$m" ]; then
            RESOLVED_MODEL="$m"
            break
        fi
    fi
done
echo "✓ Using LLM Model: ${RESOLVED_MODEL}"

# 3. Setup Process Cleanup Hook
cleanup() {
    echo ""
    echo "Shutting down background services (vLLM, Gorse)..."
    [ -n "$VLLM_PID" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
    [ -n "$GORSE_PID" ] && kill -9 "$GORSE_PID" 2>/dev/null || true
    pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -u "$USER" -f "gorse-in-one" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Clean up any orphan processes on this node
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -u "$USER" -f "gorse-in-one" 2>/dev/null || true
sleep 1

# 4. Launch Background Gorse Recommender Daemon (optional/fallback)
if command -v gorse-in-one &> /dev/null || [ -n "$CONTAINER_RUN" ]; then
    echo "Starting Gorse recommender engine on port 8088..."
    ${CONTAINER_RUN} bash -c "
    if command -v gorse-in-one &> /dev/null; then
        exec gorse-in-one -c gorse_config.toml
    fi
    " > "${BATCH_DIR}/gorse.log" 2>&1 &
    GORSE_PID=$!
fi

# 5. Launch vLLM Server on Allocated GPU
JOB_SEED=${SLURM_JOB_ID:-$$}
VLLM_PORT=$(( 18000 + (JOB_SEED % 4000) ))
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

echo "Starting vLLM server on isolated port ${VLLM_PORT}..."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_FLASHINFER_SAMPLER=0

VLLM_EXTRA_ARGS=()
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

${CONTAINER_RUN} python3 -m vllm.entrypoints.openai.api_server \
    --model "${RESOLVED_MODEL}" \
    --served-model-name "${RESOLVED_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    ${VLLM_EXTRA_ARGS[@]+"${VLLM_EXTRA_ARGS[@]}"} \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code > "${BATCH_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to become healthy on port ${VLLM_PORT} (PID: ${VLLM_PID})..."
VLLM_READY=0
for i in $(seq 1 300); do
    sleep 2
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "WARNING: vLLM process exited. Check ${BATCH_DIR}/vllm.log. Falling back to local offline mode if needed."
        break
    fi
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✓ vLLM server is healthy and ready to serve requests!"
        VLLM_READY=1
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        STATUS_LINE=$(tail -n 1 "${BATCH_DIR}/vllm.log" 2>/dev/null || echo "initializing...")
        echo "  [$(date +%T)] Waiting for vLLM (${i}/300) - Status: ${STATUS_LINE}"
    fi
done

VLLM_FLAG=""
if [ "$VLLM_READY" -eq 1 ]; then
    VLLM_FLAG="--vllm-url http://127.0.0.1:${VLLM_PORT}/v1 --model ${RESOLVED_MODEL}"
else
    echo "Notice: Proceeding without live vLLM server (using local hermetic execution)."
fi

# ==============================================================================
# 6. Execute the 10-Run CIB Experiment Matrix
# ==============================================================================
# Matrix columns: RUN_ID | PRESET | TOPOLOGY | TOPIC_MODE | BOT_RATIO | NUM_ORG | MAX_STEPS
RUNS=(
    "R01|baseline|multitopic|existing|0.00|150|12"
    "R02|s1|multitopic|existing|0.10|150|12"
    "R03|s1|multitopic|existing|0.20|150|12"
    "R04|s1|multitopic|cold_start|0.15|150|12"
    "R05|s3|multitopic|cold_start|0.15|150|12"
    "R06|baseline|polarized|existing|0.00|150|12"
    "R07|s2|polarized|existing|0.10|150|12"
    "R08|s2|polarized|existing|0.20|150|12"
    "R09|s3|polarized|existing|0.15|150|12"
    "R10|s3|polarized|cold_start|0.20|150|12"
)

TOTAL_RUNS=${#RUNS[@]}
RUN_IDX=0

echo ""
echo "===================================================================="
echo "Starting Execution of ${TOTAL_RUNS} Planned Matrix Runs"
echo "===================================================================="

for run_spec in "${RUNS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    IFS='|' read -r RUN_ID PRESET TOPOLOGY TOPIC_MODE BOT_RATIO NUM_ORG MAX_STEPS <<< "$run_spec"

    RUN_TAG="${RUN_ID}_${PRESET}_${TOPOLOGY}_${TOPIC_MODE}_ratio${BOT_RATIO}"
    RUN_DIR="${BATCH_DIR}/${RUN_TAG}"
    mkdir -p "${RUN_DIR}"
    DB_PATH="${RUN_DIR}/simulation.db"
    JSON_PATH="${RUN_DIR}/run_${RUN_ID}_results.json"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "[Run ${RUN_IDX}/${TOTAL_RUNS}] Executing ${RUN_ID}: Preset=${PRESET}, Topology=${TOPOLOGY}, TopicMode=${TOPIC_MODE}, BotRatio=${BOT_RATIO} (${MAX_STEPS} steps)"
    echo "Start Time: $(date)"
    echo "--------------------------------------------------------------------"

    RUN_START_TIME=$(date +%s)

    ${PY_EXEC} cib_zoo/runner/run_fep.py \
        --preset "${PRESET}" \
        --topology "${TOPOLOGY}" \
        --topic-mode "${TOPIC_MODE}" \
        --num-organic "${NUM_ORG}" \
        --bot-ratio "${BOT_RATIO}" \
        --max-steps "${MAX_STEPS}" \
        --db-path "${DB_PATH}" \
        --output-json "${JSON_PATH}" \
        ${VLLM_FLAG}

    RUN_END_TIME=$(date +%s)
    RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
    echo "✓ [Run ${RUN_IDX}/${TOTAL_RUNS}] Completed ${RUN_ID} in ${RUN_DURATION}s. Results: ${JSON_PATH}"
done

# ==============================================================================
# 7. Aggregate Results & Generate Publication Dashboard
# ==============================================================================
echo ""
echo "===================================================================="
echo "Compiling Global Batch Analytics & Publication Dashboard..."
echo "===================================================================="

SUMMARY_JSON="${BATCH_DIR}/cib_batch_summary.json"
DASHBOARD_PNG="${BATCH_DIR}/cib_batch_dashboard.png"

${PY_EXEC} cib_zoo/runner/aggregate_batch_results.py \
    --batch-dir "${BATCH_DIR}" \
    --output-json "${SUMMARY_JSON}" \
    --output-png "${DASHBOARD_PNG}"

echo ""
echo "===================================================================="
echo "BATCH EXPERIMENT SUITE COMPLETE!"
echo "Finished at: $(date)"
echo "Artifacts generated in: ${BATCH_DIR}"
echo "  - Summary JSON: ${SUMMARY_JSON}"
echo "  - Dashboard PNG: ${DASHBOARD_PNG}"
echo "===================================================================="
ls -lh "${BATCH_DIR}"
