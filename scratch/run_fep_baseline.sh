#!/bin/bash
#SBATCH --job-name=oasis_baseline_apptainer
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_baseline_%j.out
#SBATCH --error=slurm_baseline_%j.err

# ==============================================================================
# OASIS Remote Baseline Run Script via Apptainer on UPB Grid (fep8.grid.pub.ro)
# 
# Container: $HOME/oasis.sif (or $HOME/pytorch.sif)
# Hardware Target: 1x NVIDIA A100-SXM4-80GB (Partition: dgxa100 or ucsx)
# Model: Pre-downloaded weights in $HOME/models/Qwen3.8-27B
# Recommender: Gorse active on port 8088 inside container
# Telemetry: Automatic extraction to experiments/baseline_<timestamp>/
# ==============================================================================

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="./experiments/baseline_${TIMESTAMP}"
DATA_DIR="./data"
DB_PATH="${DATA_DIR}/baseline_simulation.db"

mkdir -p "$EXPERIMENT_DIR" "$DATA_DIR"

echo "===================================================================="
echo "Starting OASIS Remote Baseline Run on Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: $TIMESTAMP"
echo "===================================================================="

# 1. Locate Container SIF Image
SIF_CANDIDATES=(
    "$HOME/oasis.sif"
    "$HOME/camel-oasis.sif"
    "$HOME/pytorch.sif"
)

CONTAINER_SIF=""
for s in "${SIF_CANDIDATES[@]}"; do
    if [ -f "$s" ]; then
        CONTAINER_SIF="$s"
        break
    fi
done

if [ -z "$CONTAINER_SIF" ]; then
    echo "ERROR: No container SIF file found in $HOME."
    echo "Please build oasis.sif first using: apptainer build \$HOME/oasis.sif oasis.def"
    exit 1
fi
echo "✓ Using Container: $CONTAINER_SIF"

# Helper for container execution with full GPU and filesystem passthrough
CONTAINER_RUN="apptainer exec --nv \
  --bind $HOME/models:/models \
  --bind $(pwd):/app \
  --bind $(pwd)/data:/app/data \
  --bind $(pwd)/experiments:/app/experiments \
  --pwd /app \
  $CONTAINER_SIF"

# 2. Resolve Model Path
MODEL_CANDIDATES=(
    "/models/Qwen3.8-27B"
    "/models/Qwen2.5-32B-Instruct-GPTQ-Int8"
    "/models/Qwen2.5-32B-Instruct"
    "/models/Qwen-32B"
    "$HOME/models/Qwen3.8-27B"
)

RESOLVED_MODEL=""
for cand in "${MODEL_CANDIDATES[@]}"; do
    if [ -d "$cand" ] || $CONTAINER_RUN test -d "$cand" 2>/dev/null; then
        RESOLVED_MODEL="$cand"
        break
    fi
done

if [ -z "$RESOLVED_MODEL" ]; then
    echo "Checking for any directory in /models..."
    FIRST_MODEL=$($CONTAINER_RUN bash -c "find /models -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -n 1" || true)
    if [ -n "$FIRST_MODEL" ]; then
        RESOLVED_MODEL="$FIRST_MODEL"
    else
        RESOLVED_MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
    fi
fi

echo "✓ Using Model: $RESOLVED_MODEL"

# 3. Cleanup hook for background servers
cleanup() {
    echo "Shutting down background servers..."
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$GORSE_PID" ] && kill "$GORSE_PID" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# 4. Launch Gorse Recommender Engine inside container
echo "Starting Gorse recommender engine on port 8088 inside container..."
$CONTAINER_RUN bash -c "
if command -v gorse-in-one &> /dev/null; then
    gorse-in-one -c gorse_config.toml > ${EXPERIMENT_DIR}/gorse.log 2>&1 &
    echo \$!
else
    echo 'NONE'
fi
" > /tmp/gorse_pid_$$ 2>&1 || true

GORSE_PID=$(cat /tmp/gorse_pid_$$ 2>/dev/null || echo "")
rm -f /tmp/gorse_pid_$$

# 5. Launch vLLM Server on A100 GPU inside container
VLLM_PORT=8000
echo "Starting vLLM server on port $VLLM_PORT (GPU memory utilization: 0.85) inside container..."

$CONTAINER_RUN python3 -m vllm.entrypoints.openai.api_server \
    --model "$RESOLVED_MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code > "${EXPERIMENT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to become healthy on port $VLLM_PORT..."
READY=0
for i in $(seq 1 180); do
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✓ vLLM server is healthy and ready to serve requests!"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process exited unexpectedly. Log snippet:"
        tail -n 25 "${EXPERIMENT_DIR}/vllm.log"
        exit 1
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Timed out waiting for vLLM."
    exit 1
fi

# 6. Execute OASIS Baseline Simulation inside container
echo ""
echo "===================================================================="
echo "Executing OASIS Baseline Simulation (Organic Non-CIB) inside Container..."
echo "===================================================================="

$CONTAINER_RUN python3 scratch/run_baseline_vllm.py \
    --steps 5 \
    --db "$DB_PATH" \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --model "$RESOLVED_MODEL" \
    --ratio 0.4

echo "✓ Simulation completed. Database written to $DB_PATH"

# 7. Extract Telemetry and Export Metrics inside container
echo ""
echo "===================================================================="
echo "Extracting Telemetry, Engagement, and KPIs inside Container..."
echo "===================================================================="

$CONTAINER_RUN python3 scratch/extract_simulation_data.py \
    --db "$DB_PATH" \
    --out "$EXPERIMENT_DIR"

# Copy final database into experiment bundle
cp "$DB_PATH" "${EXPERIMENT_DIR}/baseline_simulation.db"

echo ""
echo "===================================================================="
echo "EXPERIMENT BUNDLE READY: $EXPERIMENT_DIR"
echo "Files generated:"
ls -lh "$EXPERIMENT_DIR"
echo "===================================================================="
