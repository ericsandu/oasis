#!/bin/bash
# ==============================================================================
# SLURM + Apptainer runner: JEV "Information Spreading in X" (OASIS paper F.2.2)
# Target: UPB Grid (fep8.grid.pub.ro), 1x NVIDIA A100-SXM4-80GB.
#
# Modeled on the P0 batch runners (launch_fep_batch.sh / scratch/run_fep_baseline.sh):
#   - mounts $HOME/models -> /models and resolves pre-downloaded weights
#   - offline HF/vLLM env for air-gapped grid nodes
#   - SIF auto-discovery, isolated vLLM port, health wait, cleanup trap
#
# Differs from the P0 CIB matrix: this reproduces ONE paper baseline
# (Information Spreading), uses stock OASIS twhin-bert recsys (NOT Gorse), and
# drives the sim with the JEV engine via jev_information_spreading.py.
#
# Submit from the repository root:
#     sbatch examples/experiment/jev_information_spreading/run_jev_hpc.sbatch
# Overrides:
#     MODEL_PATH=/models/Qwen2.5-14B-Instruct \
#     CONTAINER_IMAGE=$HOME/oasis.sif \
#     NUM_TIMESTEPS=50 \
#     sbatch .../run_jev_hpc.sbatch
# ==============================================================================
#SBATCH --job-name=jev_infospread
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_jev_infospread_%j.out
#SBATCH --error=slurm_jev_infospread_%j.err

set -eo pipefail

# Cap thread utilization across math runtime libraries (P0 convention).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ---------------------------------------------------------------------------
# Resolve OASIS project root on host (prefer sbatch submit dir).
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    OASIS_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    OASIS_DIR="$(pwd)"
else
    OASIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "${OASIS_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="${OASIS_DIR}/experiments/jev_infospread_${TIMESTAMP}"
DATA_DIR="${OASIS_DIR}/data/simu_db"
mkdir -p "${EXP_DIR}" "${DATA_DIR}" "${OASIS_DIR}/experiments"
DB_PATH="${EXP_DIR}/jev_information_spreading.db"
CONFIG="${CONFIG:-examples/experiment/jev_information_spreading/jev_information_spreading.yaml}"

echo "===================================================================="
echo "JEV Information-Spreading (OASIS F.2.2) on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual} | Partition: ${SLURM_JOB_PARTITION:-local}"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES:-0} | Project: ${OASIS_DIR}"
echo "Output: ${EXP_DIR}"
echo "===================================================================="
nvidia-smi || true

# ---------------------------------------------------------------------------
# 1. Locate Apptainer SIF image (P0 candidate convention).
# ---------------------------------------------------------------------------
SIF_CANDIDATES=(
    "${CONTAINER_IMAGE:-}"
    "${OASIS_DIR}/examples/experiment/jev_information_spreading/jev_oasis.sif"
    "${OASIS_DIR}/container/oasis_hermes.sif"
    "${OASIS_DIR}/container/oasis_base.sif"
    "$HOME/eric_sandu/oasis.sif"
    "$HOME/oasis.sif"
    "$HOME/camel-oasis.sif"
    "$HOME/pytorch.sif"
    "${OASIS_DIR}/oasis.sif"
)
CONTAINER_SIF=""
for s in "${SIF_CANDIDATES[@]}"; do
    if [ -n "$s" ] && [ -f "$s" ]; then CONTAINER_SIF="$s"; break; fi
done

if [ -n "$CONTAINER_SIF" ] && command -v apptainer &> /dev/null; then
    echo "✓ Using Apptainer container: ${CONTAINER_SIF}"
    CONTAINER_RUN="apptainer exec --nv \
        --bind $HOME/models:/models \
        --bind ${OASIS_DIR}:/app \
        --bind ${OASIS_DIR}:/workspace \
        --bind ${DATA_DIR}:/app/data/simu_db \
        --bind ${EXP_DIR}:/app/experiments/current \
        --pwd /app \
        ${CONTAINER_SIF}"
    PY_EXEC="${CONTAINER_RUN} python3"
else
    echo "Notice: no Apptainer SIF found; falling back to host Poetry env."
    CONTAINER_RUN=""
    PY_EXEC="poetry run python"
fi

# ---------------------------------------------------------------------------
# 2. Resolve model: prefer a pre-downloaded dir under $HOME/models (-> /models).
#    Paper baseline (OASIS App. C.4 / line 479): Llama-3-8B-Instruct is the
#    agent backend for the real-world information-propagation reproduction.
#    If not present locally, download it into $HOME/models ONCE and reuse it.
# ---------------------------------------------------------------------------
# HF repo id of the paper model, and the local folder name we cache it under.
PAPER_MODEL_REPO="${PAPER_MODEL_REPO:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-$(basename "$PAPER_MODEL_REPO")}"
HOST_MODELS_DIR="$HOME/models"
mkdir -p "$HOST_MODELS_DIR"

MODEL_CANDIDATES=(
    "${MODEL_PATH:-}"
    "/models/${MODEL_LOCAL_NAME}"
    "${HOST_MODELS_DIR}/${MODEL_LOCAL_NAME}"
    "/models/Meta-Llama-3-8B-Instruct"
    "$HOME/models/Meta-Llama-3-8B-Instruct"
)
RESOLVED_MODEL=""
for m in "${MODEL_CANDIDATES[@]}"; do
    if [ -n "$m" ]; then
        if [ -d "$m" ] || [ -f "$m" ]; then RESOLVED_MODEL="$m"; break; fi
        if [ -n "$CONTAINER_RUN" ] && $CONTAINER_RUN test -d "$m" 2>/dev/null; then
            RESOLVED_MODEL="$m"; break
        fi
    fi
done

# Not found locally: download the paper model into $HOME/models for reuse.
# NOTE: Meta-Llama-3 is a GATED HF repo -- export HF_TOKEN (or run `hf auth
# login`) on an ONLINE node before first use. Subsequent runs reuse the cached
# weights and run fully offline.
if [ -z "$RESOLVED_MODEL" ]; then
    DEST="${HOST_MODELS_DIR}/${MODEL_LOCAL_NAME}"
    echo "Model not found locally. Downloading ${PAPER_MODEL_REPO} -> ${DEST} (first run only) ..."
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARNING: HF_TOKEN not set. ${PAPER_MODEL_REPO} is gated; the download"
        echo "         will fail unless you have already run 'hf auth login'."
    fi
    # huggingface-cli was renamed to `hf` (huggingface_hub >= 1.x); the old
    # command is deprecated. Prefer `hf`, fall back to huggingface-cli for
    # older images. Note: --local-dir-use-symlinks was REMOVED (local-dir is
    # always a real copy now), so it is not passed.
    build_dl_cmd() {
        # $1 = cli binary name
        echo "$1 download '${PAPER_MODEL_REPO}' --local-dir '${DEST}' --exclude '*.pth' 'original/*'"
    }
    run_download() {
        # Runs inside the container when CONTAINER_RUN is set, else on host.
        local inner="
            export HF_TOKEN='${HF_TOKEN:-}'
            unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
            if command -v hf >/dev/null 2>&1; then
                $(build_dl_cmd hf)
            elif command -v huggingface-cli >/dev/null 2>&1; then
                $(build_dl_cmd huggingface-cli)
            else
                echo 'ERROR: neither hf nor huggingface-cli is available.' >&2
                exit 1
            fi
        "
        if [ -n "$CONTAINER_RUN" ]; then
            $CONTAINER_RUN bash -c "$inner"
        else
            bash -c "$inner"
        fi
    }
    run_download
    if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
        RESOLVED_MODEL="/models/${MODEL_LOCAL_NAME}"   # path as seen inside container
        [ -z "$CONTAINER_RUN" ] && RESOLVED_MODEL="$DEST"
        echo "✓ Downloaded and cached at ${DEST}"
    else
        echo "ERROR: download did not produce weights at ${DEST}."; exit 1
    fi
fi
echo "✓ Using LLM model: ${RESOLVED_MODEL}"

# ---------------------------------------------------------------------------
# 3. Cleanup hook (kill vLLM + any orphans owned by this user).
# ---------------------------------------------------------------------------
VLLM_PID=""
cleanup() {
    echo "Shutting down vLLM (pid ${VLLM_PID:-none}) ..."
    [ -n "$VLLM_PID" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
    pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM
pkill -9 -u "$USER" -f "vllm.entrypoints" 2>/dev/null || true
sleep 1

# ---------------------------------------------------------------------------
# 4. Launch vLLM server on an isolated port (offline-safe env).
# ---------------------------------------------------------------------------
JOB_SEED=${SLURM_JOB_ID:-$$}
VLLM_PORT=$(( 18000 + (JOB_SEED % 4000) ))
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Starting vLLM on port ${VLLM_PORT} for ${RESOLVED_MODEL} ..."
$CONTAINER_RUN bash -c "
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export VLLM_USE_FLASHINFER_SAMPLER=0
exec python3 -m vllm.entrypoints.openai.api_server \
    --model '${RESOLVED_MODEL}' \
    --served-model-name '${RESOLVED_MODEL}' \
    --host 127.0.0.1 \
    --port '${VLLM_PORT}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --gpu-memory-utilization '${GPU_MEM_UTIL}' \
    --enforce-eager \
    --trust-remote-code
" > "${EXP_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

# ---------------------------------------------------------------------------
# 5. Wait for vLLM health (up to 10 min for weight load).
# ---------------------------------------------------------------------------
echo "Waiting for vLLM /health on port ${VLLM_PORT} (pid ${VLLM_PID}) ..."
READY=0
for i in $(seq 1 300); do
    sleep 2
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup. Log tail:"
        tail -n 120 "${EXP_DIR}/vllm.log" 2>/dev/null || true
        exit 1
    fi
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✓ vLLM healthy after $((i*2))s."
        READY=1; break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  [$(date +%T)] waiting (${i}/300): $(tail -n1 "${EXP_DIR}/vllm.log" 2>/dev/null)"
    fi
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not become healthy in time."; tail -n 120 "${EXP_DIR}/vllm.log" || true; exit 1
fi

# ---------------------------------------------------------------------------
# 6. Run the JEV Information-Spreading simulation against the local server.
# ---------------------------------------------------------------------------
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
echo "Running JEV driver (num_timesteps=${NUM_TIMESTEPS}) ..."
$CONTAINER_RUN bash -c "
export VLLM_BASE_URL='http://127.0.0.1:${VLLM_PORT}/v1'
export VLLM_MODEL='${RESOLVED_MODEL}'
exec python3 examples/experiment/jev_information_spreading/jev_information_spreading.py \
    --config_path '${CONFIG}'
"

echo "===================================================================="
echo "JEV Information-Spreading complete."
echo "DB: ${DB_PATH}   (also default from YAML: data/simu_db/)"
echo "Artifacts: ${EXP_DIR}  (vllm.log + slurm_jev_infospread_${SLURM_JOB_ID:-manual}.{out,err})"
ls -lh "${EXP_DIR}" || true
echo "===================================================================="
