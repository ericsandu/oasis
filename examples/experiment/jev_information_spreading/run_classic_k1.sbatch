#!/bin/bash
# k=1 CLASSIC test: run ONLY the classic leg of align_with_real_world with the
# feed forced to a SINGLE post (classic_align_k1.yaml), to test whether classic's
# reposting comes from SELECTING among many posts. One engine, one GPU -- no
# two-job contention. Compares the k=1 classic cascade to real-world data and to
# the standard k=10 classic DB if present.
#SBATCH --job-name=classic_k1
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_classic_k1_%j.out
#SBATCH --error=slurm_classic_k1_%j.err
set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/eric_sandu/oasis}"
SIF="${CONTAINER_IMAGE:-$HOME/eric_sandu/oasis/examples/experiment/jev_information_spreading/jev_oasis.sif}"
cd "$REPO"
nvidia-smi || true
ALIGN=examples/experiment/twitter_simulation/align_with_real_world
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" VLLM_USE_FLASHINFER_SAMPLER=0
PORT=$(( 19000 + (${SLURM_JOB_ID:-$$} % 4000) ))
MODEL=/models/Meta-Llama-3-8B-Instruct
RUN="apptainer exec --nv --bind $HOME/models:/models --bind $REPO:/app --pwd /app $SIF"

cleanup(){ [ -n "$VP" ] && kill -9 "$VP" 2>/dev/null; pkill -9 -u "$USER" -f vllm.entrypoints 2>/dev/null; }
trap cleanup EXIT

echo "=== start vLLM on $PORT ==="
$RUN python3 -m vllm.entrypoints.openai.api_server --model $MODEL \
  --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --chat-template $ALIGN/tool_chat_template_llama3_min.jinja \
  > vllm_classic_k1.log 2>&1 &
VP=$!
for i in $(seq 1 180); do sleep 2; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM up ($((i*2))s)"; break; }; kill -0 $VP 2>/dev/null || { echo "vLLM died"; tail -40 vllm_classic_k1.log; exit 1; }; done

export JEV_VLLM_URL="http://127.0.0.1:$PORT/v1"
export JEV_VLLM_MODEL="$MODEL"
DRIVER="$ALIGN/twitter_simulation_large.py"

echo "=== CLASSIC k=1 run (single-post feed, use_jev:false) ==="
C0=$(date +%s)
$RUN bash -c "export JEV_VLLM_URL='$JEV_VLLM_URL' JEV_VLLM_MODEL='$JEV_VLLM_MODEL' OASIS_CLASSIC_TOOL_CHOICE='${OASIS_CLASSIC_TOOL_CHOICE:-auto}'; python3 $DRIVER --config_path $ALIGN/classic_align_k1.yaml"
C1=$(date +%s); echo "CLASSIC k=1 wall-clock: $((C1-C0))s"

echo ""
echo "============ ACTION COUNTS (classic k=1) ============"
$RUN python3 -c "
import sqlite3
db='data/simu_db/classic_align_k1_False_Business_0.db'
c=sqlite3.connect(db).cursor()
c.execute(\"SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY COUNT(*) DESC\")
for a,n in c.fetchall(): print(f'{a:12s} {n}')
" || echo "(action-count read failed; DB preserved)"

echo ""
echo "============ CASCADE vs REAL + vs k=10 classic ============"
# Compare k=1 classic to the standard k=10 classic DB if it exists (shows the
# feed-size effect directly), else to real-world data via the parity script.
if [ -f data/simu_db/classic_align_False_Business_0.db ]; then
  $RUN python3 examples/experiment/jev_information_spreading/compare_jev_parity.py \
    data/simu_db/classic_align_False_Business_0.db \
    data/simu_db/classic_align_k1_False_Business_0.db \
    False_Business_0 || echo "(k10-vs-k1 compare failed; DBs preserved)"
else
  echo "(no k=10 classic DB to compare; run run_compare.sbatch first for the k10 baseline)"
fi
echo "=== DONE ==="
