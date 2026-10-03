#!/bin/bash
# Compare classic-OASIS vs JEV on the align_with_real_world experiment:
# one vLLM server, two runs (use_jev false/true), timed, then parity analysis.
#SBATCH --job-name=jev_compare
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_jev_compare_%j.out
#SBATCH --error=slurm_jev_compare_%j.err
set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/eric_sandu/oasis}"
SIF="${CONTAINER_IMAGE:-$HOME/eric_sandu/oasis/examples/experiment/jev_information_spreading/jev_oasis.sif}"
cd "$REPO"
nvidia-smi || true   # confirm the host sees the GPU before launching vLLM
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
  > vllm_compare.log 2>&1 &
# tool-call parsing: vLLM 0.23 rejects tool_choice="required" unless
# --tool-call-parser is set. llama3_json CRASHES on plain Meta-Llama-3-8B
# (tokenizer lacks the <|python_tag|> token it needs); hermes is JSON/regex
# based and tokenizer-agnostic, so it loads for Llama-3. Needed only by the
# CLASSIC leg; the JEV classifier uses /v1/completions logit-bias and is
# unaffected by these flags.
VP=$!
for i in $(seq 1 180); do sleep 2; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM up ($((i*2))s)"; break; }; kill -0 $VP 2>/dev/null || { echo "vLLM died"; tail -40 vllm_compare.log; exit 1; }; done

export JEV_VLLM_URL="http://127.0.0.1:$PORT/v1"
export JEV_VLLM_MODEL="$MODEL"
DRIVER="$ALIGN/twitter_simulation_large.py"

echo "=== CLASSIC run (use_jev:false) ==="
C0=$(date +%s)
$RUN bash -c "export JEV_VLLM_URL='$JEV_VLLM_URL' JEV_VLLM_MODEL='$JEV_VLLM_MODEL'; python3 $DRIVER --config_path $ALIGN/classic_align.yaml"
C1=$(date +%s); echo "CLASSIC wall-clock: $((C1-C0))s"

echo "=== JEV run (use_jev:true) ==="
J0=$(date +%s)
$RUN bash -c "export JEV_VLLM_URL='$JEV_VLLM_URL' JEV_VLLM_MODEL='$JEV_VLLM_MODEL' OASIS_USE_JEV=1; python3 $DRIVER --config_path $ALIGN/jev_align.yaml"
J1=$(date +%s); echo "JEV wall-clock: $((J1-J0))s"

echo ""
echo "============ PERFORMANCE ============"
echo "CLASSIC: $((C1-C0))s   JEV: $((J1-J0))s   speedup: $(python3 -c "print(f'{($C1-$C0)/max(1,($J1-$J0)):.2f}x')")"
echo ""
echo "============ PARITY / SIMILARITY ============"
$RUN python3 examples/experiment/jev_information_spreading/compare_jev_parity.py \
  data/simu_db/classic_align_False_Business_0.db \
  data/simu_db/jev_align_False_Business_0.db \
  False_Business_0 || echo "(analysis failed; DBs are preserved for manual comparison)"
echo "=== DONE ==="
