#!/bin/bash
# Rigorous timing: classic vs JEV on align_with_real_world (Information
# Spreading), 5 runs each, ONE vLLM server shared. Measures SIM-ONLY wall clock
# (the driver's SIM_WALL_SECONDS marker brackets steps 1..N, excluding imports /
# recsys-model load / agent-graph gen), then reports mean/std/speedup.
#SBATCH --job-name=jev_bench5
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm_jev_bench_%j.out
#SBATCH --error=slurm_jev_bench_%j.err
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
NRUNS="${NRUNS:-5}"

cleanup(){ [ -n "$VP" ] && kill -9 "$VP" 2>/dev/null; pkill -9 -u "$USER" -f vllm.entrypoints 2>/dev/null; }
trap cleanup EXIT

echo "=== start vLLM on $PORT ==="
$RUN python3 -m vllm.entrypoints.openai.api_server --model $MODEL \
  --served-model-name $MODEL --host 127.0.0.1 --port $PORT \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --chat-template $ALIGN/tool_chat_template_llama3_min.jinja \
  > vllm_bench.log 2>&1 &
VP=$!
for i in $(seq 1 180); do sleep 2; curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM up ($((i*2))s)"; break; }; kill -0 $VP 2>/dev/null || { echo "vLLM died"; tail -40 vllm_bench.log; exit 1; }; done

export JEV_VLLM_URL="http://127.0.0.1:$PORT/v1"
export JEV_VLLM_MODEL="$MODEL"
DRIVER="$ALIGN/twitter_simulation_large.py"

CLASSIC_TIMES=""
JEV_TIMES=""

run_one(){  # $1=engine(classic|jev) $2=config $3=extra_env
  local eng="$1" cfg="$2" extra="$3" out
  out=$($RUN bash -c "export JEV_VLLM_URL='$JEV_VLLM_URL' JEV_VLLM_MODEL='$JEV_VLLM_MODEL' $extra; python3 $DRIVER --config_path $cfg" 2>&1)
  echo "$out" | grep -E "SIM_WALL_SECONDS=" | tail -1
}

echo "=== CLASSIC x$NRUNS (tool_choice=auto) ==="
for r in $(seq 1 $NRUNS); do
  line=$(run_one classic "$ALIGN/classic_align.yaml" "OASIS_CLASSIC_TOOL_CHOICE=auto")
  t=$(echo "$line" | sed -E 's/.*SIM_WALL_SECONDS=([0-9.]+).*/\1/')
  echo "classic run $r: ${t}s"
  CLASSIC_TIMES="$CLASSIC_TIMES $t"
done

JEV_EXTRA_FLAGS="OASIS_USE_JEV=1 OASIS_JEV_FEED_MODE='${OASIS_JEV_FEED_MODE:-0}' OASIS_JEV_VERBATIM='${OASIS_JEV_VERBATIM:-0}' OASIS_JEV_COMPETITIVE='${OASIS_JEV_COMPETITIVE:-0}' OASIS_JEV_GUIDED_CHOICE='${OASIS_JEV_GUIDED_CHOICE:-0}' OASIS_JEV_GENERATIVE='${OASIS_JEV_GENERATIVE:-0}' OASIS_JEV_INSTRUCT_FRAME='${OASIS_JEV_INSTRUCT_FRAME:-0}' OASIS_JEV_L0='${OASIS_JEV_L0:-0}' OASIS_JEV_L0_STRENGTH='${OASIS_JEV_L0_STRENGTH:-1.0}' OASIS_JEV_L0_GROUP='${OASIS_JEV_L0_GROUP:-0}' OASIS_JEV_CONF_DUMP='${OASIS_JEV_CONF_DUMP:-}'"

echo "=== JEV x$NRUNS ==="
for r in $(seq 1 $NRUNS); do
  line=$(run_one jev "$ALIGN/jev_align.yaml" "$JEV_EXTRA_FLAGS")
  t=$(echo "$line" | sed -E 's/.*SIM_WALL_SECONDS=([0-9.]+).*/\1/')
  echo "jev run $r: ${t}s"
  JEV_TIMES="$JEV_TIMES $t"
done

echo ""
echo "============ STATISTICS (sim-only, steps 1..N) ============"
$RUN python3 - "$CLASSIC_TIMES" "$JEV_TIMES" <<'PY'
import sys, statistics as st
classic = [float(x) for x in sys.argv[1].split()]
jev     = [float(x) for x in sys.argv[2].split()]
def rep(name, xs):
    if not xs:
        print(f"{name}: NO DATA"); return None
    m = st.mean(xs); s = st.pstdev(xs) if len(xs) > 1 else 0.0
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    print(f"{name}: n={len(xs)} mean={m:.2f}s  sample_std={sd:.2f}s  min={min(xs):.2f}  max={max(xs):.2f}")
    print(f"     raw: {['%.2f'%x for x in xs]}")
    return m
cm = rep("CLASSIC", classic)
jm = rep("JEV    ", jev)
if cm and jm and jm > 0:
    print(f"\nSPEEDUP (classic_mean / jev_mean) = {cm/jm:.2f}x")
    # crude 95% CI on the ratio via mean+-1.96*se if n>1
    import math
    if len(classic)>1 and len(jev)>1:
        def se(xs): return st.stdev(xs)/math.sqrt(len(xs))
        print(f"CLASSIC 95% CI ~ {cm-1.96*se(classic):.2f}..{cm+1.96*se(classic):.2f}s")
        print(f"JEV     95% CI ~ {jm-1.96*se(jev):.2f}..{jm+1.96*se(jev):.2f}s")
PY
echo "=== DONE ==="
