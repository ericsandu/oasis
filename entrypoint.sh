#!/bin/bash
set -e

echo "=== [OASIS Container Initializing] ==="
mkdir -p /app/data

# 1. Boot Gorse in background (master, server, worker combined)
echo "Starting Gorse recommender engine in background..."
gorse-in-one -c /app/gorse_config.toml > /app/data/gorse.log 2>&1 &
GORSE_PID=$!
echo "Gorse PID: $GORSE_PID (listening on :8088)"

# 2. Boot vLLM central node if GPU is available or explicitly requested
START_VLLM="${START_VLLM:-auto}"
if [ "$START_VLLM" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        START_VLLM="1"
    else
        START_VLLM="0"
    fi
fi

if [ "$START_VLLM" = "1" ]; then
    echo "GPU detected. Starting vLLM central node in background..."
    MODEL_NAME="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8}"
    VLLM_PORT="${VLLM_PORT:-8000}"
    MAX_LEN="${VLLM_MAX_LEN:-4096}"
    GPU_MEM="${VLLM_GPU_MEM:-0.85}"
    
    TOOL_PARSER="${VLLM_TOOL_PARSER:-hermes}"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "vLLM Model: $MODEL_NAME | Port: $VLLM_PORT | GPU Mem: $GPU_MEM | MaxLen: $MAX_LEN | Tool Parser: $TOOL_PARSER"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$VLLM_PORT" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM" \
        --enforce-eager \
        --enable-auto-tool-choice \
        --tool-call-parser "$TOOL_PARSER" \
        --trust-remote-code > /app/data/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    echo "Waiting for vLLM central node to initialize on port $VLLM_PORT..."
    READY=0
    for i in $(seq 1 180); do
        if curl -s -f "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM central node is healthy and ready to serve requests!"
            READY=1
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process exited unexpectedly. Log snippet:"
            tail -n 20 /app/data/vllm.log 2>/dev/null || true
            break
        fi
        sleep 2
    done

    if [ "$READY" -ne 1 ]; then
        echo "WARNING: vLLM did not signal ready within timeout. Check /app/data/vllm.log."
    fi
else
    echo "vLLM central node startup skipped (no GPU detected or START_VLLM=0)."
fi

echo "=== [Services Online. Proceeding with OASIS execution] ==="

# Execute the main command (e.g. python script) in the foreground
exec "$@"
