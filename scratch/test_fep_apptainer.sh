#!/bin/bash
#SBATCH --job-name=test_apptainer_gpu
#SBATCH --account=phd
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_apptainer_test_%j.out
#SBATCH --error=slurm_apptainer_test_%j.err

# ==============================================================================
# Quick Sanity Test for Apptainer Container & GPU Passthrough on fep
# ==============================================================================

set -e

SIF_PATH="${1:-$HOME/eric_sandu/oasis.sif}"
if [ ! -f "$SIF_PATH" ]; then
    # Fallback checks within $HOME/eric_sandu/ and $HOME/
    if [ -f "$HOME/eric_sandu/pytorch.sif" ]; then
        SIF_PATH="$HOME/eric_sandu/pytorch.sif"
    elif [ -f "$HOME/oasis.sif" ]; then
        SIF_PATH="$HOME/oasis.sif"
    elif [ -f "$HOME/pytorch.sif" ]; then
        SIF_PATH="$HOME/pytorch.sif"
    fi
fi

echo "===================================================================="
echo "Testing Apptainer Infrastructure on Node: $(hostname)"
echo "Container: $SIF_PATH"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "===================================================================="

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Container file $SIF_PATH not found!"
    echo "Build it first using: apptainer build \$HOME/oasis.sif oasis.def"
    exit 1
fi

echo "1. Testing GPU Passthrough and NVIDIA-SMI inside container:"
apptainer exec --nv "$SIF_PATH" nvidia-smi

echo ""
echo "2. Testing PyTorch CUDA Tensor Allocation inside container:"
apptainer exec --nv "$SIF_PATH" python3 -c "
import torch
print('✓ PyTorch Version:    ', torch.__version__)
print('✓ CUDA Available:     ', torch.cuda.is_available())
print('✓ Device Name:        ', torch.cuda.get_device_name(0))
print('✓ Total VRAM (GB):    ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

# Matrix multiplication test
x = torch.randn(4096, 4096, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('✓ GPU Matrix Multiplication: SUCCESS')
"

echo ""
echo "3. Testing Gorse binary and vLLM module inside container:"
apptainer exec --nv "$SIF_PATH" bash -c "
if command -v gorse-in-one &> /dev/null; then
    echo '✓ Gorse Recommender Binary: FOUND at ' \$(which gorse-in-one)
else
    echo '⚠ Gorse binary not installed in this container image.'
fi

python3 -c '
try:
    import vllm
    print(\"✓ vLLM Module: FOUND (v\" + vllm.__version__ + \")\")
except ImportError:
    print(\"⚠ vLLM not installed in this container image.\")
try:
    import camel
    print(\"✓ CAMEL-AI Module: FOUND (v\" + camel.__version__ + \")\")
except ImportError:
    print(\"⚠ CAMEL-AI not installed in this container image.\")
'
"

echo ""
echo "4. Testing Local Model Mount Access ($HOME/models):"
apptainer exec --nv --bind "$HOME/models":/models "$SIF_PATH" bash -c "
if [ -d '/models' ]; then
    echo '✓ Successfully mounted /models inside container. Available models:'
    ls -lh /models
else
    echo '⚠ /models directory not accessible.'
fi
"

echo ""
echo "===================================================================="
echo "ALL APPTAINER SANITY CHECKS COMPLETED!"
echo "===================================================================="
