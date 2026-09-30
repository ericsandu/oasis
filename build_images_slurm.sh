#!/usr/bin/env bash
# ==============================================================================
# SLURM Apptainer Image Build Script for UPB Grid & HPC Clusters
# 
# Builds container images on a high-CPU compute node instead of the login node.
# Supports building:
#   1. oasis_base.sif   (CUDA 13.3 + PyTorch + vLLM + Gorse + Poetry)
#   2. hermes.sif       (Standalone lightweight Hermes Agent)
#   3. oasis_hermes.sif (Composite layered simulation + diagnostic sentinel)
#
# Usage:
#   sbatch build_images_slurm.sh               # Builds all images in dependency order
#   sbatch build_images_slurm.sh base          # Builds only oasis_base.sif
#   sbatch build_images_slurm.sh hermes        # Builds only hermes.sif
#   sbatch build_images_slurm.sh composite     # Builds only oasis_hermes.sif (requires base)
#   sbatch --cpus-per-task=32 --mem=64G build_images_slurm.sh   # Extra resources
# ==============================================================================
#SBATCH --job-name=apptainer_build
#SBATCH --partition=grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=build_apptainer_%j.log
#SBATCH --error=build_apptainer_%j.err

set -eo pipefail

# 1. Resolve true project directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    OASIS_DIR="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    OASIS_DIR="$(pwd)"
else
    OASIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CONTAINER_DIR="${OASIS_DIR}/container"
TARGET="${1:-all}"

echo "===================================================================="
echo "Starting Apptainer Image Build on Compute Node: $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-manual}"
echo "CPUs allocated:   ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory allocated: ${SLURM_MEM_PER_NODE:-standard}"
echo "Project Dir:      ${OASIS_DIR}"
echo "Container Dir:    ${CONTAINER_DIR}"
echo "Build Target:     ${TARGET}"
echo "Start Time:       $(date)"
echo "===================================================================="

# 2. Load Apptainer / Singularity modules if environment requires it
if ! command -v apptainer &> /dev/null; then
    module load apptainer 2>/dev/null || module load tengine/apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi

if ! command -v apptainer &> /dev/null; then
    echo "[-] ERROR: apptainer executable not found in PATH on $(hostname)."
    exit 1
fi

echo "[+] Using Apptainer: $(apptainer --version)"

# 3. Configure Scratch & Cache directories to avoid filling /tmp or $HOME quota
# Compute nodes often have small in-memory /tmp (tmpfs) that crashes with "No space left on device"
BUILD_TMPDIR="${SLURM_TMPDIR:-${OASIS_DIR}/.tmp_build_${SLURM_JOB_ID:-manual}}"
mkdir -p "${BUILD_TMPDIR}"
export APPTAINER_TMPDIR="${BUILD_TMPDIR}"
export APPTAINER_CACHEDIR="${BUILD_TMPDIR}/cache"
mkdir -p "${APPTAINER_CACHEDIR}"

cleanup() {
    echo "[*] Cleaning up temporary build directory: ${BUILD_TMPDIR}..."
    rm -rf "${BUILD_TMPDIR}"
}
trap cleanup EXIT

# 4. Ensure container workspace has access to project configuration files
mkdir -p "${CONTAINER_DIR}"
ln -sf "${OASIS_DIR}/pyproject.toml" "${CONTAINER_DIR}/pyproject.toml"
if [ -f "${OASIS_DIR}/poetry.lock" ]; then
    ln -sf "${OASIS_DIR}/poetry.lock" "${CONTAINER_DIR}/poetry.lock"
fi

cd "${CONTAINER_DIR}"

# 5. Build function with timing and error diagnostics
build_image() {
    local def_file="$1"
    local sif_file="$2"
    local image_name="$3"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "[+] Building ${image_name} (${sif_file}) from ${def_file}..."
    echo "    Started at: $(date)"
    echo "--------------------------------------------------------------------"

    local start_ts=$(date +%s)

    # Use --fakeroot for unprivileged user building on HPC compute nodes
    apptainer build --fakeroot --force "${sif_file}" "${def_file}"

    local end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))
    local size=$(du -h "${sif_file}" | cut -f1)

    echo "[✓] SUCCESS: ${sif_file} built in ${duration}s (Size: ${size})"
}

# 6. Build target dispatching
case "${TARGET}" in
    base)
        build_image "oasis_base.def" "oasis_base.sif" "OASIS Base Simulation Image"
        ;;
    hermes)
        build_image "hermes.def" "hermes.sif" "Standalone Hermes Agent Image"
        ;;
    composite)
        if [ ! -f "oasis_base.sif" ]; then
            echo "[-] ERROR: oasis_base.sif must exist in ${CONTAINER_DIR} before building oasis_hermes.sif."
            exit 1
        fi
        build_image "oasis_hermes.def" "oasis_hermes.sif" "Layered OASIS + Hermes Image"
        ;;
    all)
        echo "[*] Building all images in dependency order..."
        build_image "oasis_base.def" "oasis_base.sif" "1/3 OASIS Base Simulation Image"
        build_image "hermes.def" "hermes.sif" "2/3 Standalone Hermes Agent Image"
        build_image "oasis_hermes.def" "oasis_hermes.sif" "3/3 Layered OASIS + Hermes Image"
        ;;
    *)
        echo "[-] Unknown build target: '${TARGET}'. Valid options: all, base, hermes, composite"
        exit 1
        ;;
esac

echo ""
echo "===================================================================="
echo "[✓] All requested container builds completed successfully at $(date)!"
echo "    Images located in: ${CONTAINER_DIR}"
ls -lh "${CONTAINER_DIR}"/*.sif 2>/dev/null || true
echo "===================================================================="
