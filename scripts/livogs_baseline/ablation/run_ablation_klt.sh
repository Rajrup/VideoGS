#!/bin/bash
# KLT Color Space Ablation for HiFi4G (VideoGS) sequences.
#
# Compares three color space transforms (KLT, YUV, RGB) on:
#   - 4K_Actor1_Greeting (frame 0)
#
# Usage:
#   bash scripts/livogs_baseline/ablation/run_ablation_klt.sh [OPTIONS]
#     --skip_eval   Skip quality evaluation step
#     --no_save_ply Skip saving decompressed PLYs

set -euo pipefail

VIDEOGS_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

SKIP_EVAL=false
SAVE_PLY=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_eval)   SKIP_EVAL=true;    shift ;;
        --no_save_ply) SAVE_PLY=false;    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SAVE_PLY_FLAG=""
if [ "${SAVE_PLY}" = true ]; then
    SAVE_PLY_FLAG="--save_ply"
fi

DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
DATA_ROOT="/synology/rajrup/VideoGS"
HIFI4G_J=12
HIFI4G_SH_DEGREE=3
HIFI4G_FRAME_ID=0
QUANTIZE_STEP=0.0001

PLY_PATH="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"
OUTPUT_FOLDER="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/ablation/livogs_klt"

# ============================================================================
# Step 1: LiVoGS Compression (all 3 color space variants)
# ============================================================================
echo "======================================================================"
echo "KLT Ablation: ${SEQUENCE_NAME} (HiFi4G)"
echo "  Frame: ${HIFI4G_FRAME_ID}, J=${HIFI4G_J}, QP=${QUANTIZE_STEP}"
echo "  Output: ${OUTPUT_FOLDER}"
echo "======================================================================"

python "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/ablation_klt.py" \
    --ply_path "${PLY_PATH}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --format videogs \
    --frame_ids "${HIFI4G_FRAME_ID}" \
    --sh_degree ${HIFI4G_SH_DEGREE} --J ${HIFI4G_J} \
    --quantize_step ${QUANTIZE_STEP} \
    --nvcomp_algorithm ANS \
    ${SAVE_PLY_FLAG}

# ============================================================================
# Step 2: Quality Evaluation (VideoGS evaluate_decompress.py)
# ============================================================================
if [ "${SKIP_EVAL}" = false ] && [ "${SAVE_PLY}" = true ]; then
    DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"

    for VARIANT in klt yuv rgb; do
        echo ""
        echo "Evaluating ${VARIANT} variant for ${SEQUENCE_NAME}..."
        python "${VIDEOGS_ROOT}/scripts/evaluate_decompress.py" \
            --gt_ply_path "${PLY_PATH}" \
            --decompressed_ply_path "${OUTPUT_FOLDER}/${VARIANT}/decompressed_ply" \
            --dataset_path "${DATASET_PATH}" \
            --output_render_path "${OUTPUT_FOLDER}/${VARIANT}/evaluation" \
            --save_renders \
            --sh_degree ${HIFI4G_SH_DEGREE} \
            --resolution 2 \
            --frame_start ${HIFI4G_FRAME_ID} --frame_end $((HIFI4G_FRAME_ID + 1)) --interval 1
    done
fi

# ============================================================================
# Generate plots
# ============================================================================
echo ""
echo "======================================================================"
echo "Generating comparison plots..."
echo "======================================================================"
python "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/plot_ablation_klt.py" \
    --output_folder "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/plots" \
    --format png

echo ""
echo "======================================================================"
echo "Done! Results in: ${OUTPUT_FOLDER}"
echo "======================================================================"
