#!/bin/bash

# Evaluate VideoGS compression pipeline (combined compress + decompress)
#
# Usage: evaluate_videogs_compression.sh [OPTIONS]
#   --dataset_name     Dataset name             (default: HiFi4G_Dataset)
#   --sequence_name    Sequence name            (default: 4K_Actor1_Greeting)
#   --resolution       Resolution scale         (default: 2)
#   --frame_start      Start frame              (default: 0)
#   --frame_end        End frame                (default: 200)
#   --interval         Frame interval           (default: 1)
#   --group_size       Group size for H.264     (default: 20)
#   --qp               H.264 QP                 (default: 25)

DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
RESOLUTION=2

START_FRAME=0
END_FRAME=200
INTERVAL=1
SH_DEGREE=3
GROUP_SIZE=20

# H.264 QP (0=lossless, 51=worst)
QP=25

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)    DATASET_NAME="$2";    shift 2 ;;
        --sequence_name)   SEQUENCE_NAME="$2";   shift 2 ;;
        --resolution)      RESOLUTION="$2";      shift 2 ;;
        --frame_start)     START_FRAME="$2";     shift 2 ;;
        --frame_end)       END_FRAME="$2";       shift 2 ;;
        --interval)        INTERVAL="$2";        shift 2 ;;
        --group_size)      GROUP_SIZE="$2";      shift 2 ;;
        --qp)              QP="$2";              shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/synology/rajrup/VideoGS"
dataset_path="${data_path}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"
gt_model_path="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"

output_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/videogs/qp_${QP}"

VIDEOGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

### 1. VideoGS Compress + Decompress (combined pipeline)
echo "======================================================================"
echo "Step 1: VideoGS Compress + Decompress (combined pipeline)"
echo "======================================================================"
echo "  Dataset:      ${dataset_path}"
echo "  GT model:     ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "  QP:           ${QP}"
echo "======================================================================"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate videogs
cd "${VIDEOGS_ROOT}"

python scripts/videogs_baseline/compress_decompress_pipeline.py \
    --ply_path "${gt_model_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --group_size ${GROUP_SIZE} \
    --sh_degree ${SH_DEGREE} \
    --qp ${QP}

### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"

python scripts/evaluate_decompress.py \
    --gt_ply_path "${gt_model_path}" \
    --decompressed_ply_path "${output_folder}/decompressed_ply" \
    --dataset_path "${dataset_path}" \
    --output_render_path "${output_folder}/evaluation" \
    --save_renders \
    --sh_degree ${SH_DEGREE} \
    --resolution ${RESOLUTION} \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL}

echo ""
echo "======================================================================"
echo "Done! Results in: ${output_folder}"
echo "======================================================================"
