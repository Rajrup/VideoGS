#!/bin/bash

# Evaluate MesonGS compression pipeline for VideoGS-trained models
#
# Usage: evaluate_mesongs_compression.sh [OPTIONS]
#   --dataset_name     Dataset name           (default: HiFi4G_Dataset)
#   --sequence_name    Sequence name          (default: 4K_Actor1_Greeting)
#   --resolution       Resolution scale       (default: 2)
#   --frame_start      Start frame            (default: 0)
#   --frame_end        End frame              (default: 200)
#   --interval         Frame interval         (default: 1)
#   --depth            Octree depth           (default: from config)
#   --n_block          Block quant count      (default: from config)
#   --codebook_size    VQ codebook size       (default: from config)
#   --prune            Enable pruning         (flag)

DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
RESOLUTION=2

START_FRAME=0
END_FRAME=200
INTERVAL=10
SH_DEGREE=3

# --- Parse named arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_name)    DATASET_NAME="$2";    shift 2 ;;
        --sequence_name)   SEQUENCE_NAME="$2";   shift 2 ;;
        --resolution)      RESOLUTION="$2";      shift 2 ;;
        --frame_start)     START_FRAME="$2";     shift 2 ;;
        --frame_end)       END_FRAME="$2";       shift 2 ;;
        --interval)        INTERVAL="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

data_path="/synology/rajrup/VideoGS"
dataset_path="${data_path}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"
gt_model_path="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"

# Build output folder name from parameters
output_tag="params_default"
output_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/mesongs/${output_tag}"

VIDEOGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MESONGS_ROOT="${VIDEOGS_ROOT}/MesonGS"

### 1. MesonGS Compress + Decompress (in mesongs conda env, from MesonGS dir)
echo "======================================================================"
echo "Step 1: MesonGS Compress + Decompress"
echo "======================================================================"
echo "  Dataset:      ${dataset_path}"
echo "  GT model:     ${gt_model_path}"
echo "  Output:       ${output_folder}"
echo "  Scene:        ${SEQUENCE_NAME}"
echo "======================================================================"

cd "${MESONGS_ROOT}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate mesongs

python "${VIDEOGS_ROOT}/scripts/mesongs_baseline/compress_decompress_pipeline.py" \
    --ply_path "${gt_model_path}" \
    --dataset_path "${dataset_path}" \
    --output_folder "${output_folder}" \
    --output_ply_folder "${output_folder}/decompressed_ply" \
    --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL} \
    --sh_degree ${SH_DEGREE} \
    --resolution ${RESOLUTION} \
    --scene_name "${SEQUENCE_NAME}"

### 2. Evaluate Decompression Quality (PSNR/SSIM vs GT)
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"

conda activate videogs
cd "${VIDEOGS_ROOT}"

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
