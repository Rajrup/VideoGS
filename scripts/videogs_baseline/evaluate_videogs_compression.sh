#!/bin/bash

# Evaluate VideoGS compression pipeline for baseline
DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
RESOLUTION=2

# VideoGS compression parameters
START_FRAME=0
END_FRAME=200
GROUP_SIZE=20
INTERVAL=1
SH_DEGREE=3
QP=22

data_path="/synology/rajrup/VideoGS"
dataset_path="${data_path}/${DATASET_NAME}_processed/${SEQUENCE_NAME}"
gt_model_path="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"
output_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/videogs/qp_${QP}"

echo "======================================================================"
echo "Step 1: VideoGS Compress + Decompress"
echo "======================================================================"
### 1. Compress PLY to PNG (Full SH)
echo "Compressing PLY to PNG (Full SH)..."
python scripts/videogs_baseline/compress_to_png_full_sh.py --frame_start ${START_FRAME} --frame_end ${END_FRAME} --group_size ${GROUP_SIZE} --interval ${INTERVAL} --ply_path ${gt_model_path} --output_folder "${output_folder}/compressed_png" --sh_degree ${SH_DEGREE}

### 2. Compress PNG to MP4 (H.264)
echo "Compressing PNG to MP4 (H.264)..."
python scripts/videogs_baseline/compress_png_2_video.py --input_folder "${output_folder}/compressed_png" --output_folder "${output_folder}/compressed_video" --qp ${QP} --sh_degree ${SH_DEGREE}

### 3. Decompress MP4 to PNG
echo "Decompressing MP4 to PNG..."
python scripts/videogs_baseline/decompress_video_2_png.py --input_folder "${output_folder}/compressed_video" --output_folder "${output_folder}/decompressed_png"

### 4. Decompress PNG to PLY (Full SH)
echo "Decompressing PNG to PLY (Full SH)..."
python scripts/videogs_baseline/decompress_from_png_full_sh.py --compressed_folder "${output_folder}/decompressed_png" --output_ply_folder "${output_folder}/decompressed_ply" --sh_degree ${SH_DEGREE}

### 5. Evaluate Decompression Quality
echo ""
echo "======================================================================"
echo "Step 2: Evaluate Decompression Quality"
echo "======================================================================"
python scripts/evaluate_decompress.py --gt_ply_path ${gt_model_path} --decompressed_ply_path "${output_folder}/decompressed_ply" --dataset_path ${dataset_path} --output_render_path "${output_folder}/evaluation" --save_renders --sh_degree ${SH_DEGREE} --resolution ${RESOLUTION} --frame_start ${START_FRAME} --frame_end ${END_FRAME} --interval ${INTERVAL}

echo "Done!"