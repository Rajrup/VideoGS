#!/bin/bash

# Plot benchmark results for VideoGS compression
DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"

# H.264 QP (0=lossless, 51=worst)
QP=25

VIDEOGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_path="/synology/rajrup/VideoGS"
input_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/videogs"
plot_script_folder="${VIDEOGS_ROOT}/scripts/videogs_baseline/plots"

QP_ARGS="--qp ${QP}"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}

# Plot compression/decompression time
python ${plot_script_folder}/plot_compression_time.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}

# Plot quality
python ${plot_script_folder}/plot_quality.py \
    --input_folder ${input_folder} \
    --dataset_name ${DATASET_NAME} \
    --sequence_name ${SEQUENCE_NAME} \
    ${QP_ARGS} \
    --output_folder ${plot_script_folder}
