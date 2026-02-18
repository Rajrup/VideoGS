#!/bin/bash

# Plot benchmark results for LiVoGS compression
DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor2_Dancing"
J=15
QUANTIZE_STEP=0.0001
SH_COLOR_SPACE="klt"
CONFIG_NAME="J_${J}_qstep_${QUANTIZE_STEP}_${SH_COLOR_SPACE}"

working_dir="/home/rajrup/Project/VideoGS"
data_path="/synology/rajrup/VideoGS"
input_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/livogs_compression"
plot_script_folder="${working_dir}/scripts/livogs_baseline/plots"

# Plot compressed size + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot encode/decode time (stacked area)
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}

# Plot quality (PSNR / SSIM)
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --j ${J} --qstep ${QUANTIZE_STEP} --sh_color_space ${SH_COLOR_SPACE} --output_folder ${plot_script_folder}
