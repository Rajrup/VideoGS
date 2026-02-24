#!/bin/bash

# Plot benchmark results for videogs_baseline
DATASET_NAME="HiFi4G_Dataset"
SEQUENCE_NAME="4K_Actor1_Greeting"
QP=22

working_dir="/home/rajrup/Project/VideoGS"
data_path="/synology/rajrup/VideoGS"
input_folder="${data_path}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/compression/videogs"
plot_script_folder="${working_dir}/scripts/videogs_baseline/plots"

# Plot compressed size breakdown + point counts
python ${plot_script_folder}/plot_compressed_size.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --qp ${QP} --output_folder ${plot_script_folder}

# Plot compression/decompression time
python ${plot_script_folder}/plot_compression_time.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --qp ${QP} --output_folder ${plot_script_folder}

# Plot quality
python ${plot_script_folder}/plot_quality.py --input_folder ${input_folder} --dataset_name ${DATASET_NAME} --sequence_name ${SEQUENCE_NAME} --qp ${QP} --output_folder ${plot_script_folder}
