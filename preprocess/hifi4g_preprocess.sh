#!/bin/bash

conda activate videogs
current_dir=$(pwd)
original_data_dir="/synology/rajrup/Datasets/HiFi4G_Dataset"
processed_data_dir="/synology/rajrup/VideoGS/HiFi4G_Dataset_processed"

sequences=(
    "4K_Actor1_Greeting"
    "4K_Actor2_Dancing"
    "4K_Actor3_Violin"
    "4K_Actor4_Dancing"
    "4K_Actor5_Oil-paper_Umbrella"
    "4K_Actor6_Changing_Clothes"
    "4K_Actor7_Nunchaku"
)

echo "Processing HiFi4G Dataset..."

for sequence_name in "${sequences[@]}"; do
    echo "Processing ${sequence_name}..."
    if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
        mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
        mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
        rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
    fi
    python hifi4g_process.py --input "${original_data_dir}/${sequence_name}" --output "${processed_data_dir}/${sequence_name}"
done

cd "$current_dir"
echo "HiFi4G Dataset processed successfully"
