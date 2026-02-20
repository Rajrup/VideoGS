#!/bin/bash

conda activate videogs
current_dir=$(pwd)
original_data_dir="/synology/rajrup/Datasets/HiFi4G_Dataset"
processed_data_dir="/synology/rajrup/VideoGS/HiFi4G_Dataset_processed"

echo "Processing HiFi4G Dataset..."

sequence_name="4K_Actor1_Greeting"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name}

sequence_name="4K_Actor2_Dancing"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name}

sequence_name="4K_Actor3_Violin"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name} 

sequence_name="4K_Actor4_Dancing"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name} 

sequence_name="4K_Actor5_Oil-paper_Umbrella"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name} 

sequence_name="4K_Actor6_Changing_Clothes"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name} 

sequence_name="4K_Actor7_Nunchaku"
echo "Processing ${sequence_name}..."
if [ -d "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" ]; then
    mv "${original_data_dir}/${sequence_name}/image_white_undistortion/colmap" "${original_data_dir}/${sequence_name}/"
    mv "${original_data_dir}/${sequence_name}/colmap/sparse/0/"* "${original_data_dir}/${sequence_name}/colmap/sparse/"
    rm -rf "${original_data_dir}/${sequence_name}/colmap/sparse/0/"
fi
python hifi4g_process.py --input ${original_data_dir}/${sequence_name} --output ${processed_data_dir}/${sequence_name} 

cd $current_dir
echo "HiFi4G Dataset processed successfully"