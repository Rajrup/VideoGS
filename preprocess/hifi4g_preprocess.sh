#!/bin/bash

conda activate videogs
current_dir=$(pwd)

move_option="False"

echo "Processing HiFi4G Dataset..."
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor1_Greeting --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor1_Greeting --move $move_option                      # Done
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor2_Dancing --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing --move $move_option                        # Done
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor3_Violin --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor3_Violin --move $move_option                          # Done
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor4_Dancing --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor4_Dancing --move $move_option                       # Done
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor5_Oil-paper_Umbrella --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor5_Oil-paper_Umbrella --move $move_option  # Done 
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor6_Changing_Clothes --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor6_Changing_Clothes --move $move_option      # Done
python hifi4g_process.py --input /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor7_Nunchaku --output /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor7_Nunchaku --move $move_option                     # Done

cd $current_dir
echo "HiFi4G Dataset processed successfully"