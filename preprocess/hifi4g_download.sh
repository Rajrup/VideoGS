#!/bin/bash

conda activate videogs
current_dir=$(pwd)

echo "Downloading HiFi4G Dataset..."
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor1_Greeting/*"            # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor2_Dancing/*"             # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor3_Violin/*"              # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor4_Dancing/*"             # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor5_Oil-paper_Umbrella/*"  # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor6_Changing_Clothes/*"    # Done
hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir /synology/rajrup/Datasets/ --include "HiFi4G_Dataset/4K_Actor7_Nunchaku/*"            # Done

echo "Processing 4K_Actor1_Greeting..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor1_Greeting                          # Done
cat 4K_Actor1_Greeting.zip.parta* > 4K_Actor1_Greeting.zip
unzip -t 4K_Actor1_Greeting.zip
unzip 4K_Actor1_Greeting.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor1_Greeting.zip.parta*

echo "Processing 4K_Actor2_Dancing..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor2_Dancing                           # Done
cat 4K_Actor2_Dancing.zip.parta* > 4K_Actor2_Dancing.zip
unzip -t 4K_Actor2_Dancing.zip
unzip 4K_Actor2_Dancing.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor2_Dancing.zip.parta*

echo "Processing 4K_Actor3_Violin..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor3_Violin                            # Done
cat 4K_Actor3_Violin.zip.parta* > 4K_Actor3_Violin.zip
unzip -t 4K_Actor3_Violin.zip
unzip 4K_Actor3_Violin.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor3_Violin.zip.parta*

echo "Processing 4K_Actor4_Dancing..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor4_Dancing                           # Done
cat 4K_Actor4_Dancing.zip.parta* > 4K_Actor4_Dancing.zip
unzip -t 4K_Actor4_Dancing.zip
unzip 4K_Actor4_Dancing.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor4_Dancing.zip.parta*

echo "Processing 4K_Actor5_Oil-paper_Umbrella..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor5_Oil-paper_Umbrella                # Done
cat 4K_Actor5_Oil-paper_Umbrella.zip.parta* > 4K_Actor5_Oil-paper_Umbrella.zip
unzip -t 4K_Actor5_Oil-paper_Umbrella.zip
unzip 4K_Actor5_Oil-paper_Umbrella.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor5_Oil-paper_Umbrella.zip.parta*

echo "Processing 4K_Actor6_Changing_Clothes..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor6_Changing_Clothes                  # Done
cat 4K_Actor6_Changing_Clothes.zip.parta* > 4K_Actor6_Changing_Clothes.zip
unzip -t 4K_Actor6_Changing_Clothes.zip
unzip 4K_Actor6_Changing_Clothes.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor6_Changing_Clothes.zip.parta*

echo "Processing 4K_Actor7_Nunchaku..."
cd /synology/rajrup/Datasets/HiFi4G_Dataset/4K_Actor7_Nunchaku                          # Done
cat 4K_Actor7_Nunchaku.zip.parta* > 4K_Actor7_Nunchaku.zip
unzip -t 4K_Actor7_Nunchaku.zip
unzip 4K_Actor7_Nunchaku.zip
mv image_white_undistortion/colmap ./
mv colmap/sparse/0/* colmap/sparse/ && rm -rf colmap/sparse/0
rm -rf 4K_Actor7_Nunchaku.zip.parta*

cd $current_dir
echo "HiFi4G Dataset downloaded and processed successfully"