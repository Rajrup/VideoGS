CUDA_VISIBLE_DEVICES=0 conda run -n videogs python \
  /ssd1/haodongw/workspace/3dstream/VideoGS/scripts/gpcc_baseline/compress_decompress_pipeline.py \
  --input_dir /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak \
  --output_dir /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/gpcc/J17_rest4_dc4_op4/frame1 \
  --output_ply_dir /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/gpcc/J17_rest4_dc4_op4/frame1/decompressed_ply \
  --tmc3_path /ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3 \
  --voxel_depth 17 \
  --qp_opacity 4 \
  --qp_dc 4 \
  --qp_rest 4 \
  --frame_start 1 \
  --num_frames 1 \
&& \
CUDA_VISIBLE_DEVICES=0 conda run -n queen python \
  /ssd1/haodongw/workspace/3dstream/queen/scripts/evaluate_decompress.py \
  --config configs/dynerf.yaml \
  -s /synology/rajrup/Queen/Neural_3D_Video/sear_steak \
  -m /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak \
  --decompressed_ply_path /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/gpcc/J17_rest4_dc4_op4/frame1/decompressed_ply \
  --output_render_path /synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/gpcc/J17_rest4_dc4_op4/frame1/evaluation \
  --frame_start 1 \
  --frame_end 1 \
  --interval 1 \
  --save_renders