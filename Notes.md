# Notes

## ActorsHQ Dataset Preprocess

**Download Location:** /synology/actorshq/

**Input layout:** COLMAP per frame under `.../Actor08/Sequence1/4x/frames/frame<N>/` with `images/`, `masks/`, and `sparse/` (cameras.bin, images.bin, points3D.bin). See `preprocess/actorshq_preprocess.md` for details.

```bash
cd preprocess

# Note: Check that aabb_scale is set to 2 in actorshq_preprocess.py (which is passed to colmap2k.py), so that the head is not clipped.

# Process all frames; Optional: --start 0 --end 99 to process a frame range; --no_white_background to skip mask compositing; --move to move the original data to the output folder.
python actorshq_preprocess.py --input /synology/actorshq/colmap/Actor08/Sequence1/4x/frames --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor08_Sequence1_4x --start 0 --end 39

python actorshq_preprocess.py --input /synology/actorshq/colmap/Actor01/Sequence1/4x/frames --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor01_Sequence1_4x # Done

python actorshq_preprocess.py --input /synology/actorshq/colmap/Actor08/Sequence1/4x/frames --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor08_Sequence1_4x # Done

python actorshq_preprocess.py --input /synology/actorshq/colmap/Actor01/Sequence1/4x/frames --output /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed2/Actor01_Sequence1_4x --start 0 --end 19 # Doing
```

## Training

### Training on HiFi4G Dataset

**Note:** The original README says sh = 0 by default (perhaps the iOS viewer only supports sh = 0). However, I use sh = 3 to be check compressed size compared to LiVoGS.

```bash
python train_sequence.py --start 0 --end 200 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing --sh 3 --interval 1 --group_size 20 --resolution 2    # Done

python train_sequence.py --start 0 --end 200 --cuda 1 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor3_Violin --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor3_Violin --sh 3 --interval 1 --group_size 20 --resolution 2       # Done

python train_sequence.py --start 0 --end 40 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res4 --sh 0 --interval 1 --group_size 20 --resolution 4 # App working

python train_sequence.py --start 0 --end 40 --cuda 0 --data /synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing --output /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res2 --sh 0 --interval 1 --group_size 20 --resolution 2 # App working
```

### Training on ActorsHQ Dataset

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence.py --start 0 --end 2200 --cuda 0 --data /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor01_Sequence1_4x --output /synology/rajrup/VideoGS/train_output/ActorsHQ_Dataset/Actor01_Sequence1_4x --sh 3 --interval 1 --group_size 20 --resolution 1    # Done till 1039, error at 1040

CUDA_VISIBLE_DEVICES=1 python train_sequence.py --start 0 --end 2360 --cuda 1 --data /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed/Actor08_Sequence1_4x --output /synology/rajrup/VideoGS/train_output/ActorsHQ_Dataset/Actor08_Sequence1_4x --sh 3 --interval 1 --group_size 20 --resolution 1    # Done till 1613, error at 1614

python train_sequence.py --start 0 --end 20 --cuda 0 --data /synology/rajrup/VideoGS/ActorsHQ_Dataset_processed2/Actor01_Sequence1_4x --output /synology/rajrup/VideoGS/train_output/ActorsHQ_Dataset2/Actor01_Sequence1_4x --sh 3 --interval 1 --group_size 20 --resolution 1 # Doing
```

## Compression

### Compression on HiFi4G Dataset

```bash
cd compress
python compress_ckpt_2_image_precompute.py --frame_start 0 --frame_end 200 --group_size 20 --interval 1 --ply_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/checkpoint --output_folder /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/feature_image --sh_degree 0    # Done

#/home/rajrup/Project/VideoGS/compress/compress_ckpt_2_image_precompute.py:11: RuntimeWarning: invalid value encountered in divide
# normalized = (data - min_val) / (max_val - min_val) * 255.0
#/home/rajrup/Project/VideoGS/compress/compress_ckpt_2_image_precompute.py:12: RuntimeWarning: invalid value encountered in cast
# return normalized.astype(np.uint8), min_val, max_val

python compress_ckpt_2_image_precompute.py --frame_start 0 --frame_end 40 --group_size 20 --interval 1 --ply_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res4/checkpoint --output_folder /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res4/feature_image --sh_degree 0 # App working

python compress_ckpt_2_image_precompute.py --frame_start 0 --frame_end 40 --group_size 20 --interval 1 --ply_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res2/checkpoint --output_folder /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res2/feature_image --sh_degree 0 # App working

# QP = lower refers to higher quality, but larger size
# QP = 22 is the highest recommended value used in the paper
python compress_image_2_video.py --frame_start 0 --frame_end 200 --group_size 20 --output_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing --qp 22    # Done

python compress_image_2_video.py --frame_start 0 --frame_end 40 --group_size 20 --output_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res4 --qp 15 # App working

python compress_image_2_video.py --frame_start 0 --frame_end 40 --group_size 20 --output_path /synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing_sh0_res2 --qp 15 # Testing
```
