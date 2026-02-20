#!/bin/bash
GPU_ID=0
wordir="/home/rajrup/Project/VideoGS"
datadir="/synology/rajrup/VideoGS"
logdir="${wordir}/scripts/test/logs"

mkdir -p ${logdir}
LOG_FILE="${logdir}/train_sequence_neus2_actorshq_$(date +%Y%m%d_%H%M%S).log"

# Use script command for proper tqdm support
script -q -c "python train_sequence_neus2_actorshq.py \
    --start 0 \
    --end 20 \
    --cuda ${GPU_ID} \
    --data ${datadir}/ActorsHQ_Dataset_processed/Actor01_Sequence1_4x \
    --output ${datadir}/neus2_output/ActorsHQ_Dataset/Actor01_Sequence1_4x \
    --sh 3 \
    --interval 1 \
    --group_size 20 \
    --resolution 1 \
    --aabb_scale 2" "${LOG_FILE}"