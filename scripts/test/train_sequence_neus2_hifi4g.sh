#!/bin/bash
GPU_ID=0
wordir="/home/rajrup/Project/VideoGS"
datadir="/synology/rajrup/VideoGS"
logdir="${wordir}/scripts/test/logs"
sequence_name="4K_Actor1_Greeting"

mkdir -p ${logdir}
LOG_FILE="${logdir}/train_sequence_neus2_${sequence_name}_$(date +%Y%m%d_%H%M%S).log"

# Use script command for proper tqdm support
script -q -c "python train_sequence_neus2.py \
    --start 0 \
    --end 100 \
    --cuda ${GPU_ID} \
    --data ${datadir}/HiFi4G_Dataset_processed/${sequence_name} \
    --output ${datadir}/neus2_output/HiFi4G_Dataset/${sequence_name} \
    --sh 3 \
    --interval 1 \
    --group_size 20 \
    --resolution 1" "${LOG_FILE}"