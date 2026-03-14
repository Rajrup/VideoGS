#!/bin/bash

set -euo pipefail

VIDEOGS_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

FORMAT="videogs"
SEQUENCE_NAME=""
FRAME_START=""
FRAME_END=""
INTERVAL=1

SH_COLOR_SPACE="klt"
NVCOMP="ANS"
QUANTIZE_STEP=0.0001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --format)         FORMAT="$2";         shift 2 ;;
        --sequence_name)  SEQUENCE_NAME="$2";  shift 2 ;;
        --frame_start)    FRAME_START="$2";    shift 2 ;;
        --frame_end)      FRAME_END="$2";      shift 2 ;;
        --interval)       INTERVAL="$2";       shift 2 ;;
        --quantize_step)  QUANTIZE_STEP="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "${FORMAT}" = "videogs" ]; then
    DATA_ROOT="/synology/rajrup/VideoGS"
    DATASET_NAME="HiFi4G_Dataset"
    if [ -z "${SEQUENCE_NAME}" ]; then
        SEQUENCE_NAME="4K_Actor1_Greeting"
    fi
    if [ -z "${FRAME_START}" ]; then
        FRAME_START=0
    fi
    if [ -z "${FRAME_END}" ]; then
        FRAME_END=200
    fi
    J=12
    SH_DEGREE=3
    PLY_PATH="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/checkpoint"
    OUTPUT_FOLDER="${DATA_ROOT}/train_output/${DATASET_NAME}/${SEQUENCE_NAME}/ablation/livogs_rlgr"
    RLGR_QP_ARGS=(--quantize_step "${QUANTIZE_STEP}")
elif [ "${FORMAT}" = "queen" ]; then
    DATA_ROOT="/synology/rajrup/Queen"
    DATASET_NAME="Neural_3D_Video"
    if [ -z "${SEQUENCE_NAME}" ]; then
        SEQUENCE_NAME="sear_steak"
    fi
    if [ -z "${FRAME_START}" ]; then
        FRAME_START=1
    fi
    if [ -z "${FRAME_END}" ]; then
        FRAME_END=300
    fi
    J=14
    SH_DEGREE=2
    QPS=0.01
    QPQ=0.04
    QPO=0.04
    QPDC=$(awk 'BEGIN{printf "%.10f", 1.0/255.0}')
    QPAC=$(awk 'BEGIN{printf "%.10f", 4.0/255.0}')
    PLY_PATH="${DATA_ROOT}/pretrained_output/${DATASET_NAME}/queen_compressed_${SEQUENCE_NAME}"
    OUTPUT_FOLDER="${PLY_PATH}/ablation/livogs_rlgr"
    RLGR_QP_ARGS=(--qps "${QPS}" --qpq "${QPQ}" --qpo "${QPO}" --qpdc "${QPDC}" --qpac "${QPAC}")
else
    echo "Unsupported format: ${FORMAT}. Use 'videogs' or 'queen'."
    exit 1
fi

echo "======================================================================"
echo "RLGR Ablation Study"
echo "  Format:    ${FORMAT}"
echo "  Sequence:  ${SEQUENCE_NAME}"
echo "  Frames:    ${FRAME_START} to ${FRAME_END} (interval=${INTERVAL})"
echo "  Output:    ${OUTPUT_FOLDER}"
echo "======================================================================"

python "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/ablation_rlgr.py" \
    --format "${FORMAT}" \
    --ply_path "${PLY_PATH}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --frame_start "${FRAME_START}" --frame_end "${FRAME_END}" --interval "${INTERVAL}" \
    --sh_degree "${SH_DEGREE}" --J "${J}" \
    "${RLGR_QP_ARGS[@]}" \
    --sh_color_space "${SH_COLOR_SPACE}" \
    --nvcomp_algorithm "${NVCOMP}"

echo ""
echo "Generating plots..."
python "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/plot_ablation_rlgr.py" \
    --input_csv "${OUTPUT_FOLDER}/ablation_rlgr.csv" \
    --output_folder "${VIDEOGS_ROOT}/scripts/livogs_baseline/ablation/plots"

echo ""
echo "======================================================================"
echo "Done! Results in: ${OUTPUT_FOLDER}"
echo "======================================================================"
