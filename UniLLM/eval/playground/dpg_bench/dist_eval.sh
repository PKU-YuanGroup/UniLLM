#!/bin/bash
CKPT_PATH="/storage/jp/Janus/work_dirs_0219_1/videollama3_qwen2.5_2b/stage_1/checkpoint-1250"
NAME="videollama3_qwen2.5_2b_stage1_new"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

ROOT_PATH="eval/playground/dpg_bench"
DATA_PATH="eval/eval_data/dpg_bench"
IMAGE_PATH="$DATA_PATH/images/$NAME"
PROMPT_PATH="$DATA_PATH/prompts"
RESOLUTION=${PIC_NUM:-384}
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-29500}



source /storage/miniconda3/etc/profile.d/conda.sh
conda activate dpgbench 
accelerate launch --num_machines 1 --num_processes $PROCESSES --mixed_precision "fp16" --main_process_port $PORT \
  $ROOT_PATH/compute_dpg_bench.py \
  --csv $DATA_PATH/dpg_bench.csv \
  --image-root-path $IMAGE_PATH \
  --res-path $DATA_PATH/${NAME}_results.txt \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug
