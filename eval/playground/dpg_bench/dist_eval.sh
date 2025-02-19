#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

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
 
for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) $ROOT_PATH/generation.py --model_path $CKPT_PATH --prompt_dirs $PROMPT_PATH \
        --outdir $IMAGE_PATH --n_samples $PIC_NUM --size $RESOLUTION\
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

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
