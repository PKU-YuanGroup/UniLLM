#!/bin/bash
cd /storage/jp/Janus
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

ROOT_PATH="eval/playground/geneval"
DATA_PATH="eval/eval_data/geneval"
IMAGE_PATH="$DATA_PATH/<IMAGE_FOLDER>/$NAME"
PROMPT_PATH="$DATA_PATH/prompts"



source /storage/miniconda3/etc/profile.d/conda.sh
conda activate gen_eval

python $DATA_PATH/geneval/evaluation/evaluate_images.py \
    $IMAGE_PATH \
    --outfile "$DATA_PATH/<RESULTS_FOLDER>/${NAME}_results.jsonl" \
    --model-path "$DATA_PATH/<OBJECT_DETECTOR_FOLDER>"

python $DATA_PATH/geneval/evaluation/summary_scores.py \
    "$DATA_PATH/<RESULTS_FOLDER>/${NAME}_results.jsonl"
