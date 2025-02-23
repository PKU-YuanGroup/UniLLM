#!/bin/bash
CKPT_PATH="/storage/jp/Janus/work_dirs_0219_1/videollama3_qwen2.5_2b/stage_1/checkpoint-1250"
NAME="videollama3_qwen2.5_2b_stage1_new"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

DATA_PATH="eval/eval_data/pope"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) eval/scripts/model_vqa_loader.py \
        --model-path $CKPT_PATH \
        --question-file ${DATA_PATH}/llava_pope_test.jsonl \
        --image-folder ${DATA_PATH}/val2014 \
        --answers-file ${DATA_PATH}/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done
wait

output_file=${DATA_PATH}/answers/$NAME.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_PATH}/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 eval/scripts/eval_pope.py \
    --annotation-dir ${DATA_PATH}/coco \
    --question-file ${DATA_PATH}/llava_pope_test.jsonl \
    --result-file $output_file