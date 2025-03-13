#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

SPLIT="MMBench_TEST_EN"
DATA_PATH="eval/eval_data/mmbench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed eval/scripts/model_vqa_mmbench.py \
        --model-path $CKPT_PATH \
        --question-file ${DATA_PATH}/$SPLIT.tsv \
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
    
mkdir -p ${DATA_PATH}/answers_upload/

python3 eval/scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_PATH}/$SPLIT.tsv \
    --result-dir ${DATA_PATH}/answers/${NAME}.jsonl   \
    --upload-dir ${DATA_PATH}/answers_upload/ \
    --experiment ${NAME}