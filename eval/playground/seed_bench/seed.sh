#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

DATA_PATH="eval/eval_data/seed_bench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) eval/scripts/model_vqa_loader.py \
        --model-path $CKPT_PATH \
        --question-file $DATA_PATH/new-seed-bench.json \
        --image-folder $DATA_PATH \
        --answers-file $DATA_PATH/answers/${NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=$DATA_PATH/answers/${NAME}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $DATA_PATH/answers/${NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $DATA_PATH/answers_upload/
# Evaluate
python eval/scripts/convert_seed_for_submission.py \
    --annotation-file $DATA_PATH/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $DATA_PATH/answers_upload/${NAME}.jsonl	
