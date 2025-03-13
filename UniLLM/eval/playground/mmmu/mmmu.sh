#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

EVAL="eval/playground/mmmu"
DATA_PATH="eval/eval_data/mmmu"
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) ${EVAL}/generation.py \
        --output_path $DATA_PATH/${SPLIT}_${CHUNKS}_${IDX}.jsonl \
        --prompt_config ${EVAL}/configs/llava1.5.yaml \
        --data_path $DATA_PATH/MMMU \
        --split $SPLIT \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait



python ${EVAL}/merge_json.py --answer_path ${EVAL}/answers/ --split $SPLIT

python main_eval_only.py --output_path ${EVAL}/answers/${SPLIT}_merge.jsonl

