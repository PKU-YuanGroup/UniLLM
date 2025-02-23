#!/bin/bash
cd /storage/jp/Janus
CKPT_PATH="/storage/jp/Janus/work_dirs_0219_1/videollama3_qwen2.5_2b/stage_1/checkpoint-1250"
NAME="videollama3_qwen2.5_2b_stage1_new"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
EVAL="eval/playground/gqa"
GQADIR="eval/eval_data/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) eval/scripts/model_vqa_loader.py \
        --model-path ${CKPT_PATH} \
        --question-file ${GQADIR}/$SPLIT.jsonl \
        --image-folder ${GQADIR}/data/images \
        --answers-file ${GQADIR}/answers/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done
wait
output_file=${GQADIR}/answers/$NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${GQADIR}/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $GQADIR/$SPLIT/${NAME}
python3 eval/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${NAME}/testdev_balanced_predictions.json

python3 $EVAL/eval.py --tier $GQADIR/$SPLIT/${NAME}/testdev_balanced \
                         --questions $GQADIR/data/questions1.2/testdev_balanced_questions.json