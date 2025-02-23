#!/bin/bash
CKPT_PATH="/storage/jp/Janus/work_dirs_0219_1/videollama3_qwen2.5_2b/stage_1/checkpoint-1250"
NAME="videollama3_qwen2.5_2b_stage1_new"

DATA_PATH="eval/eval_data/mm-vet"

python eval/scripts/model_vqa.py \
    --model-path $CKPT_PATH\
    --question-file $DATA_PATH/llava-mm-vet.jsonl \
    --image-folder $DATA_PATH/images \
    --answers-file $DATA_PATH/answers/$NAME.jsonl 

mkdir -p $DATA_PATH/results
python eval/scripts/convert_mmvet_for_eval.py \
    --src $DATA_PATH/answers/$NAME.jsonl  \
    --dst $DATA_PATH/results/$NAME.json
python3 eval/scripts/eval_gpt_mmvet.py \
    --mmvet_path $DATA_PATH \
    --ckpt_name $NAME \
    --result_path $DATA_PATH/results
