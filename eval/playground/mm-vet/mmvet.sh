#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

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
