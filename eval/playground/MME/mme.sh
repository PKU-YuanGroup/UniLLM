#!/bin/bash
CKPT_PATH="/storage/jp/Janus/Janus-Pro-1B"
NAME="test"

DATA_PATH="eval/eval_data/MME"
deepspeed eval/scripts/model_vqa_loader.py \
    --model-path ${CKPT_PATH} \
    --question-file $DATA_PATH/llava_mme.jsonl \
    --image-folder $DATA_PATH/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file $DATA_PATH/answers/${NAME}.jsonl 

cd $DATA_PATH
mkdir results/$NAME
python convert_answer_to_mme.py --experiment $NAME

python eval_tool/calculation.py --results_dir eval_tool/answers/$NAME

