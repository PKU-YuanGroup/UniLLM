#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_NPROC_PER_NODE=8
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16668
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=80
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export WANDB_PROJECT=videollama3_qwen2.5_2b
RUN_NAME=stage_1
DATA_DIR=/storage/dataset/filter_aes/final_coyo
OUTP_DIR=work_dirs_0221

cd /storage/jp/Janus
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro

# conda activate janus_pro
torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    trainer/train_v2.py \
    --deepspeed scripts/zero1.json \
    --model_type videollama3_qwen2 \
    --model_path /storage/jp/Janus/Janus-Pro-1B \
    --data_path ${DATA_DIR}/coyo_part0_aes8639812.json \
    --data_folder ${DATA_DIR} \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --llm_lr 4e-5 \
    --vision_encoder_lr 4e-5 \
    --aligner_lr 4e-5 \
    --gen_aligner_lr 4e-5 \
    --gen_head_lr 4e-5 \
    --gen_embed_lr 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --sample_rate  5,4,1 \
    --batchsize_list  1,1,1 \
    --samples_per_epoch 10000 \
    --dataset "image_under||image_gen||text_chat" \
    --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json \
    --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json \
    --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    --text_chat_data_files  /storage/yqs/dataset/BAAI/Infinity-Instruct/7M_domains/subjective/mini_output.json \




 