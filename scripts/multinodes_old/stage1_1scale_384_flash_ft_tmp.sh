#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=1
ARG_RANK=0
# ARG_RANK=${2:-8}

echo $ARG_WORLD_SIZE  $ARG_RANK
# ARG_NPROC_PER_NODE=${2:-8}
ARG_NPROC_PER_NODE=1

ARG_MASTER_ADDR="127.0.0.1"
# ARG_MASTER_ADDR=$3
ARG_MASTER_PORT=16652


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



# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32




 
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
# GLOBAL_BATCH_SIZE=256
GLOBAL_BATCH_SIZE=$[256 / 16] # 
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS


# Log Arguments
export WANDB_PROJECT=videollama3_qwen2.5_2b
RUN_NAME=stage_1
DATA_DIR=/storage/dataset/filter_aes/final_coyo
OUTP_DIR=checkpoints/stage1_1scale_384_flash_ft_tmp
 
cd /storage/zhubin/Janus-MoE/ 
source /storage/miniconda3/etc/profile.d/conda.sh
conda activate janus_pro
 

export CUDA_LAUNCH_BLOCKING=1
# conda activate janus_pro
torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    train_files/train.py \
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
    --llm_lr 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --sample_rate  0,1,0 \
    --batchsize_list  0,8,0 \
    --samples_per_epoch $[20000*256] \
    --dataset "image_under||image_gen||text_chat" \
    --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json \
    --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json \
    --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    --text_chat_data_files /storage/yqs/dataset/BAAI/Infinity-Instruct/mini_uni_Gen.json  \
    --_attn_implementation_new  "flash_attention_2"   
    
    #     : bool = field(default=True)
    # is_causal: bool = field(default=False)
    # : bool = field(default=False)


    # 上面的text_chat_data_files 是mini版本
    # --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/uni_DenseFusion-4V-100k.json \
    # --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    # --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/uni_part0_cap6595998.json \
    # --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    # --text_chat_data_files /storage/yqs/dataset/BAAI/Infinity-Instruct/uni_7M.json/storage/yqs/dataset/BAAI/Infinity-Instruct/uni_Gen.json 

    # img_und  bs 12  76g  zero1 
    # txet_chat  bs 2 65927MiB  zero1 
    # --image_under_data_files  /storage/yqs/dataset/BAAI/DenseFusion-1M/DenseFusion-4V-100k/mini_uni_DenseFusion-4V-100k.json \
    # --image_under_rootdir /storage/yqs/dataset/BAAI/DenseFusion-1M/images \
    # --image_gen_data_files  /storage/dataset/filter_aes/cap_merge_final_640/recap2/mini_janus_part0_cap6595998.json \
    # --image_gen_rootdir  /storage/dataset/recap_datacomp_1b_data_20241023_supply/output_undownloaded \
    # --text_chat_data_files  /storage/yqs/dataset/BAAI/Infinity-Instruct/mini_uni_Gen.json \




 