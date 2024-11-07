#!/bin/bash

CUDA=2,3,4,5
BATCH_SIZE=12
NPROC_PER_NODE=4
DATA_ARGS="
    --dataset dped \
    --apply_coco_msk \
"
TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 1e-3 \
    --max_epoch 100 \
"
MODEL_ARGS="
    --model croco_catseg \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_flow_interp \
    --output_ca_map \
    --softmax_camap \
    --correlation \
    --reciprocity \
    --freeze croco_all \
"
# --freeze none     # full fine tuning
# --freeze croco_enc    # freeze only croco encoder
# --freeze croco_all    # freeze all croco parameters but the aggregator
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_ddp_test \
"
ETC_ARGS="
    --multi_gpu \
"

CUDA_VISIBLE_DEVICES=${CUDA} \
    torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_static_stage1_multigpu' \
    ${DATA_ARGS} \
    ${TRAIN_ARGS} \
    ${MODEL_ARGS} \
    ${LOG_ARGS} \
    ${ETC_ARGS}



# CUDA_VISIBLE_DEVICES=${CUDA} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_static_ddp' \
#                                                                                             ${DATA_ARGS} \
#                                                                                             ${TRAIN_ARGS} \
#                                                                                             ${MODEL_ARGS} \
#                                                                                             ${LOG_ARGS} \
#                                                                                             ${ETC_ARGS}


