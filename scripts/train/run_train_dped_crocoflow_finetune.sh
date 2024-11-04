#!/bin/bash
CUDA=5
BATCH_SIZE=12

DATA_ARGS="
    --dataset dped \
    --apply_coco_msk \
"
TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 2e-5 \
    --max_epoch 100 \
"
# --img_size 224 224 
# --img_size 512 512 
MODEL_ARGS="
    --model crocoflow \
    --croco_ckpt ./pretrained_weights/crocoflow.pth \
    --freeze croco_enc \
"
# --freeze none     # full fine tuning
# --freeze croco_enc    # freeze only croco encoder
# --freeze croco_all    # freeze all croco parameters but the aggregator
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_TRAIN_dpedmsk_img224_bs${BATCH_SIZE}_lr2e5_crocoflow_finetuning_freezeCrocoEnc \
"
ETC_ARGS="

"
# --multi_gpu

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static' \
                                                                                    ${DATA_ARGS} \
                                                                                    ${TRAIN_ARGS} \
                                                                                    ${MODEL_ARGS} \
                                                                                    ${LOG_ARGS} \
                                                                                    ${ETC_ARGS}
