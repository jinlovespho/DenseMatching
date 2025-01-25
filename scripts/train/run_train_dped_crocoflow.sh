#!/bin/bash

CUDA=3
BATCH_SIZE=12

# training crocoflow on DPED from crocoflow ckpt

for model in crocoflow; do
    for learning_rate in 1e-4; do


        DATA_ARGS="
            --dataset dped \
            --apply_coco_msk \
        "
        TRAIN_ARGS="
            --seed 1997 \
            --img_size 224 224 \
            --batch_size ${BATCH_SIZE} \
            --lr ${learning_rate} \
            --max_epoch 100 \
        "
        MODEL_ARGS="
            --model ${model} \
            --crocoflow_ckpt ./pretrained_weights/crocoflow.pth \
            --freeze croco_all \
        "
        # --freeze none     # full fine tuning
        # --freeze croco_enc    # freeze only croco encoder
        # --freeze croco_all    # freeze all croco parameters but the aggregator
        LOG_ARGS="
            --log_tool wandb \
            --wandb_path ./ \
            --wandb_proj_name matching_dped \
            --wandb_exp_name server8_pho${CUDA}_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs${BATCH_SIZE}_lr${learning_rate}_${model}_freezeCrocoAll \
        "
        ETC_ARGS="

        "

        CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_stage1' \
                                                                                            ${DATA_ARGS} \
                                                                                            ${TRAIN_ARGS} \
                                                                                            ${MODEL_ARGS} \
                                                                                            ${LOG_ARGS} \
                                                                                            ${ETC_ARGS}


    done 
done