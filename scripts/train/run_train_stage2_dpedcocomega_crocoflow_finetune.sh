#!/bin/bash
CUDA=2,3,4,5
BATCH_SIZE=6
NPROC_PER_NODE=4

DATA_ARGS="
    --dataset dped_coco_mega \
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
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho4_TRAIN_dpedmsk_img224_bs12_lr2e5_crocoflow_finetuning_freezeNone/CroCoDownstreamBinocular_ep0070.pth.tar
    --uncertainty \
    --freeze croco_enc \
"
# --freeze none     # full fine tuning
# --freeze croco_enc    # freeze only croco encoder
# --freeze croco_all    # freeze all croco parameters but the aggregator
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name SUPPL_pho${CUDA}_TRAIN_stage2_dpedcocomega_img224_bs${BATCH_SIZE}_lr2e5_crocoflow_freezeCrocoEnc \
"
ETC_ARGS="
    --multi_gpu
"

CUDA_VISIBLE_DEVICES=${CUDA} \
    torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_dynamic_stage2' \
    ${DATA_ARGS} \
    ${TRAIN_ARGS} \
    ${MODEL_ARGS} \
    ${LOG_ARGS} \
    ${ETC_ARGS}