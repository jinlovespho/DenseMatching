#!/bin/bash

CUDA=2,3,4,5
BATCH_SIZE=8
NPROC_PER_NODE=4

DATA_ARGS="
    --dataset dped_coco_mega \
    --apply_coco_msk \
"
TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 5e-5 \
    --max_epoch 100 \
"
MODEL_ARGS="
    --model croco_catseg \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_model_best.pth.tar \
    --output_flow_interp \
    --output_ca_map \
    --softmax_camap \
    --correlation \
    --reciprocity \
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
    --wandb_exp_name pho${CUDA}_TRAIN_stage2_dpedcocomega_img224_bs${BATCH_SIZE}_lr5e5_croco_catseg_freezeCrocoEnc_uncertainty_newweight \
"
ETC_ARGS="
    --multi_gpu \
"


CUDA_VISIBLE_DEVICES=${CUDA} \
    torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_dynamic_stage2' \
    ${DATA_ARGS} \
    ${TRAIN_ARGS} \
    ${MODEL_ARGS} \
    ${LOG_ARGS} \
    ${ETC_ARGS}


