#!/bin/bash
CUDA=0,1,2,3,4,5
NPROC_PER_NODE=6
BATCH_SIZE=18

DATA_ARGS="
    --apply_coco_msk \
"
TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 1e-5 \
    --max_epoch 100 \
"
# --img_size 224 224 
# --img_size 512 512 
MODEL_ARGS="
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
    --output_flow_interp \
    --output_ca_map \
    --reciprocity \
    --freeze croco_enc \
    --output_correlation ca_map \
    --softargmax_beta 1e-4 \
    --uncertainty \
"
# 한번 끊기다 만 ckpt--path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho4,5,6_TRAIN_stage2_dpedcocomega_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_model_best.pth.tar \
# stage1 ckpt --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
# --freeze none     # full fine tuning
# --freeze croco_enc    # freeze only croco encoder
# --freeze croco_all    # freeze all croco parameters but the aggregator
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_TRAIN_stage2_dpedcocomega_img224_bs${BATCH_SIZE}_lr1e5_crocov2_camap_freezeCrocoEnc_beta1e4_uncertainty_train \
"
ETC_ARGS="
    --multi_gpu \
"
# --multi_gpu

# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_stage2_dpedcoco' \
#                                                                                     ${DATA_ARGS} \
#                                                                                     ${TRAIN_ARGS} \
#                                                                                     ${MODEL_ARGS} \
#                                                                                     ${LOG_ARGS} \
#                                                                                     ${ETC_ARGS}


# CUDA_VISIBLE_DEVICES=${CUDA} \
#     torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_dynamic_stage2' \
#     ${DATA_ARGS} \
#     ${TRAIN_ARGS} \
#     ${MODEL_ARGS} \
#     ${LOG_ARGS} \
#     ${ETC_ARGS}

CUDA_VISIBLE_DEVICES=${CUDA} \
    torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run_training.py 'croco' 'train_croco_static_multigpu_2stage' \
    ${DATA_ARGS} \
    ${TRAIN_ARGS} \
    ${MODEL_ARGS} \
    ${LOG_ARGS} \
    ${ETC_ARGS}