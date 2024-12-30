#!/bin/bash
CUDA=3
BATCH_SIZE=12

DATA_ARGS="
    --dataset dped \
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
    --output_flow_interp \
    --output_ca_map \
    --reciprocity \
    --freeze croco_enc \
"
# --freeze none     # full fine tuning
# --freeze croco_enc    # freeze only croco encoder
# --freeze croco_all    # freeze all croco parameters but the aggregator
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_TRAIN_stage1_dped_img224_bs${BATCH_SIZE}_lr1e5_crocov2_camap_freezeCrocoEnc \
"
ETC_ARGS="
"
# --multi_gpu

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_stage1' \
                                                                                    ${DATA_ARGS} \
                                                                                    ${TRAIN_ARGS} \
                                                                                    ${MODEL_ARGS} \
                                                                                    ${LOG_ARGS} \
                                                                                    ${ETC_ARGS}

# CUDA_VISIBLE_DEVICES=${CUDA} \
#     torchrun --standalone --nproc_per_node 1 run_training.py 'croco' 'train_croco_static_stage1_multigpu' \
#     ${DATA_ARGS} \
#     ${TRAIN_ARGS} \
#     ${MODEL_ARGS} \
#     ${LOG_ARGS} \
#     ${ETC_ARGS}