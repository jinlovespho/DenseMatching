#!/bin/bash

CUDA=4
BATCH_SIZE=8

DATA_ARGS="
    --dataset dped \
    --apply_coco_msk \
"
TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 1e-4 \
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
LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_TRAIN_dpedmsk_img224_bs${BATCH_SIZE}_lr1e4_croco_catseg_freezeCrocoAll \
"
ETC_ARGS="

"


CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static' \
                                                                                    ${DATA_ARGS} \
                                                                                    ${TRAIN_ARGS} \
                                                                                    ${MODEL_ARGS} \
                                                                                    ${LOG_ARGS} \
                                                                                    ${ETC_ARGS}



    




CUDA=6

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static' \
    --dataset dped \
    --apply_coco_msk \

    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_${CUDA}_dped_img224_bs2_croco_cats_swin_decoder_TEST \
    --img_size 224 224 \
    --batch_size 2 \
    --softmaxattn \
    --reciprocity \
    --cost_agg cats_swin_decoder \
    --cost_transformer \
    --correlation \
#  --multi_gpu \



# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
#  --tag hierarchical_conv4d_cats_level_4stage_again \
#  --img_size 224 224 \
#  --softmaxattn \
#  --reciprocity \
#  --cost_agg hierarchical_conv4d_cats_level_4stage \
#  --cost_transformer \
#  --correlation \
#  --hierarchical \
#  --batch_size 1 \
#  --cats_depth 2