#!/bin/bash
CUDA=5

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_gpu5_dped_img224_bs2_croco_hierarchical_cats_aggregatesloss_freeze_decfeat \
    --img_size 224 224 \
    --batch_size 12 \
    --softmaxattn \
    --reciprocity \
    --cost_agg hierarchical_cats \
    --cost_transformer \
    --correlation \
    --hierarchical \
#  --multi_gpu \

