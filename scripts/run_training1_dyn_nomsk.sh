#!/bin/bash
CUDA=5
BS=6

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats' \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_gpu${CUDA}_dpedcoco_nomsk_img224_bs${BS}_croco_hierarchical_conv4d_cats_level_4stage \
    --img_size 224 224 \
    --batch_size ${BS} \
    --softmaxattn \
    --reciprocity \
    --cost_agg hierarchical_conv4d_cats_level_4stage \
    --cost_transformer \
    --correlation \
    --hierarchical \
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