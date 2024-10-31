#!/bin/bash
CUDA=5

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_gpu5_dped_img224_bs2_croco_cats_TEST \
    --img_size 224 224 \
    --batch_size 2 \
    --softmaxattn \
    --reciprocity \
    --cost_agg cats \
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