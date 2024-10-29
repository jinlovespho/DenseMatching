#!/bin/bash
CUDA=5

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats' \
    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_gpu5_dpedcoco_img224_bs2_croco_try1_cats_attnmap_with_encfeat \
    --img_size 224 224 \
    --batch_size 2 \
    --softmaxattn \
    --reciprocity \
    --cost_agg try1_cats_attnmap_with_encfeat \
    --cost_transformer \
    --hierarchical \
#  --correlation \
#  --multi_gpu 

