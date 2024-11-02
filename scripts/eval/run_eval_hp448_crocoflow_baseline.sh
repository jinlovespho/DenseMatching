#!/bin/bash

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1984 \
    --dataset hp-448 \
    --eval_img_size 448 448 \
    --model crocoflow \
    --pre_trained_models croco \
    --croco_ckpt_path ./pretrained_weights/crocoflow.pth \
    --save_dir ./vis/eval/hp448_crocoflow_baseline \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp448_crocoflow_baseline \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \


# CUDA=7

# CUDA_VISIBLE_DEVICES=${CUDA} python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco_flow \
#  --pre_trained_models croco \
#  --pretrain_croco_path /home/cvlab11/projects/jinlovespho/github/matching/DenseMatching_hg/pretrained_weights/crocoflow.pth \
#  --save_dir /media/dataset3/jinlovespho/ckpt/DenseMatching/eval/hpatches/imgsize224_crocoflow_baseline \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching/train_settings/croco/train_croco_dynamic_cats__pho_gpu6_dpedcoco_img224_bs6_croco_hierarchical_conv4d_cats_level_4stage/CroCoNet_model_best.pth.tar \