#!/bin/bash

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1984 \
    --dataset hp-224 \
    --eval_img_size 224 224 \
    --model croco_hierarchical_conv4d_cats_level_4stage \
    --pre_trained_models croco \
    --croco_ckpt_path ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching/train_settings/croco/train_croco_static_cats__pho_gpu5_dped_img224_bs2_croco_hierarchical_conv4d_cats_level_4stage/CroCoNet_ep0057.pth.tar \
    --save_dir ./vis/eval/hp224_dped_img224_bs2_croco_hierarchical_conv4d_cats_level_4stage \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp224_dped_img224_bs2_croco_hierarchical_conv4d_cats_level_4stage \
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