#!/bin/bash

CUDA=2
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-224 \
    --eval_img_size 224 224 \
    --model_img_size 224 224 \
    --model crocov1 \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/crocoflow.pth \
    --save_dir ./vis/eval/hp224_crocoflow_zeroshot \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp224_crocoflow_zeroshot \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \
