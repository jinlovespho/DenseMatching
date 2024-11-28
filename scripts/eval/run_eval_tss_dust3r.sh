#!/bin/bash

CUDA=7
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset TSS \
    --eval_img_size 224 224 \
    --model_img_size 224 224 \
    --model dust3r \
    --pre_trained_models croco \
    --output_flow_interp \
    --output_ca_map \
    --reciprocity \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --save_dir ./vis/eval/tss_224_crocov2_zeroshot \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_tss_224_crocov2_zeroshot \
    --wandb_log_img \
    # --uncertainty \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \
