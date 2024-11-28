#!/bin/bash

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model mast3r \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --output_mode ca_map \
    --save_dir ./vis/eval/hp240_suppl_zeroshot_mast3r_512_dpt \
    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_SUPPL_hp240_zeroshot_mast3r_512_dpt \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \
