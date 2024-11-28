#!/bin/bash

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model crocov2 \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth \
    --output_mode ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp_suppl_zeroshot_crocov2_BaseBase \
    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_SUPPL_hp_zeroshot_crocov2_BaseBase \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \
