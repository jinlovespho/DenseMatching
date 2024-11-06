#!/bin/bash

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset eth3d \
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 2 3 \
    --model croco_catseg \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0080.pth.tar \
    --output_flow_interp \
    --output_ca_map \
    --softmax_camap \
    --correlation \
    --reciprocity \
    --save_dir ./vis/eval/zoom23_eth3d_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0080_fineflow \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_zoom23_eth3d_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0080_fineflow\
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \
