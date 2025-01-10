#!/bin/bash

CUDA=1
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --log_tool wandb \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_dift_sd \
    --dataset spair \
    --eval_img_size 768 768 \
    --feat_save_path ./extracted_feats/spair_dift \
    --model dift_sd \
    --is_feat_extracted True \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 8 \
    --save_dir './vis/eval/spair_kpts/dift' \
    --vis_pred_kpts \
    
    
    
    

