#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 768 768 \
"

MODEL_ARGS="
    --model sd3_baseline_joint \
    --time_stop 200 \
    --feat_save_path ./extracted_feats/spair_sd3_baseline_joint/final_output_before_proj_t200 \
    --feat_already_extracted 
"

# --feat_already_extracted 

LOG_ARGS="
    --log_tool wandba \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_sd3_baseline_joint_final_output_before_proj_t200 \
    --save_dir ./vis/spair/sd3_baseline_joint/final_output_before_proj_t200 \
    --vis_pred_kpts \
    --vis_pca \
    --vis_attn_maps \

"

# --vis_pred_kpts
# --vis_pca
# --vis_attn_maps 


ETC_ARGS="
    --seed 1997 \
    --is_joint \
    --output_attn_maps \
"
# --output_attn_maps \

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LOG_ARGS} ${ETC_ARGS}


