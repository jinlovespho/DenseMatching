#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 768 768 \
"

MODEL_ARGS="
    --model dift_sd \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 8 \
    --feat_save_path ./extracted_feats/spair_dift \
    --feat_already_extracted \
"

# --feat_already_extracted 

LOG_ARGS="
    --log_tool wandba \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_dift_sd \
    --save_dir ./vis/eval/spair_kpts/dift \
    --vis_pca \
"

# --vis_pred_kpts \
# --vis_pca \


ETC_ARGS="
    --seed 1997 \

"

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LOG_ARGS} ${ETC_ARGS}


# CUDA=1
# CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
#     --seed 1997 \
#     --log_tool wandba \
#     --wandb_path ./wandb \
#     --wandb_proj_name zeroshot_matching \
#     --wandb_exp_name server5_spair_dift_sd \
#     --dataset spair \
#     --eval_img_size 768 768 \
#     --feat_save_path ./extracted_feats/spair_dift \
#     --model dift_sd \
#     --feat_already_extracted \
#     --t 261 \
#     --up_ft_index 1 \
#     --ensemble_size 8 \
#     --save_dir './vis/eval/spair_kpts/dift' \
#     --vis_pred_kpts \
    
    
    
    

