#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

MODEL_ARGS="
    --model sd3_baseline \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 8 \
    --feat_save_path ./extracted_feats/spair_sd3_baseline \

"

# --feat_already_extracted 

LOG_ARGS="
    --log_tool wandba \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_sd3_baseline \
    --save_dir ./vis/eval/spair_kpts/sd3_baseline \
    --vis_pred_kpts \
"

ETC_ARGS="
    --seed 1997 \
"

CUDA=2
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LOG_ARGS} ${ETC_ARGS}


