#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 768 768 \
"

MODEL_ARGS="
    --model dift_sd \
    --t 261 \
    --up_ft_index 1 \
    --ensemble_size 1 \
    --feat_save_path ./extracted_feats/spair_dift_ensemble1 \
    --feat_already_extracted \

"
# --feat_already_extracted 

LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_EVALSAMPLE_ALL_dift_sd_ensemble1 \
    --save_dir ./vis/spair/EVALSAMPLE_ALL_dift_sd_ensemble1 \
"

VIS_ARGS="
    --WANDB_LOG_FREQ 75 \
    --VIS_PCA_SINGLE_IMG \
    --VIS_PCA_JOINT_IMG \
    --VIS_KPTS_PREDICTION \
"

ETC_ARGS="
    --seed 1997 \
    --EVAL_SAMPLE_NUM -1 \
"

# EVAL_SAMPLE_NUM -1 for evaluating on all samples

CUDA=1
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}



# ensemble_size 1 


