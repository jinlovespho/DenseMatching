#!/bin/bash

STOP_STEP=20                  # Choose from: 15, 20, 25
FEAT_TYPE="mmdit_ff"          # Choose from: query, key, value, mmdit_attn, mmdit_ff
LAYER=23                      # Choose from: 0-23
CUDA=2

DATA_ARGS="
    --dataset spair \
    --eval_img_size 768 768 \
"

MODEL_ARGS="
    --model sd3_single \
    --inf_max_step 28 \
    --inf_stop_step ${STOP_STEP} \
    --feat_save_path ./extracted_feats/spair_sd3_single/maxstep28_stopstep${STOP_STEP}_${FEAT_TYPE}_layer${LAYER} \
    --feat_already_extracted \

"

LAYER_SELECTION_ARGS="
    --output_feat_type ${FEAT_TYPE} \
    --output_layer ${LAYER} \
"

LOG_ARGS="
    --log_tool wandba \
    --wandb_path ./wandb \
    --wandb_proj_name zeroshot_matching \
    --wandb_exp_name server5_spair_sd3_single_maxstep28_stopstep${STOP_STEP}_${FEAT_TYPE}_layer${LAYER} \
    --save_dir ./vis/spair/sd3_single/maxstep28_stopstep${STOP_STEP}_${FEAT_TYPE}_layer${LAYER} \
"

VIS_ARGS="
    --WANDB_LOG_FREQ 50 \
    --VIS_PCA_SINGLE_IMG \
    --VIS_PCA_JOINT_IMG \
    --VIS_KPTS_PREDICTION \
"

ETC_ARGS="
    --seed 1997 \
"

CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}
