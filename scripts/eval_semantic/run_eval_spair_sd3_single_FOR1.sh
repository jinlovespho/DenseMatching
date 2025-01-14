#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 768 768 \
"
CUDA=1
# Loop through different stop steps
for stop_step in 25 20; do
    # Loop through different feature types
    for feat_type in mmdit_attn value key query; do
        # Loop through different layers
        for layer in 18 21 23; do

            MODEL_ARGS="
                --model sd3_single \
                --inf_max_step 28 \
                --inf_stop_step ${stop_step} \
                --feat_save_path ./extracted_feats/spair_sd3_single/maxstep28_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            LAYER_SELECTION_ARGS="
                --output_feat_type ${feat_type} \
                --output_layer ${layer} \
            "

            LOG_ARGS="
                --log_tool wandb \
                --wandb_path ./wandb \
                --wandb_proj_name zeroshot_matching \
                --wandb_exp_name server5_spair_sd3_single_maxstep28_stopstep${stop_step}_${feat_type}_layer${layer}_gpu${CUDA} \
                --save_dir ./vis/spair/sd3_single/maxstep28_stopstep${stop_step}_${feat_type}_layer${layer} \
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

        done
    done
done
