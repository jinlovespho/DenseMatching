#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 480 720 \
"
CUDA=0
# Loop through different stop steps
for stop_step in 23; do
    # Loop through different feature types
    for feat_type in cogvid_attn; do
        # Loop through different layers
        for layer in 12; do

            MODEL_ARGS="
                --model cogvid_single \
                --inf_max_step 25 \
                --inf_stop_step ${stop_step} \
                --feat_save_path ./extracted_feats/spair_cogvid_single/maxstep25_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            LAYER_SELECTION_ARGS="
                --output_feat_type ${feat_type} \
                --output_layer ${layer} \
            "

            LOG_ARGS="
                --log_tool wandb \
                --wandb_path ./wandb \
                --wandb_proj_name zeroshot_matching \
                --wandb_exp_name server5_spair_VALSPLIT360_cogvid_SINGLE_maxstep25_stopstep${stop_step}_${feat_type}_layer${layer}_gpu${CUDA} \
                --save_dir ./vis/spair_VALSPLIT360/cogvid_SINGLE/maxstep25_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            VIS_ARGS="
                --WANDB_LOG_FREQ 2 \
                --VIS_PCA_SINGLE_IMG \
                --VIS_PCA_JOINT_IMG \
                --VIS_KPTS_PREDICTION \
            "

            ETC_ARGS="
                --seed 1997 \
                --SPAIR_VAL_SPLIT_360 \
                --INFERENCE_FEAT_NO_SAVE \
            "

            CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}

        done
    done
done
