#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

# feat_types:   attn_map, query, key, value
#               mmdit_attn, mmdit_attn_1_scale, mmdit_attn_2_res, mmdit_attn_3_scaleshift, mmdit_attn_4_norm
#               mmdit_ff, mmdit_ff_1_scale, mmdit_ff_2_res

CUDA=3
# Loop through different stop steps
for stop_step in 24; do
    # Loop through different feature types
    for feat_type in mmdit_attn_4_norm; do
        # Loop through different layers
        for layer in 10 9 11; do

            MODEL_ARGS="
                --model sd3_joint \
                --CONCAT_WIDTH \
                --inf_max_step 28 \
                --inf_stop_step ${stop_step} \
                --feat_save_path ./extracted_feats/spair_sd3_joint_concat/maxstep28_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            LAYER_SELECTION_ARGS="
                --output_feat_type ${feat_type} \
                --output_layer ${layer} \
            "

            LOG_ARGS="
                --log_tool wandb \
                --wandb_path ./wandb \
                --wandb_proj_name zeroshot_matching \
                --wandb_exp_name server5_spair_sd3_joint_concat_EVALSAMPLE20_maxstep28_stopstep${stop_step}_${feat_type}_layer${layer}_gpu${CUDA} \
                --save_dir ./vis/spair/sd3_joint_concat/EVALSAMPLE20_maxstep28_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            VIS_ARGS="
                --WANDB_LOG_FREQ 4 \
                --VIS_PCA_JOINT_IMG \
                --VIS_KPTS_PREDICTION \
            "

            ETC_ARGS="
                --seed 1997 \
                --EVAL_SAMPLE_NUM 20 \
            "

            CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}

        done
    done
done
