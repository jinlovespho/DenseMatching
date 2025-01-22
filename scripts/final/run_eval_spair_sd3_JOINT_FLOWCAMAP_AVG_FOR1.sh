#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

# feat_types:   attn_map query key value
#               mmdit_attn mmdit_attn_1_scale mmdit_attn_2_res mmdit_attn_3_scaleshift mmdit_attn_4_norm
#               mmdit_ff mmdit_ff_1_scale mmdit_ff_2_res

CUDA=1
# Loop through different stop steps
for stop_step in 22; do
    # Loop through different feature types
    for feat_type in attn_map; do
        # Loop through different layers
        for layer in 10; do

            MODEL_ARGS="
                --model sd3_joint \
                --inf_max_step 28 \
                --inf_stop_step ${stop_step} \
            "

            LAYER_SELECTION_ARGS="
                --output_feat_type ${feat_type} \
                --output_layer ${layer} \
            "

            LOG_ARGS="
                --log_tool wandba \
                --wandb_path ./wandb \
                --wandb_proj_name zeroshot_matching \
                --wandb_exp_name server5_spair_SORTED_VALSPLIT360_sd3_JOINT_maxstep28_stopstep${stop_step}_${feat_type}_layer${layer}_gpu${CUDA} \
                --save_dir ./vis/spair_SORTED_VALSPLIT360/sd3_JOINT/maxstep28_stopstep${stop_step}_${feat_type}_layer${layer} \
            "

            VIS_ARGS="
                --WANDB_LOG_FREQ 2 \
                --VIS_PCA_JOINT_IMG \
                --VIS_KPTS_PREDICTION \
                --AVG_ATTN_MAP \
            "

            ETC_ARGS="
                --seed 1997 \
                --SPAIR_VAL_SPLIT_360 \
                --SORT_VAL_JSON \
                --INFERENCE_FEAT_NO_SAVE \
                --FLOW_CAMAP \
            "

            CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}

        done
    done
done
