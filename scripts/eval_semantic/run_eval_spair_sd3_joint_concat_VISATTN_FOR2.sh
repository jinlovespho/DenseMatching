#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

CUDA=2
# Loop through different stop steps
for stop_step in 22; do
    # Loop through different feature types
    for feat_type in attn_map; do

        MODEL_ARGS="
            --model sd3_joint \
            --CONCAT_WIDTH \
            --inf_max_step 28 \
            --inf_stop_step ${stop_step} \
            --feat_save_path ./extracted_feats/spair_sd3_joint_concat/maxstep28_stopstep${stop_step}_${feat_type} \
        "

        LAYER_SELECTION_ARGS="
            --output_feat_type ${feat_type} \
        "

        LOG_ARGS="
            --log_tool wandba \
            --wandb_path ./wandb \
            --wandb_proj_name zeroshot_matching \
            --wandb_exp_name server5_spair_sd3_joint_concat_EVALSAMPLE20_maxstep28_stopstep${stop_step}_${feat_type}_gpu${CUDA} \
            --save_dir ./vis/spair/sd3_joint_concat/EVALSAMPLE20_maxstep28_stopstep${stop_step}_${feat_type} \
        "

        VIS_ARGS="
            --WANDB_LOG_FREQ 4 \
            --VIS_ATTN_MAP \
            --VIS_ATTN_SRC_TO_SRC \
        "

        ETC_ARGS="
            --seed 1997 \
            --EVAL_SAMPLE_NUM 20 \
        "

        CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}

    done
done
