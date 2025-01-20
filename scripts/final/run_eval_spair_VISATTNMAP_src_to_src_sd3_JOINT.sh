#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

CUDA=0
# Loop through different stop steps
for stop_step in 18 20 22 24; do
    # Loop through different feature types
    for feat_type in attn_map; do

        MODEL_ARGS="
            --model sd3_joint \
            --inf_max_step 28 \
            --inf_stop_step ${stop_step} \
        "

        LAYER_SELECTION_ARGS="
            --output_feat_type ${feat_type} \
        "

        LOG_ARGS="
            --save_dir ./vis/spair_SORTED_VALSPLIT360/sd3_JOINT/maxstep28_stopstep${stop_step}_${feat_type} \
        "

        VIS_ARGS="
            --VIS_ATTN_MAP \
            --VIS_ATTN_SRC_TO_SRC \
        "

        ETC_ARGS="
            --seed 1997 \
            --SPAIR_VAL_SPLIT_360 \
            --SORT_VAL_JSON \
            --INFERENCE_FEAT_NO_SAVE \
        "

        CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}

    done
done
