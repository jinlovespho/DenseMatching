#!/bin/bash

DATA_ARGS="
    --dataset spair \
    --eval_img_size 1024 1024 \
"

# feat_types:   attn_map query key value
#               mmdit_attn mmdit_attn_1_scale mmdit_attn_2_res mmdit_attn_3_scaleshift mmdit_attn_4_norm
#               mmdit_ff mmdit_ff_1_scale mmdit_ff_2_res

CUDA=0
# Loop through different stop steps
for stop_step in 20; do
    # Loop through different feature types
    for feat_type in attn_map; do
        # Loop through different step counts
        for inf_step_count in 1; do
            # Loop through different mask attention
            for msk_attn in -1; do 
                # select attention layers 
                for vis_layer in "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"; do
                    # Loop through different beta softargmax
                    for beta in 1e-4; do

                        MODEL_ARGS="
                            --model sd3_joint \
                            --inf_max_step 28 \
                            --inf_stop_step ${stop_step} \
                            --inf_step_count ${inf_step_count} \
                            --FLOW_CAMAP \
                            --softargmax_beta ${beta} \
                        "

                        LAYER_SELECTION_ARGS="
                            --output_feat_type ${feat_type} \
                        "

                        LOG_ARGS="
                            --log_tool wandb \
                            --wandb_path ./wandb \
                            --wandb_proj_name zeroshot_matching \
                            --wandb_exp_name server5_spair_SORTED_VALSPLIT360_sd3_JOINT_FLOWCAMAP_BETA${beta}_step_max28_stop${stop_step}_count${inf_step_count}_${feat_type}_MASKATTN${msk_attn}_layer${vis_layer}_gpu${CUDA} \
                            --save_dir ./vis/spair_SORTED_VALSPLIT360/sd3_JOINT/FLOWCAMAP_BETA${beta}_step_max28_stop${stop_step}_count${inf_step_count}_${feat_type}_MASKATTN${msk_attn}_layer${vis_layer} \
                        "

                        VIS_ARGS="
                            --WANDB_LOG_FREQ 2 \
                            --VIS_PCA_JOINT_IMG \
                            --VIS_KPTS_PREDICTION \
                            --VIS_ATTN_MAP \
                            --VIS_LAYER ${vis_layer} \
                        "

                        ETC_ARGS="
                            --seed 1997 \
                            --SPAIR_VAL_SPLIT_360 \
                            --SORT_VAL_JSON \
                            --INFERENCE_FEAT_NO_SAVE \
                            --MSK_ATTN ${msk_attn} \
                        "

                        CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py ${DATA_ARGS} ${MODEL_ARGS} ${LAYER_SELECTION_ARGS} ${LOG_ARGS} ${ETC_ARGS} ${VIS_ARGS}
                    done
                done
            done
        done
    done
done
