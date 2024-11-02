#!/bin/bash
CUDA=7
BATCH_SIZE=12

DATA_ARGS="
    --dataset dped \
    --apply_coco_msk \
"
# --apply_coco_msk \


TRAIN_ARGS="
    --seed 1997 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --lr 2e-5 \
    --max_epoch 100 \
"

MODEL_ARGS="
    --model crocoflow \
    --croco_ckpt ./pretrained_weights/crocoflow.pth \
    --freeze_croco_enc \
"


LOG_ARGS="
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho${CUDA}_TRAIN_dpedmsk_img224_bs${BATCH_SIZE}_lr2e5_crocoflow_baseline_freezeEnc \
"


ETC_ARGS="

"


CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static' \
                                                                                    ${DATA_ARGS} \
                                                                                    ${TRAIN_ARGS} \
                                                                                    ${MODEL_ARGS} \
                                                                                    ${LOG_ARGS} \
                                                                                    ${ETC_ARGS}



    


    # --softmaxattn \
    # --reciprocity \
    # --cost_agg cats_swin_decoder \
    # --cost_transformer \
    # --correlation \
#  --multi_gpu \
#  --apply_coco_msk \



# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
#  --tag hierarchical_conv4d_cats_level_4stage_again \
#  --img_size 224 224 \
#  --softmaxattn \
#  --reciprocity \
#  --cost_agg hierarchical_conv4d_cats_level_4stage \
#  --cost_transformer \
#  --correlation \
#  --hierarchical \
#  --batch_size 1 \
#  --cats_depth 2