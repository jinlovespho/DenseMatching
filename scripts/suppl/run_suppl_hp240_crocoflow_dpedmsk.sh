#!/bin/bash

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-240\
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --model crocoflow \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/crocoflow.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho4_TRAIN_dpedmsk_img224_bs12_lr2e5_crocoflow_finetuning_freezeNone/CroCoDownstreamBinocular_ep0070.pth.tar \
    --save_dir ./vis/eval/hp240_dpedmsk_crocoflow_freezeNone_ep0070 \
    --log_tool wandba \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp240_dpedmsk_crocoflow_freezeNone_ep0070 \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \


# 1. For evaluating full fine tuned crocoflow on dpedmsk
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho4_TRAIN_dpedmsk_img224_bs14_lr2e5_crocoflow_baseline_fullfinetuning/CroCoDownstreamBinocular_ep0003.pth.tar \

# 2. For evaluating  fine tuned crocoflow(frozen croco encoder) on dpedmsk
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho5_TRAIN_dpedmsk_img224_bs14_lr2e5_crocoflow_baseline_freezeCrocoEnc/CroCoDownstreamBinocular_ep0003.pth.tar \