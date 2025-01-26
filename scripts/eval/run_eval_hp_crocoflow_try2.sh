#!/bin/bash

CUDA=1

for epoch in 10 20 30 40 50 60; do 

    for dpt_head_input in query_feature; do

            CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
                --seed 1997 \
                --dataset hp \
                --model_img_size 224 224 \
                --pre_trained_models croco \
                --model crocoflow \
                --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
                --crocoflow_ckpt ./pretrained_weights/crocoflow.pth \
                --DPT_HEAD_INPUT ${dpt_head_input} \
                --path_to_pre_trained_models /media/dataset1/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server5_pho2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs64_lr1e-4_crocoflow_freezeCrocoAll_try2/CroCoDownstreamBinocular_ep00${epoch}.pth.tar \
                --save_dir ./vis/eval/hp_CVPR2025REBUTTAL_dpedmsk_img224_bs64_lr1e-4_crocoflow_freezeCrocoAll_try2_CroCoNet_ep00${epoch} \
                --log_tool wandb \
                --wandb_path ./ \
                --wandb_proj_name matching_dped \
                --wandb_exp_name server5_pho${CUDA}_EVAL_hp_CVPR2025REBUTTAL_dpedmsk_img224_bs64_lr1e-4_crocoflow_freezeCrocoAll_try2_CroCoNet_ep00${epoch} \
                --wandb_log_img 
    done
done