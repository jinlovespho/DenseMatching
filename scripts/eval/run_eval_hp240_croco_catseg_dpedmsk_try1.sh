#!/bin/bash



CUDA=3

# lr 1e-3
# /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1 
# lr 1e-4
# /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1

for epoch in 20; do 

    CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
        --seed 1997 \
        --dataset hp-240 \
        --eval_img_size 240 240 \
        --model_img_size 224 224 \
        --model croco_catseg \
        --pre_trained_models croco \
        --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
        --path_to_pre_trained_models /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1/CroCoNet_ep00${epoch}.pth.tar \
        --output_flow_interp \
        --output_ca_map \
        --softmax_camap \
        --correlation \
        --reciprocity \
        --save_dir ./vis/eval/hp240_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep00${epoch} \
        --log_tool wandb \
        --wandb_path ./ \
        --wandb_proj_name matching_dped \
        --wandb_exp_name server8_pho${CUDA}_EVAL_CVPR2025REBUTTAL_hp240_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep00${epoch} \
        --wandb_log_img \
        --output_correlation attn_map \

done
