#!/bin/bash



CUDA=0

for epoch in 70 80 90; do 

    for corr in enc_feat; do

        CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
            --seed 1997 \
            --dataset hp \
            --model_img_size 224 224 \
            --model croco_catseg \
            --pre_trained_models croco \
            --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
            --path_to_pre_trained_models /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try3/CroCoNet_ep00${epoch}.pth.tar \
            --output_flow_interp \
            --output_ca_map \
            --softmax_camap \
            --correlation \
            --reciprocity \
            --save_dir ./vis/eval/hp_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try3_CroCoNet_ep00${epoch} \
            --log_tool wandb \
            --wandb_path ./ \
            --wandb_proj_name matching_dped \
            --wandb_exp_name server8_pho${CUDA}_EVAL_CVPR2025REBUTTAL_hp_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try3_CroCoNet_ep00${epoch} \
            --wandb_log_img \
            --output_correlation ${corr} 
    done
done
