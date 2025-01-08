#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset spair \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0020.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --uncertainty \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/spair/straight_to_stage2_ZeroCo_beta2e2 \
    --log_warped_images \
