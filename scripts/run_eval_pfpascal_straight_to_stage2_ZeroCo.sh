#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset PFPascal \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0020.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --uncertainty \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/pfpascal/straight_to_stage2_ZeroCo_beta2e2 \
    --log_warped_images \

# beta2e-2 
# Validation EPE: 43.889068, alpha=0.01: 0.034684, alpha=0.05: 0.285654, alpha=0.1: 0.511043, alpha=0.15: 0.639153

# beta1e-4
# Validation EPE: 46.839941, alpha=0.01: 0.023039, alpha=0.05: 0.274912, alpha=0.1: 0.492228, alpha=0.15: 0.621237

# 