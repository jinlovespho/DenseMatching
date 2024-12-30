#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset PFPascal \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/pfpascal/ZeroCo_beta2e2 \
    --log_warped_images \

# Validation EPE: 29.494521, alpha=0.01: 0.039313, alpha=0.05: 0.402454, alpha=0.1: 0.654100, alpha=0.15: 0.785757