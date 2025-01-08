#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=5
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset TSS \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/tss/ZeroCo_beta2e2 \
    --log_warped_images \

# TSS evaluating on subdata:  FG3DCar
# Validation EPE: 28.064628, alpha=0.01: 0.103572, alpha=0.05: 0.860095, alpha=0.1: 0.963124, alpha=0.15: 0.981058
# TSS evaluating on subdata:  JODS
# Validation EPE: 26.232237, alpha=0.01: 0.076754, alpha=0.05: 0.657398, alpha=0.1: 0.835669, alpha=0.15: 0.889797
# TSS evaluating on subdata:  PASCAL
# Validation EPE: 62.785520, alpha=0.01: 0.054279, alpha=0.05: 0.504658, alpha=0.1: 0.679757, alpha=0.15: 0.750875