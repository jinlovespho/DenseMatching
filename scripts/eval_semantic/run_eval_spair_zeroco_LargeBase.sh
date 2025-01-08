#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=2
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset spair \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/spair/ZeroCo_LargeBase_beta2e-2 \
    --log_warped_images \

