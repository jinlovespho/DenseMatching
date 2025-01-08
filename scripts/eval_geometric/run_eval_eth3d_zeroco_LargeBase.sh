#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset eth3d \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/eth3d/ZeroCo_LargeBase \
    --log_warped_images \

# eth3d - Validation EPE: 17.694414019142368, rate_3_AEPE: 16.27351925102631, rate_5_AEPE: 16.713629108739305, rate_7_AEPE: 16.921768807565723, rate_9_AEPE: 17.373323602081445, rate_11_AEPE: 17.442348225823572, rate_13_AEPE: 18.862180215270236, rate_15_AEPE: 20.274128923489997