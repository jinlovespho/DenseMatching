#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/hp/ZeroCo_LargeBase_beta2e-2 \
    --log_warped_images \


# beta1e-4
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# Validation EPE: 24.359504, 1px: 0.001138, 3px: 0.011035, 5px: 0.037385
# Validation EPE: 31.301022, 1px: 0.001063, 3px: 0.009664, 5px: 0.026823
# Validation EPE: 43.473994, 1px: 0.001161, 3px: 0.010502, 5px: 0.029083
# Validation EPE: 48.421752, 1px: 0.001309, 3px: 0.011119, 5px: 0.029513
# Validation EPE: 51.441742, 1px: 0.001112, 3px: 0.009820, 5px: 0.027151
# Validation EPE: 39.799603, 1px: 0.001154, 3px: 0.010442, 5px: 0.030242


# beta2e-2
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# Validation EPE: 23.849704, 1px: 0.001184, 3px: 0.011391, 5px: 0.038323
# Validation EPE: 30.825826, 1px: 0.001116, 3px: 0.010108, 5px: 0.027952
# Validation EPE: 42.962858, 1px: 0.001204, 3px: 0.010843, 5px: 0.030075
# Validation EPE: 47.904225, 1px: 0.001360, 3px: 0.011524, 5px: 0.030467
# Validation EPE: 50.827722, 1px: 0.001153, 3px: 0.010187, 5px: 0.028315
# Validation EPE: 39.274067, 1px: 0.001201, 3px: 0.010826, 5px: 0.031273
