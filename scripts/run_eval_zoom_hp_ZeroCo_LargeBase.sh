#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 3 4 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --save_dir ./vis/eval/hp/zoom34/ZeroCo_LargeBase \
    --log_warped_images \


## 1. zoom 23
# DENSE_ZOOM_IN_RATIO:  [2, 3]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# Validation EPE: 19.341708, 1px: 0.003189, 3px: 0.028455, 5px: 0.083369
# Validation EPE: 25.351138, 1px: 0.003653, 3px: 0.032336, 5px: 0.085391
# Validation EPE: 37.735302, 1px: 0.003384, 3px: 0.029702, 5px: 0.078431
# Validation EPE: 41.481935, 1px: 0.003570, 3px: 0.030739, 5px: 0.079253
# Validation EPE: 44.142928, 1px: 0.003211, 3px: 0.027896, 5px: 0.073478
# Validation EPE: 33.610602, 1px: 0.003403, 3px: 0.029875, 5px: 0.080378

## 2. zoom 34
# AEPE":17.348361775026483
# AEPE":23.231514106362553
# AEPE":35.25992580995722
# AEPE":39.52330531104136
# AEPE":41.263561361927096
# AEPE":31.325333672862943