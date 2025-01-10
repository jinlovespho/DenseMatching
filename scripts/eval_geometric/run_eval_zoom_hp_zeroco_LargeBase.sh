#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=1
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
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/hp/zoom34/ZeroCo_LargeBase_beta2e-2 \
    --log_warped_images \

# zoom34_beta1e-4 
# DENSE_ZOOM_IN_RATIO:  [3, 4]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# SOFT_ARGMAX_BETA:  0.0001
# Hpatches Eval Img Size:  Original size
# Validation EPE: 18.083448, 1px: 0.005130, 3px: 0.044854, 5px: 0.122775
# Validation EPE: 24.221810, 1px: 0.005572, 3px: 0.048576, 5px: 0.123750
# Validation EPE: 36.185283, 1px: 0.005379, 3px: 0.046111, 5px: 0.117066
# Validation EPE: 40.749052, 1px: 0.005482, 3px: 0.046841, 5px: 0.116794
# Validation EPE: 43.096558, 1px: 0.004984, 3px: 0.042817, 5px: 0.107407
# Validation EPE: 32.467230, 1px: 0.005316, 3px: 0.045935, 5px: 0.118125

# zoom34_beta2e-2 
# DENSE_ZOOM_IN_RATIO:  [3, 4]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# SOFT_ARGMAX_BETA:  0.02
# Hpatches Eval Img Size:  Original size
# Validation EPE: 17.582736, 1px: 0.005459, 3px: 0.048390, 5px: 0.130340
# Validation EPE: 23.570610, 1px: 0.005893, 3px: 0.051645, 5px: 0.130558
# Validation EPE: 35.625658, 1px: 0.005772, 3px: 0.048659, 5px: 0.122891
# Validation EPE: 40.208234, 1px: 0.005845, 3px: 0.049871, 5px: 0.123276
# Validation EPE: 42.650463, 1px: 0.005294, 3px: 0.045286, 5px: 0.113378
# Validation EPE: 31.927540, 1px: 0.005659, 3px: 0.048898, 5px: 0.124708