#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=1
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --eval_img_size 240 240 \
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
    --save_dir ./vis/eval/hp240/zoom34/ZeroCo_LargeBase_beta2e-2 \
    --log_warped_images \


# 1. zoom 23
# missing keys:  ['x_normal', 'y_normal']
# unexpected keys:  ['prediction_head.weight', 'prediction_head.bias']
# CROCOV2 WEIGHT WELL LOADED:  _IncompatibleKeys(missing_keys=['x_normal', 'y_normal'], unexpected_keys=['prediction_head.weight', 'prediction_head.bias'])
# No pre-trained model path provided
# DENSE_ZOOM_IN_RATIO:  [2, 3]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# AEPE":4.75992416325262
# AEPE":6.9411467536021085
# AEPE":10.298904657363892
# AEPE":11.254751160993415
# AEPE":11.838370003942716
# AEPE":9.01861934783095

# 2. zoom34
# DENSE_ZOOM_IN_RATIO:  [3, 4]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 4.352754, 1px: 0.095948, 3px: 0.461388, 5px: 0.673923
# Validation EPE: 6.572270, 1px: 0.090986, 3px: 0.450925, 5px: 0.674320
# Validation EPE: 9.814213, 1px: 0.085500, 3px: 0.426630, 5px: 0.646585
# Validation EPE: 10.804961, 1px: 0.087260, 3px: 0.429947, 5px: 0.648893
# Validation EPE: 11.293702, 1px: 0.081376, 3px: 0.414965, 5px: 0.638975
# Validation EPE: 8.567580, 1px: 0.088723, 3px: 0.438527, 5px: 0.658013



# zoom34_beta1e-4 (pho0)


# zoom34_beta2e-2 (pho1)