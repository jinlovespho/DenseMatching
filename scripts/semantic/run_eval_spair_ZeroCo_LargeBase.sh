#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 1e-4 \
    --save_dir ./vis/eval/hp240/ZeroCo_LargeBase_RE \


# (heuristic attention not applied)
# Validation EPE: 6.662045, 1px: 0.028129, 3px: 0.195589, 5px: 0.440810
# Validation EPE: 10.420144, 1px: 0.019333, 3px: 0.169038, 5px: 0.420197
# Validation EPE: 18.173081, 1px: 0.021324, 3px: 0.177063, 5px: 0.421563
# Validation EPE: 14.897583, 1px: 0.021344, 3px: 0.167673, 5px: 0.405281
# Validation EPE: 17.314646, 1px: 0.020684, 3px: 0.170257, 5px: 0.405598
# Validation EPE: 13.493500, 1px: 0.022353, 3px: 0.176681, 5px: 0.419955



# 홍규님꺼랑 똒같이 나오는거 확인! heuristic attn 을 안해주었음
# No pre-trained model path provided
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.923801, 1px: 0.028134, 3px: 0.195704, 5px: 0.441268
# Validation EPE: 8.121039, 1px: 0.019423, 3px: 0.170077, 5px: 0.422592
# Validation EPE: 11.548154, 1px: 0.021533, 3px: 0.178710, 5px: 0.426127
# Validation EPE: 12.458891, 1px: 0.021491, 3px: 0.168934, 5px: 0.408238
# Validation EPE: 13.113729, 1px: 0.020871, 3px: 0.171634, 5px: 0.409267
# Validation EPE: 10.233123, 1px: 0.022475, 3px: 0.177727, 5px: 0.422655




# inference temp=2e-2였을 때 결과
# Validation EPE: 5.788244, 1px: 0.028364, 3px: 0.199830, 5px: 0.452777
# Validation EPE: 8.001377, 1px: 0.020459, 3px: 0.175582, 5px: 0.435510
# Validation EPE: 11.430947, 1px: 0.022397, 3px: 0.185227, 5px: 0.439082
# Validation EPE: 12.330099, 1px: 0.022149, 3px: 0.175499, 5px: 0.422343
# Validation EPE: 12.961441, 1px: 0.021586, 3px: 0.178007, 5px: 0.421825
# Validation EPE: 10.102440, 1px: 0.023165, 3px: 0.183457, 5px: 0.435434