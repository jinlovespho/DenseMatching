#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=3
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
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/hp240/ZeroCo_beta2e-2 \
    --log_warped_images \


# beta1e-4
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.923947, 1px: 0.028136, 3px: 0.195716, 5px: 0.441206
# Validation EPE: 8.121093, 1px: 0.019398, 3px: 0.170044, 5px: 0.422599
# Validation EPE: 11.551069, 1px: 0.021519, 3px: 0.178664, 5px: 0.426145
# Validation EPE: 12.452170, 1px: 0.021494, 3px: 0.168950, 5px: 0.408267
# Validation EPE: 13.114278, 1px: 0.020892, 3px: 0.171551, 5px: 0.409147
# Validation EPE: 10.232512, 1px: 0.022471, 3px: 0.177702, 5px: 0.422631

# beta2e-2
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.788239, 1px: 0.028366, 3px: 0.199830, 5px: 0.452786
# Validation EPE: 8.001320, 1px: 0.020462, 3px: 0.175594, 5px: 0.435530
# Validation EPE: 11.430671, 1px: 0.022396, 3px: 0.185218, 5px: 0.439071
# Validation EPE: 12.330729, 1px: 0.022146, 3px: 0.175496, 5px: 0.422333
# Validation EPE: 12.961935, 1px: 0.021583, 3px: 0.177969, 5px: 0.421811
# Validation EPE: 10.102579, 1px: 0.023166, 3px: 0.183455, 5px: 0.435419

