#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset eth3d \
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 3 4 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --save_dir ./vis/eval/eth3d/zoom34/ZeroCo_LargeBase \
    --log_warped_images \

# "avg":{"AEPE":12.717718026733424,
# "rate_3":{"AEPE":11.643104333039702
# "rate_5":{"AEPE":11.876944575012526
# "rate_7":{"AEPE":11.999819507423256
# "rate_9":{"AEPE":12.305325289156189
# "rate_11":{"AEPE":12.517150613851234
# "rate_13":{"AEPE":13.694446098998048
# "rate_15":{"AEPE":14.987235769653006


