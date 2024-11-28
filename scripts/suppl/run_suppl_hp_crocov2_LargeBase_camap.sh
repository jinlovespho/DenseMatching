#!/bin/bash

# output_mode: enc_feat, dec_feat, ca_map 


CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model crocov2 \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_mode ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/suppl/hp/ca_map/crocov2_LargeBase \
