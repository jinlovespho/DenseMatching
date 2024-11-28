#!/bin/bash

CUDA=2
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-240 \
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --model mast3r \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --output_mode dec_feat \
    --save_dir ./vis/suppl/hp240/dec_feat \
