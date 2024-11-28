#!/bin/bash

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-240 \
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --model crocov2 \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo.pth \
    --output_mode enc_feat \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/suppl/hp240/enc_feat/crocov1_BaseSmall \
