#!/bin/bash

CUDA=5
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho1_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeNone/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp/stage1_dped_crocov2_camap_freezeNone_bestpath_67ep \
    --log_warped_images \


## 1. bestpath 67ep - freezeNone
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho1_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeNone/CroCoNet_model_best.pth.tar
# missing keys:  ['x_normal', 'y_normal']
# unexpected keys:  []
# ---------------------------------------
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# Validation EPE: 20.570223, 1px: 0.001734, 3px: 0.015893, 5px: 0.050448
# Validation EPE: 28.708764, 1px: 0.001488, 3px: 0.013653, 5px: 0.038132
# Validation EPE: 37.928085, 1px: 0.001431, 3px: 0.013318, 5px: 0.037318
# Validation EPE: 42.804175, 1px: 0.001577, 3px: 0.013662, 5px: 0.037286
# Validation EPE: 52.270444, 1px: 0.001426, 3px: 0.012669, 5px: 0.034711
# Validation EPE: 36.456338, 1px: 0.001539, 3px: 0.013933, 5px: 0.040043
