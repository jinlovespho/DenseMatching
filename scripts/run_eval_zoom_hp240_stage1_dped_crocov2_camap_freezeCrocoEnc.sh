#!/bin/bash

CUDA=4
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 3 4 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp240/zoom34/stage1_dped_crocov2_camap_freezeCrocoEnc_bestpath_97ep \
    --log_warped_images \


## 1. zoom34 bestpath 97ep (pho4)
# DENSE_ZOOM_IN_RATIO:  [3, 4]                                                                                                                                                                                                                      
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth                                                                                                                                                                               
# OUTPUT_CORRELATION:  ca_map                                                                                                                                                                                                                       
# Hpatches Eval Img Size:  [240, 240]                                                                                                                                                                                                               
# Validation EPE: 2.816381, 1px: 0.167891, 3px: 0.668701, 5px: 0.870017                                                                                                                                                                             
# Validation EPE: 5.140754, 1px: 0.160924, 3px: 0.651965, 5px: 0.859670                                                                                                                                                                             
# Validation EPE: 6.398983, 1px: 0.148458, 3px: 0.616046, 5px: 0.829535                                                                                                                                                                             
# Validation EPE: 7.912957, 1px: 0.146927, 3px: 0.603773, 5px: 0.811478                                                                                                                                                                             
# Validation EPE: 9.604740, 1px: 0.144312, 3px: 0.590934, 5px: 0.801124
# Validation EPE: 6.374763, 1px: 0.154632, 3px: 0.629384, 5px: 0.837188