#!/bin/bash

CUDA=5
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 2 3 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho1_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeNone/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp/zoom23/stage1_dped_crocov2_camap_freezeNone_bestpath_67ep \
    --log_warped_images \


## 1. zoom23 - freezeNone - bestpath 67ep
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho1_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeNone/CroCoNet_model_best.pth.tar
# missing keys:  ['x_normal', 'y_normal']
# unexpected keys:  []
# ---------------------------------------
# DENSE_ZOOM_IN_RATIO:  [2, 3]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# AEPE":14.6516627053083,"PCK_1_per_image":0.0060598174325189,"PCK_3_per_image":0.053843058122842874,"PCK_5_per_image":0.1374353233535411,"PCK_1_per_dataset":0.005111531540428676,"PCK_3_per_dataset":0.04488760015832903,"PCK_5_per_dataset":0.12356634913445715,"num_pixels_pck_1":233118.0,"num_pixels_pck_3":2047157.0,"num_pixels_pck_5":5635403.0,"num_valid_corr":45606292},
# AEPE":22.244917918059784,"PCK_1_per_image":0.005936608012573634,"PCK_3_per_image":0.051243935554447166,"PCK_5_per_image":0.13187387336407574,"PCK_1_per_dataset":0.005387532014242181,"PCK_3_per_dataset":0.046619961124555444,"PCK_5_per_dataset":0.12095886534689716,"num_pixels_pck_1":232966.0,"num_pixels_pck_3":2015926.0,"num_pixels_pck_5":5230466.0,"num_valid_corr":43241692},
# AEPE":30.944811675508145,"PCK_1_per_image":0.005040368319958176,"PCK_3_per_image":0.04356225993966692,"PCK_5_per_image":0.11340754752696523,"PCK_1_per_dataset":0.004910494689287043,"PCK_3_per_dataset":0.042274577022761196,"PCK_5_per_dataset":0.11010199627497315,"num_pixels_pck_1":194325.0,"num_pixels_pck_3":1672949.0,"num_pixels_pck_5":4357111.0,"num_valid_corr":39573406},
# AEPE":35.44459763219801,"PCK_1_per_image":0.00501166191741319,"PCK_3_per_image":0.04268108119128047,"PCK_5_per_image":0.1088113372808054,"PCK_1_per_dataset":0.004919486801513032,"PCK_3_per_dataset":0.04193636928102114,"PCK_5_per_dataset":0.10720965056840749,"num_pixels_pck_1":184178.0,"num_pixels_pck_3":1570033.0,"num_pixels_pck_5":4013764.0,"num_valid_corr":37438458},
# AEPE":44.093765630560405,"PCK_1_per_image":0.00458856772851412,"PCK_3_per_image":0.0405998627146631,"PCK_5_per_image":0.10393908074473661,"PCK_1_per_dataset":0.00455277394020636,"PCK_3_per_dataset":0.04058999648988475,"PCK_5_per_dataset":0.10430863994176298,"num_pixels_pck_1":154452.0,"num_pixels_pck_3":1377008.0,"num_pixels_pck_5":3538651.0,"num_valid_corr":33924812},
# AEPE":29.4759511284909,"PCK_1_per_image":0.005327407686105535,"PCK_3_per_image":0.046386039504580104,"PCK_5_per_image":0.11909343245402482,"PCK_1_per_dataset":0.00500058412893162,"PCK_3_per_dataset":0.04346216070843477,"PCK_5_per_dataset":0.11399971849690561,"num_pixels_pck_1":999040.0,"num_pixels_pck_3":8683073.0,"num_pixels_pck_5":22775395.0,"num_valid_corr":199784660}}}