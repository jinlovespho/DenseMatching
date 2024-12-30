#!/bin/bash

CUDA=4
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --eval_img_size 240 240 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp240/stage1_dped_crocov2_camap_freezeCrocoEnc_bestpath_97ep \
    --log_warped_images \

# bestpath_97ep
# Validation EPE: 4.643737, 1px: 0.041352, 3px: 0.288789, 5px: 0.593832
# Validation EPE: 7.115308, 1px: 0.030907, 3px: 0.255230, 5px: 0.563736
# Validation EPE: 8.420992, 1px: 0.030288, 3px: 0.248098, 5px: 0.550130
# Validation EPE: 9.983325, 1px: 0.029258, 3px: 0.232577, 5px: 0.524529
# Validation EPE: 11.759537, 1px: 0.028822, 3px: 0.231258, 5px: 0.519017
# Validation EPE: 8.384580, 1px: 0.032524, 3px: 0.253248, 5px: 0.553083