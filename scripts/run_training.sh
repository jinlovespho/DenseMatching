#!/bin/bash
CUDA=0,2,3,7

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats_multigpu' \
 --tag hierarchical_conv4d_cats_level_4stage_from_pretraincats \
 --softmaxattn \
 --reciprocity \
 --cost_agg hierarchical_conv4d_cats_level_4stage \
 --cost_transformer \
 --correlation \
 --hierarchical \
 --batch_size 48 \
 --pretrain_cats /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_cats__reciprocity_lr1e4_aftersoftmax_correlation/CroCoNet_model_best.pth.tar
