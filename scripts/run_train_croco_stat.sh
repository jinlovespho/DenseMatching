#!/bin/bash
CUDA=6

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
 --tag hierarchical_cats_aggregatesloss_freeze_decfeat \
 --img_size 224 224 \
 --softmaxattn \
 --reciprocity \
 --cost_agg hierarchical_cats \
 --cost_transformer \
 --correlation \
 --hierarchical \
 --batch_size 4 \



