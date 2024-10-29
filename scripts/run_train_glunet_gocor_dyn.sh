#!/bin/bash
CUDA=6

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'GLUNet' 'train_GLUNet_GOCor_dynamic' \
    --batch_size 4