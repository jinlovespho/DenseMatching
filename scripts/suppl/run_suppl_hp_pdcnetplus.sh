#!/bin/bash

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python eval_matching.py    --dataset hp \
                                                        --model PDCNet_plus \
                                                        --pre_trained_models dynamic \
                                                        --optim_iter 3 \
                                                        --local_optim_iter 7 \
                                                        --save_dir ./vis/suppl/hp \
                                                        --path_to_pre_trained_models ./pretrained_weights/PDCNet_plus_megadepth.pth.tar