#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --save_dir ./vis/eval/hp/ZeroCo_LargeBase \
    --log_warped_images \

# missing keys:  ['x_normal', 'y_normal']
# unexpected keys:  ['prediction_head.weight', 'prediction_head.bias']
# CROCOV2 WEIGHT WELL LOADED:  _IncompatibleKeys(missing_keys=['x_normal', 'y_normal'], unexpected_keys=['prediction_head.weight', 'prediction_head.bias'])
# No pre-trained model path provided
# ARGS.CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# ARGS.OUTPUT_MODE:  ca_map_correlation
# Hpatches Eval Img Size:  original size
# Validation EPE: 25.098247, 1px: 0.001101, 3px: 0.010631, 5px: 0.036047
# Validation EPE: 33.027848, 1px: 0.000955, 3px: 0.008691, 5px: 0.024208
# Validation EPE: 45.051622, 1px: 0.000982, 3px: 0.008859, 5px: 0.024649
# Validation EPE: 50.262740, 1px: 0.001051, 3px: 0.009156, 5px: 0.024814
# Validation EPE: 54.241661, 1px: 0.000953, 3px: 0.008637, 5px: 0.023988
# Validation EPE: 41.536424, 1px: 0.001011, 3px: 0.009245, 5px: 0.027074