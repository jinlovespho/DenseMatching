#!/bin/bash

CUDA=0
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp/stage1_dped_crocov2_camap_freezeCrocoEnc_bestpath_100ep \
    --log_warped_images \


## 1. bestpath 25ep
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar
# missing keys:  []
# unexpected keys:  []
# ---------------------------------------
# Hpatches Eval Img Size:  original size
# SUPPL ARGS.CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# SUPPL ARGS.OUTPUT_MODE:  ca_map
# 100%|████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.91it/s]
# Validation EPE: 21.462945, 1px: 0.001458, 3px: 0.013589, 5px: 0.043930
# 100%|████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:29<00:00,  1.99it/s]
# Validation EPE: 28.215886, 1px: 0.001360, 3px: 0.012232, 5px: 0.033881
# 100%|████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:29<00:00,  1.99it/s]
# Validation EPE: 35.539890, 1px: 0.001383, 3px: 0.012775, 5px: 0.035391
# 100%|████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.96it/s]
# Validation EPE: 41.819876, 1px: 0.001411, 3px: 0.012569, 5px: 0.034078
# 100%|████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:33<00:00,  1.75it/s]
# Validation EPE: 46.265134, 1px: 0.001398, 3px: 0.012544, 5px: 0.034533
# 100%|██████████████████████████████████████████████████████████████████████████████████| 295/295 [02:30<00:00,  1.96it/s]
# Validation EPE: 34.660746, 1px: 0.001403, 3px: 0.012765, 5px: 0.036622


## 2. bestpath 68ep
# <All keys matched successfully>
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar
# missing keys:  []
# unexpected keys:  []
# ---------------------------------------
# Hpatches Eval Img Size:  original size
# SUPPL ARGS.CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# SUPPL ARGS.OUTPUT_MODE:  ca_map
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.93it/s]
# Validation EPE: 20.234947, 1px: 0.001750, 3px: 0.016340, 5px: 0.052113
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.92it/s]
# Validation EPE: 27.855748, 1px: 0.001589, 3px: 0.014330, 5px: 0.039848
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.96it/s]
# Validation EPE: 32.989401, 1px: 0.001566, 3px: 0.014141, 5px: 0.039186
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.93it/s]
# Validation EPE: 40.879948, 1px: 0.001548, 3px: 0.014043, 5px: 0.038494
# 100%|████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:30<00:00,  1.95it/s]
# Validation EPE: 44.030265, 1px: 0.001477, 3px: 0.013266, 5px: 0.036238
# 100%|██████████████████████████████████████████████████████████████████████████████████████████| 295/295 [02:28<00:00,  1.99it/s]
# Validation EPE: 33.198062, 1px: 0.001595, 3px: 0.014517, 5px: 0.041650


## 3. bestpath 82ep
# AEPE":19.738332651429257
# AEPE":27.29762840270996
# AEPE":32.966515444092835
# AEPE":39.227307529772744
# AEPE":44.19593248528949
# AEPE":32.68514330265886

## 4. bestpath 92ep
# Validation EPE: 19.692216, 1px: 0.001938, 3px: 0.017613, 5px: 0.054889
# Validation EPE: 27.281487, 1px: 0.001698, 3px: 0.015357, 5px: 0.042939
# Validation EPE: 32.863719, 1px: 0.001680, 3px: 0.015277, 5px: 0.041942
# Validation EPE: 38.558989, 1px: 0.001667, 3px: 0.015231, 5px: 0.041273
# Validation EPE: 44.311169, 1px: 0.001572, 3px: 0.014135, 5px: 0.038723
# Validation EPE: 32.541516, 1px: 0.001722, 3px: 0.015625, 5px: 0.044441

## 5. bestpath 100ep
# Validation EPE: 19.572560, 1px: 0.001900, 3px: 0.017446, 5px: 0.054701
# Validation EPE: 27.173984, 1px: 0.001668, 3px: 0.015198, 5px: 0.042606
# Validation EPE: 32.884409, 1px: 0.001657, 3px: 0.015222, 5px: 0.041973
# Validation EPE: 39.478179, 1px: 0.001653, 3px: 0.015101, 5px: 0.040839
# Validation EPE: 44.239093, 1px: 0.001569, 3px: 0.014172, 5px: 0.038938
# Validation EPE: 32.669645, 1px: 0.001699, 3px: 0.015524, 5px: 0.044288