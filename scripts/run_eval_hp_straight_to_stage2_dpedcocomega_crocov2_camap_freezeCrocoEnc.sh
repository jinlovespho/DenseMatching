#!/bin/bash

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0001.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --uncertainty \
    --softargmax_beta 1e-4 \
    --save_dir ./vis/eval/hp/straight_to_stage2_dpedcocomega_crocov2_camap_freezeCrocoEnc_beta2e2_ep1 \
    --log_warped_images \

# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0027.pth.tar \
# <ZeroCo_LargeBase>
# Validation EPE: 25.098247, 1px: 0.001101, 3px: 0.010631, 5px: 0.036047
# Validation EPE: 33.027848, 1px: 0.000955, 3px: 0.008691, 5px: 0.024208
# Validation EPE: 45.051622, 1px: 0.000982, 3px: 0.008859, 5px: 0.024649
# Validation EPE: 50.262740, 1px: 0.001051, 3px: 0.009156, 5px: 0.024814
# Validation EPE: 54.241661, 1px: 0.000953, 3px: 0.008637, 5px: 0.023988
# Validation EPE: 41.536424, 1px: 0.001011, 3px: 0.009245, 5px: 0.027074


# <stage1-dped> bestpath 100ep
# Validation EPE: 19.572560, 1px: 0.001900, 3px: 0.017446, 5px: 0.054701
# Validation EPE: 27.173984, 1px: 0.001668, 3px: 0.015198, 5px: 0.042606
# Validation EPE: 32.884409, 1px: 0.001657, 3px: 0.015222, 5px: 0.041973
# Validation EPE: 39.478179, 1px: 0.001653, 3px: 0.015101, 5px: 0.040839
# Validation EPE: 44.239093, 1px: 0.001569, 3px: 0.014172, 5px: 0.038938
# Validation EPE: 32.669645, 1px: 0.001699, 3px: 0.015524, 5px: 0.044288


# <beta-2e-2>

# 1. ep1 
# Validation EPE: 21.283069, 1px: 0.001530, 3px: 0.014043, 5px: 0.038261
# Validation EPE: 27.603212, 1px: 0.001396, 3px: 0.012871, 5px: 0.035594
# Validation EPE: 32.239763, 1px: 0.001458, 3px: 0.013513, 5px: 0.036568
# Validation EPE: 35.047630, 1px: 0.001503, 3px: 0.013268, 5px: 0.036676
# Validation EPE: 42.440466, 1px: 0.001480, 3px: 0.013408, 5px: 0.036413
# Validation EPE: 31.722828, 1px: 0.001473, 3px: 0.013431, 5px: 0.036737

# 2. ep10 
# Validation EPE: 22.699090, 1px: 0.001291, 3px: 0.012475, 5px: 0.034124
# Validation EPE: 27.937800, 1px: 0.001212, 3px: 0.010936, 5px: 0.030629
# Validation EPE: 33.531694, 1px: 0.001319, 3px: 0.011839, 5px: 0.032908
# Validation EPE: 35.161670, 1px: 0.001402, 3px: 0.012386, 5px: 0.034269
# Validation EPE: 41.830091, 1px: 0.001445, 3px: 0.012767, 5px: 0.034799
# Validation EPE: 32.232069, 1px: 0.001327, 3px: 0.012049, 5px: 0.033268

# 3. ep20 


# 4. recent
# Validation EPE: 24.830449, 1px: 0.001136, 3px: 0.010634, 5px: 0.029276
# Validation EPE: 31.612117, 1px: 0.001036, 3px: 0.009473, 5px: 0.026554
# Validation EPE: 35.472928, 1px: 0.001178, 3px: 0.010771, 5px: 0.029862
# Validation EPE: 37.819083, 1px: 0.001196, 3px: 0.011258, 5px: 0.030350
# Validation EPE: 43.567295, 1px: 0.001295, 3px: 0.011473, 5px: 0.030828
# Validation EPE: 34.660374, 1px: 0.001161, 3px: 0.010669, 5px: 0.029268




# <beta-1e-4>
# 1. ep1


# 2. ep10


# 3. ep20