#!/bin/bash

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0020.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --uncertainty \
    --softargmax_beta 2e-2 \
    --save_dir ./vis/eval/hp240/straight_to_stage2_dpedcocomega_crocov2_camap_freezeCrocoEnc_beta2e2_ep20_beta2e2 \
    --log_warped_images \

# RERE
# ep1 (inf temp=2e-2)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.112328, 1px: 0.028017, 3px: 0.235395, 5px: 0.525951
# Validation EPE: 7.197368, 1px: 0.024742, 3px: 0.211946, 5px: 0.497106
# Validation EPE: 8.733730, 1px: 0.025610, 3px: 0.211496, 5px: 0.496634
# Validation EPE: 9.528928, 1px: 0.025200, 3px: 0.204355, 5px: 0.476499
# Validation EPE: 12.401749, 1px: 0.023316, 3px: 0.194909, 5px: 0.463538
# Validation EPE: 8.594820, 1px: 0.025513, 3px: 0.212959, 5px: 0.494137

# ep1 (inf temp=1e-4)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.280307, 1px: 0.026451, 3px: 0.223912, 5px: 0.499946
# Validation EPE: 7.326096, 1px: 0.023736, 3px: 0.201973, 5px: 0.475521
# Validation EPE: 8.929909, 1px: 0.024448, 3px: 0.201269, 5px: 0.476296
# Validation EPE: 9.680604, 1px: 0.023973, 3px: 0.193013, 5px: 0.456700
# Validation EPE: 12.579602, 1px: 0.022514, 3px: 0.186021, 5px: 0.443590
# Validation EPE: 8.759304, 1px: 0.024340, 3px: 0.202514, 5px: 0.472394


# ep5 (inf temp=2e-2)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 4.863292, 1px: 0.036639, 3px: 0.279188, 5px: 0.579030
# Validation EPE: 7.094421, 1px: 0.030859, 3px: 0.244769, 5px: 0.531561
# Validation EPE: 7.527510, 1px: 0.028147, 3px: 0.234507, 5px: 0.517330
# Validation EPE: 9.276049, 1px: 0.028359, 3px: 0.229003, 5px: 0.504269
# Validation EPE: 10.329780, 1px: 0.024899, 3px: 0.206840, 5px: 0.469461
# Validation EPE: 7.818210, 1px: 0.030178, 3px: 0.241310, 5px: 0.524091


# ep5 (inf temp=1e-4)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.058606, 1px: 0.034342, 3px: 0.261123, 5px: 0.550313
# Validation EPE: 7.324289, 1px: 0.028813, 3px: 0.229324, 5px: 0.505423
# Validation EPE: 7.729349, 1px: 0.026223, 3px: 0.220284, 5px: 0.494464
# Validation EPE: 9.455922, 1px: 0.026969, 3px: 0.214196, 5px: 0.478695
# Validation EPE: 10.542724, 1px: 0.023175, 3px: 0.194715, 5px: 0.448859
# Validation EPE: 8.022178, 1px: 0.028276, 3px: 0.226183, 5px: 0.499046


# ep10 (inf temp=2e-2)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 2.398470, 1px: 0.232297, 3px: 0.756305, 5px: 0.929556
# Validation EPE: 4.767906, 1px: 0.171732, 3px: 0.625846, 5px: 0.849650
# Validation EPE: 6.327904, 1px: 0.156176, 3px: 0.604038, 5px: 0.832561
# Validation EPE: 7.266704, 1px: 0.145393, 3px: 0.581526, 5px: 0.797915
# Validation EPE: 9.131154, 1px: 0.120419, 3px: 0.493537, 5px: 0.726582
# Validation EPE: 5.978428, 1px: 0.168997, 3px: 0.620953, 5px: 0.834241

# ep10 (inf temp=1e-4)
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.924769, 1px: 0.021564, 3px: 0.183491, 5px: 0.419038
# Validation EPE: 7.944200, 1px: 0.019095, 3px: 0.163832, 5px: 0.404275
# Validation EPE: 9.290338, 1px: 0.020758, 3px: 0.173729, 5px: 0.416919
# Validation EPE: 10.048774, 1px: 0.021979, 3px: 0.172459, 5px: 0.411080
# Validation EPE: 11.705629, 1px: 0.018983, 3px: 0.156618, 5px: 0.377722
# Validation EPE: 8.982742, 1px: 0.020515, 3px: 0.170722, 5px: 0.406997


# ep20 (inf temp=2e-2) 
# Validation EPE: 1.832398, 1px: 0.396040, 3px: 0.844893, 5px: 0.965257
# Validation EPE: 4.206167, 1px: 0.281549, 3px: 0.714859, 5px: 0.887041
# Validation EPE: 6.478674, 1px: 0.250269, 3px: 0.673155, 5px: 0.861929
# Validation EPE: 7.495895, 1px: 0.240575, 3px: 0.654923, 5px: 0.832519
# Validation EPE: 8.384122, 1px: 0.183274, 3px: 0.565176, 5px: 0.772684
# Validation EPE: 5.679451, 1px: 0.277447, 3px: 0.700067, 5px: 0.870590


# ep20 (inf temp=1e-4) 
# Validation EPE: 6.530406, 1px: 0.019486, 3px: 0.161031, 5px: 0.374120
# Validation EPE: 8.294458, 1px: 0.016434, 3px: 0.146580, 5px: 0.372901
# Validation EPE: 10.344569, 1px: 0.018882, 3px: 0.160096, 5px: 0.392082
# Validation EPE: 11.104981, 1px: 0.018707, 3px: 0.150596, 5px: 0.375487
# Validation EPE: 12.096585, 1px: 0.016958, 3px: 0.144722, 5px: 0.354703
# Validation EPE: 9.674200, 1px: 0.018136, 3px: 0.153027, 5px: 0.374422

# ep25 (inf temp=2e-2) 1
# Validation EPE: 1.788211, 1px: 0.408760, 3px: 0.857883, 5px: 0.962645
# Validation EPE: 4.339093, 1px: 0.283091, 3px: 0.712652, 5px: 0.884441
# Validation EPE: 6.723362, 1px: 0.261271, 3px: 0.672465, 5px: 0.852636
# Validation EPE: 7.317772, 1px: 0.253658, 3px: 0.660635, 5px: 0.829746
# Validation EPE: 9.223360, 1px: 0.192339, 3px: 0.581618, 5px: 0.779806
# Validation EPE: 5.878359, 1px: 0.286865, 3px: 0.706269, 5px: 0.868257

# ep25 (inf temp=1e-4) 2
# Validation EPE: 6.652013, 1px: 0.019735, 3px: 0.161049, 5px: 0.372307
# Validation EPE: 8.621281, 1px: 0.016489, 3px: 0.144377, 5px: 0.367505
# Validation EPE: 10.741592, 1px: 0.018342, 3px: 0.157072, 5px: 0.385720
# Validation EPE: 11.143785, 1px: 0.019008, 3px: 0.150745, 5px: 0.373669
# Validation EPE: 13.020339, 1px: 0.016757, 3px: 0.142938, 5px: 0.351643
# Validation EPE: 10.035802, 1px: 0.018120, 3px: 0.151683, 5px: 0.370721


# ep28 (inf temp=2e-2) 5
# Validation EPE: 1.813976, 1px: 0.416542, 3px: 0.855959, 5px: 0.960181
# Validation EPE: 4.339630, 1px: 0.291868, 3px: 0.721362, 5px: 0.885086
# Validation EPE: 6.689637, 1px: 0.259252, 3px: 0.679742, 5px: 0.852705
# Validation EPE: 7.334051, 1px: 0.269813, 3px: 0.671565, 5px: 0.833260
# Validation EPE: 8.670753, 1px: 0.201641, 3px: 0.596064, 5px: 0.784808
# Validation EPE: 5.769609, 1px: 0.294740, 3px: 0.713631, 5px: 0.869343

# ep28 (inf temp=1e-4) 6
# Validation EPE: 6.711248, 1px: 0.019110, 3px: 0.158902, 5px: 0.368021
# Validation EPE: 8.777407, 1px: 0.015817, 3px: 0.142072, 5px: 0.363033
# Validation EPE: 10.798529, 1px: 0.018529, 3px: 0.156789, 5px: 0.386154
# Validation EPE: 11.176158, 1px: 0.019082, 3px: 0.151317, 5px: 0.374805
# Validation EPE: 12.759961, 1px: 0.016751, 3px: 0.143294, 5px: 0.352161
# Validation EPE: 10.044661, 1px: 0.017882, 3px: 0.150802, 5px: 0.369156












# ep1 
# <All keys matched successfully>
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0001.pth.tar
# missing keys:  []
# unexpected keys:  []
# ---------------------------------------
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.274962, 1px: 0.026455, 3px: 0.223947, 5px: 0.500059
# Validation EPE: 7.302633, 1px: 0.023872, 3px: 0.202468, 5px: 0.476309
# Validation EPE: 9.009169, 1px: 0.024506, 3px: 0.201712, 5px: 0.477038
# Validation EPE: 9.560747, 1px: 0.024000, 3px: 0.193248, 5px: 0.457220
# Validation EPE: 12.567716, 1px: 0.022530, 3px: 0.186142, 5px: 0.443869
# Validation EPE: 8.743046, 1px: 0.024390, 3px: 0.202781, 5px: 0.472882


# ep2 
# <All keys matched successfully>
# ---------------------------------------
# Weight Loaded from .tar !
# Checkpoint Path:  /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2_TRAIN_straight_to_stage2_dpedcocomega_img224_bs16_lr5e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty/CroCoNet_ep0002.pth.tar
# missing keys:  []
# unexpected keys:  []
# ---------------------------------------
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  [240, 240]
# Validation EPE: 5.165984, 1px: 0.029086, 3px: 0.239074, 5px: 0.528482
# Validation EPE: 6.770550, 1px: 0.025361, 3px: 0.212800, 5px: 0.494132
# Validation EPE: 8.922857, 1px: 0.024595, 3px: 0.209233, 5px: 0.485823
# Validation EPE: 9.236049, 1px: 0.024745, 3px: 0.200080, 5px: 0.463392
# Validation EPE: 11.269530, 1px: 0.021770, 3px: 0.185770, 5px: 0.445050
# Validation EPE: 8.272994, 1px: 0.025346, 3px: 0.211202, 5px: 0.486365


# ep10
# Validation EPE: 5.922820, 1px: 0.021593, 3px: 0.183575, 5px: 0.419173
# Validation EPE: 7.944321, 1px: 0.019134, 3px: 0.164087, 5px: 0.404729
# Validation EPE: 9.310980, 1px: 0.020803, 3px: 0.173765, 5px: 0.416708
# Validation EPE: 10.057634, 1px: 0.021976, 3px: 0.172439, 5px: 0.410769
# Validation EPE: 11.726911, 1px: 0.018923, 3px: 0.156092, 5px: 0.376685
# Validation EPE: 8.992533, 1px: 0.020528, 3px: 0.170711, 5px: 0.406852








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