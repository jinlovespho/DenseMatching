#!/bin/bash

CUDA=2
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 7 8 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_stage1/pho3_TRAIN_stage1_dped_img224_bs12_lr1e5_crocov2_camap_freezeCrocoEnc/CroCoNet_model_best.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --save_dir ./vis/eval/hp/zoom78/stage1_dped_crocov2_camap_freezeCrocoEnc_bestpath_100ep \
    --log_warped_images \



# 1. zoom23 bestpath 82ep
# AEPE":14.002018063755358,"PCK_1_per_image":0.006807890611734195,"PCK_3_per_image":0.0590256986906752,"PCK_5_per_image":0.1500045694410801,"PCK_1_per_dataset":0.005596881237352074,"PCK_3_per_dataset":0.04874230950413597,"PCK_5_per_dataset":0.13308996925248823,"num_pixels_pck_1":255253.0,"num_pixels_pck_3":2222956.0,"num_pixels_pck_5":6069740.0,"num_valid_corr":45606292},
# AEPE":20.77704397298522,"PCK_1_per_image":0.006559349268958983,"PCK_3_per_image":0.05629144773984543,"PCK_5_per_image":0.14462798807924906,"PCK_1_per_dataset":0.0059345041354996,"PCK_3_per_dataset":0.051026241063832564,"PCK_5_per_dataset":0.13148368477348205,"num_pixels_pck_1":256618.0,"num_pixels_pck_3":2206461.0,"num_pixels_pck_5":5685577.0,"num_valid_corr":43241692},
# AEPE":25.8490001549155,"PCK_1_per_image":0.005526581071674211,"PCK_3_per_image":0.048177947024730924,"PCK_5_per_image":0.12437361228191965,"PCK_1_per_dataset":0.005232882911316757,"PCK_3_per_dataset":0.045904792728733025,"PCK_5_per_dataset":0.1196445158144841,"num_pixels_pck_1":207083.0,"num_pixels_pck_3":1816609.0,"num_pixels_pck_5":4734741.0,"num_valid_corr":39573406},
# AEPE":31.65060652716685,"PCK_1_per_image":0.0054706342357271984,"PCK_3_per_image":0.04672469962617655,"PCK_5_per_image":0.11823730699232606,"PCK_1_per_dataset":0.005376076119374361,"PCK_3_per_dataset":0.046088650339177965,"PCK_5_per_dataset":0.11711313003329357,"num_pixels_pck_1":201272.0,"num_pixels_pck_3":1725488.0,"num_pixels_pck_5":4384535.0,"num_valid_corr":37438458},
# AEPE":37.252081111326056,"PCK_1_per_image":0.005396602089141675,"PCK_3_per_image":0.046261357748659006,"PCK_5_per_image":0.11723026484333647,"PCK_1_per_dataset":0.005245806520607984,"PCK_3_per_dataset":0.04531674339123825,"PCK_5_per_dataset":0.11599625076772717,"num_pixels_pck_1":177963.0,"num_pixels_pck_3":1537362.0,"num_pixels_pck_5":3935151.0,"num_valid_corr":33924812},
# AEPE":25.906149966029798,"PCK_1_per_image":0.005952211455447253,"PCK_3_per_image":0.05129623317150587,"PCK_5_per_image":0.13089474832758227,"PCK_1_per_dataset":0.005496863472901273,"PCK_3_per_dataset":0.047595631216130405,"PCK_5_per_dataset":0.12418242721938712,"num_pixels_pck_1":1098189.0,"num_pixels_pck_3":9508877.0,"num_pixels_pck_5":24809744.0,"num_valid_corr":199784660}}}


# 2. zoom34 bestpath 92ep
# DENSE_ZOOM_IN_RATIO:  [3, 4]
# CROCO_CKPT:  ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth
# OUTPUT_CORRELATION:  ca_map
# Hpatches Eval Img Size:  Original size
# Validation EPE:   11.297812, 1px: 0.010857, 3px: 0.091631, 5px: 0.227425
# Validation EPE: 17.961576, 1px: 0.011109, 3px: 0.092393, 5px: 0.220742
# Validation EPE: 23.246380, 1px: 0.010323, 3px: 0.086241, 5px: 0.207523
# Validation EPE: 31.039485, 1px: 0.010202, 3px: 0.083925, 5px: 0.199550
# Validation EPE: 34.574096, 1px: 0.009914, 3px: 0.082306, 5px: 0.196353
# Validation EPE: 23.623870, 1px: 0.010523, 3px: 0.087701, 5px: 0.211536

## 3. zoom34 bestpath 100ep (pho1)
# Validation EPE: 11.260138, 1px: 0.010885, 3px: 0.090615, 5px: 0.224868
# Validation EPE: 18.280423, 1px: 0.011092, 3px: 0.092818, 5px: 0.222523
# Validation EPE: 23.633522, 1px: 0.010280, 3px: 0.085880, 5px: 0.206993
# Validation EPE: 29.510561, 1px: 0.010283, 3px: 0.084614, 5px: 0.200348
# Validation EPE: 33.643412, 1px: 0.009978, 3px: 0.083205, 5px: 0.198124
# Validation EPE: 23.265611, 1px: 0.010543, 3px: 0.087771, 5px: 0.211684


## 4. zoom234 bestpath 100ep (pho2)
# Validation EPE: 11.462083, 1px: 0.010338, 3px: 0.086446, 5px: 0.215439                                                
# Validation EPE: 18.480459, 1px: 0.010639, 3px: 0.088889, 5px: 0.213450  
# Validation EPE: 23.898340, 1px: 0.009823, 3px: 0.082235, 5px: 0.198450 
# Validation EPE: 29.471067, 1px: 0.009834, 3px: 0.081027, 5px: 0.192577                                                
# Validation EPE: 33.914129, 1px: 0.009519, 3px: 0.079655, 5px: 0.190406                                                
# Validation EPE: 23.445215, 1px: 0.010068, 3px: 0.083972, 5px: 0.203108

## 5. zoom45 bestpath 100ep (pho3)
# Validation EPE: 9.891045, 1px: 0.017070, 3px: 0.136415, 5px: 0.310901
# Validation EPE: 16.845970, 1px: 0.017605, 3px: 0.140317, 5px: 0.310695
# Validation EPE: 22.099138, 1px: 0.016637, 3px: 0.132371, 5px: 0.295093
# Validation EPE: 28.116416, 1px: 0.016005, 3px: 0.126330, 5px: 0.280013
# Validation EPE: 32.191197, 1px: 0.015553, 3px: 0.123039, 5px: 0.274214

## 6. zoom56 bestpath 100ep (pho0)
# Validation EPE: 9.037555, 1px: 0.024836, 3px: 0.185935, 5px: 0.388435
# Validation EPE: 15.980782, 1px: 0.025214, 3px: 0.188486, 5px: 0.385648
# Validation EPE: 21.070719, 1px: 0.023871, 3px: 0.179776, 5px: 0.369965
# Validation EPE: 27.325073, 1px: 0.022958, 3px: 0.171470, 5px: 0.352015
# Validation EPE: 31.254229, 1px: 0.021913, 3px: 0.164842, 5px: 0.342764
# Validation EPE: 20.933672, 1px: 0.023878, 3px: 0.178975, 5px: 0.369593

## 7. zoom67 bestpath 100ep (pho1)
# Validation EPE: 8.422942, 1px: 0.033528, 3px: 0.235164, 5px: 0.452044
# Validation EPE: 15.425696, 1px: 0.033293, 3px: 0.233849, 5px: 0.443375
# Validation EPE: 20.621776, 1px: 0.031548, 3px: 0.223837, 5px: 0.426926
# Validation EPE: 27.115025, 1px: 0.029979, 3px: 0.211554, 5px: 0.405740
# Validation EPE: 30.854411, 1px: 0.028168, 3px: 0.201195, 5px: 0.392429
# Validation EPE: 20.487970, 1px: 0.031510, 3px: 0.222443, 5px: 0.426392

## 8. zoom78 bestpath 100ep (pho2)
# Validation EPE: 8.064273, 1px: 0.042136, 3px: 0.278854, 5px: 0.497555
# Validation EPE: 15.064446, 1px: 0.041177, 3px: 0.272826, 5px: 0.484562
# Validation EPE: 20.363822, 1px: 0.039422, 3px: 0.262131, 5px: 0.468855
# Validation EPE: 26.900329, 1px: 0.036617, 3px: 0.243047, 5px: 0.441070
# Validation EPE: 30.620922, 1px: 0.034789, 3px: 0.235042, 5px: 0.431634
# Validation EPE: 20.202759, 1px: 0.039109, 3px: 0.260087, 5px: 0.467279