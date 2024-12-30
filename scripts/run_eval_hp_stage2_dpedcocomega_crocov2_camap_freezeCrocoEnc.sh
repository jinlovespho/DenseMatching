#!/bin/bash

CUDA=6
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp\
    --model_img_size 224 224 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2,3,4,5_TRAIN_stage2_dpedcocomega_img224_bs18_lr1e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty_continuefrom23ep/CroCoNet_ep0025.pth.tar \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --uncertainty \
    --softargmax_beta 1e-4 \
    --save_dir ./vis/eval/hp/stage2_dpedcocomega_crocov2_camap_freezeCrocoEnc_ep25_contFrom23ep \
    --log_warped_images \


# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2,3,4,5_TRAIN_stage2_dpedcocomega_img224_bs18_lr1e5_crocov2_camap_freezeCrocoEnc_beta1e4_uncertainty/CroCoNet_ep0050.pth.tar \

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


# <stage2-dpedcocomega> bestpath_1ep 




# <stage2-dpedcocomega> bestpath_13ep
# Validation EPE: 21.534795, 1px: 0.001676, 3px: 0.015286, 5px: 0.044635
# Validation EPE: 29.533286, 1px: 0.001484, 3px: 0.013432, 5px: 0.037046
# Validation EPE: 41.207740, 1px: 0.001445, 3px: 0.013095, 5px: 0.035556
# Validation EPE: 41.271619, 1px: 0.001413, 3px: 0.013293, 5px: 0.036386
# Validation EPE: 48.749005, 1px: 0.001473, 3px: 0.012899, 5px: 0.034736
# Validation EPE: 36.459289, 1px: 0.001505, 3px: 0.013672, 5px: 0.037967


# <stage2-dpedcocomega> bestpath_16ep
# Validation EPE: 22.841067, 1px: 0.001197, 3px: 0.011562, 5px: 0.031909
# Validation EPE: 27.779168, 1px: 0.001227, 3px: 0.011083, 5px: 0.030816
# Validation EPE: 34.753211, 1px: 0.001244, 3px: 0.011481, 5px: 0.032235
# Validation EPE: 35.681949, 1px: 0.001334, 3px: 0.012087, 5px: 0.033074
# Validation EPE: 38.787046, 1px: 0.001408, 3px: 0.012339, 5px: 0.033587
# Validation EPE: 31.968488, 1px: 0.001275, 3px: 0.011673, 5px: 0.032240

# <stage2-dpedcocomega> bestpath_23ep
# Validation EPE: 23.054479, 1px: 0.001192, 3px: 0.011459, 5px: 0.031683
# Validation EPE: 28.164589, 1px: 0.001214, 3px: 0.010837, 5px: 0.030014
# Validation EPE: 34.353361, 1px: 0.001251, 3px: 0.011290, 5px: 0.031854
# Validation EPE: 34.509644, 1px: 0.001330, 3px: 0.011834, 5px: 0.032584
# Validation EPE: 38.891994, 1px: 0.001411, 3px: 0.012454, 5px: 0.034000
# Validation EPE: 31.794813, 1px: 0.001272, 3px: 0.011530, 5px: 0.031918

# <stage2-dpedcocomega> bestpath_24ep
# Validation EPE: 23.205471, 1px: 0.001184, 3px: 0.011538, 5px: 0.031956
# Validation EPE: 29.192022, 1px: 0.001199, 3px: 0.010696, 5px: 0.029660
# Validation EPE: 35.110503, 1px: 0.001246, 3px: 0.011320, 5px: 0.031921
# Validation EPE: 34.993671, 1px: 0.001320, 3px: 0.011723, 5px: 0.032444
# Validation EPE: 39.486200, 1px: 0.001451, 3px: 0.012820, 5px: 0.034808
# Validation EPE: 32.397574, 1px: 0.001270, 3px: 0.011565, 5px: 0.032028

# <stage2-dpedcocomega> bestpath_30ep
# Validation EPE: 23.345305, 1px: 0.001174, 3px: 0.011503, 5px: 0.031534
# Validation EPE: 29.564140, 1px: 0.001163, 3px: 0.010398, 5px: 0.028947
# Validation EPE: 35.703143, 1px: 0.001212, 3px: 0.010986, 5px: 0.030944
# Validation EPE: 35.139438, 1px: 0.001319, 3px: 0.011811, 5px: 0.032028
# Validation EPE: 42.423754, 1px: 0.001385, 3px: 0.012286, 5px: 0.033491
# Validation EPE: 33.235156, 1px: 0.001242, 3px: 0.011352, 5px: 0.031282


# <stage2-dpedcocomega> bestpath_43ep
# Validation EPE: 23.650886, 1px: 0.001147, 3px: 0.011180, 5px: 0.030892
# Validation EPE: 29.737732, 1px: 0.001146, 3px: 0.010298, 5px: 0.028723
# Validation EPE: 36.652434, 1px: 0.001258, 3px: 0.011335, 5px: 0.031372
# Validation EPE: 35.465480, 1px: 0.001293, 3px: 0.011604, 5px: 0.031704
# Validation EPE: 41.429356, 1px: 0.001338, 3px: 0.011917, 5px: 0.032436
# Validation EPE: 33.387178, 1px: 0.001229, 3px: 0.011224, 5px: 0.030932

# <stage2-dpedcocomega> bestpath_ep25_contfrom23ep
# /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static_multigpu_2stage/pho0,1,2,3,4,5_TRAIN_stage2_dpedcocomega_img224_bs18_lr1e5_crocov2_camap_freezeCrocoEnc_beta2e2_uncertainty_continuefrom23ep/CroCoNet_ep0025.pth.tar



# RE - stage2-1e-4 - 1epoch
Validation EPE: 19.587748, 1px: 0.001996, 3px: 0.018298, 5px: 0.050287
Validation EPE: 27.552841, 1px: 0.001599, 3px: 0.014880, 5px: 0.041930
Validation EPE: 33.260501, 1px: 0.001669, 3px: 0.015035, 5px: 0.041779
Validation EPE: 37.717423, 1px: 0.001698, 3px: 0.015267, 5px: 0.041396
Validation EPE: 43.487108, 1px: 0.001567, 3px: 0.014448, 5px: 0.039516
Validation EPE: 32.321124, 1px: 0.001716, 3px: 0.015690, 5px: 0.043298

# RE - stage2-1e-4 - 10epoch
Validation EPE: 20.347487, 1px: 0.001791, 3px: 0.015981, 5px: 0.043171
Validation EPE: 27.431611, 1px: 0.001460, 3px: 0.013705, 5px: 0.038006
Validation EPE: 35.826633, 1px: 0.001491, 3px: 0.013563, 5px: 0.037501
Validation EPE: 38.022640, 1px: 0.001394, 3px: 0.013394, 5px: 0.036693
Validation EPE: 44.687395, 1px: 0.001431, 3px: 0.013001, 5px: 0.035725
Validation EPE: 33.263153, 1px: 0.001524, 3px: 0.014018, 5px: 0.038452 

# RE - stage2-1e-4 - 20epoch
Validation EPE: 20.342165, 1px: 0.001758, 3px: 0.016364, 5px: 0.045436
Validation EPE: 26.499959, 1px: 0.001586, 3px: 0.014397, 5px: 0.039862
Validation EPE: 37.771618, 1px: 0.001504, 3px: 0.013875, 5px: 0.038397
Validation EPE: 36.975225, 1px: 0.001508, 3px: 0.014107, 5px: 0.038101
Validation EPE: 46.772801, 1px: 0.001502, 3px: 0.013409, 5px: 0.036452
Validation EPE: 33.672354, 1px: 0.001580, 3px: 0.014520, 5px: 0.039935

# RE - stage2-1e-4 - 30epoch
Validation EPE: 20.598747, 1px: 0.001850, 3px: 0.014904, 5px: 0.040892
Validation EPE: 27.038537, 1px: 0.001428, 3px: 0.013002, 5px: 0.035942
Validation EPE: 37.783904, 1px: 0.001414, 3px: 0.012931, 5px: 0.035678
Validation EPE: 37.987685, 1px: 0.001415, 3px: 0.012838, 5px: 0.035591
Validation EPE: 41.496832, 1px: 0.001517, 3px: 0.013433, 5px: 0.035957
Validation EPE: 32.981141, 1px: 0.001534, 3px: 0.013464, 5px: 0.036956

# # RE - stage2-1e-4 - 40epoch
Validation EPE: 20.546634, 1px: 0.001816, 3px: 0.015977, 5px: 0.044593
Validation EPE: 26.048547, 1px: 0.001568, 3px: 0.013949, 5px: 0.038790
Validation EPE: 38.485141, 1px: 0.001504, 3px: 0.013566, 5px: 0.037243
Validation EPE: 40.581358, 1px: 0.001465, 3px: 0.013291, 5px: 0.037014
Validation EPE: 45.328661, 1px: 0.001589, 3px: 0.013624, 5px: 0.036683
Validation EPE: 34.198068, 1px: 0.001596, 3px: 0.014158, 5px: 0.039118


# RE - stage2-1e-4 - 50epoch
Validation EPE: 20.428958, 1px: 0.002107, 3px: 0.017138, 5px: 0.045989
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:38<00:00,  1.52it/s]
Validation EPE: 26.614834, 1px: 0.001646, 3px: 0.014494, 5px: 0.040180
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:38<00:00,  1.55it/s]
Validation EPE: 38.681297, 1px: 0.001580, 3px: 0.014289, 5px: 0.038959
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:38<00:00,  1.54it/s]
Validation EPE: 39.464159, 1px: 0.001589, 3px: 0.014235, 5px: 0.038906
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [00:37<00:00,  1.57it/s]
Validation EPE: 45.633151, 1px: 0.001583, 3px: 0.014034, 5px: 0.038084
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 295/295 [01:02<00:00,  4.73it/s]
Validation EPE: 34.164480, 1px: 0.001717, 3px: 0.014930, 5px: 0.040670



# RE - stage2-1e-4 - 1epoch
Validation EPE: 32.321124, 1px: 0.001716, 3px: 0.015690, 5px: 0.043298

# RE - stage2-1e-4 - 10epoch
Validation EPE: 33.263153, 1px: 0.001524, 3px: 0.014018, 5px: 0.038452 

# RE - stage2-1e-4 - 20epoch
Validation EPE: 33.672354, 1px: 0.001580, 3px: 0.014520, 5px: 0.039935

# RE - stage2-1e-4 - 30epoch
Validation EPE: 32.981141, 1px: 0.001534, 3px: 0.013464, 5px: 0.036956

# # RE - stage2-1e-4 - 40epoch
Validation EPE: 34.198068, 1px: 0.001596, 3px: 0.014158, 5px: 0.039118

# RE - stage2-1e-4 - 50epoch
Validation EPE: 34.164480, 1px: 0.001717, 3px: 0.014930, 5px: 0.040670


