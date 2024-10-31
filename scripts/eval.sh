#!/bin/bash

CUDA=5

# # Validation EPE: 6.095501, 1px: 0.562947, 3px: 0.873727, 5px: 0.917504, 8px: 0.939769, 16px: 0.960465
# # "all":{"AEPE":6.0955008190567215,"PCK_1_per_image":0.5046785742398381,"PCK_3_per_image":0.8229347365499885,"PCK_5_per_image":0.8753760952597199,"PCK_1_per_dataset":0.5629466549563371,"PCK_3_per_dataset":0.8737273805662708,"PCK_5_per_dataset":0.9175037260692569,"num_pixels_pck_1":5396316.0,"num_pixels_pck_3":8375410.0,"num_pixels_pck_5":8795043.0,"num_valid_corr":9585839}}}
# CUDA_VISIBLE_DEVICES=${CUDA} python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/dataset3/jinlovespho/ckpt/DenseMatching/eval \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching/train_settings/croco/train_croco_dynamic_cats__pho_gpu6_dpedcoco_img224_bs6_croco_hierarchical_conv4d_cats_level_4stage/CroCoNet_model_best.pth.tar \
#  --cost_agg hierarchical_conv4d_cats_level_4stage \
#  --cost_transformer \
#  --reciprocity \
#  --correlation \
#  --softmaxattn \
#  --cats_depth 4 \


## Validation EPE: 5.833508, 1px: 0.589472, 3px: 0.859884, 5px: 0.916256, 8px: 0.949385, 16px: 0.976500
## "all":{"AEPE":5.833508258855949,"PCK_1_per_image":0.5199283953878182,"PCK_3_per_image":0.794408842524557,"PCK_5_per_image":0.8591965006972205,"PCK_1_per_dataset":0.589472136972048,"PCK_3_per_dataset":0.8598843564971204,"PCK_5_per_dataset":0.9162564695693304,"num_pixels_pck_1":5650585.0,"num_pixels_pck_3":8242713.0,"num_pixels_pck_5":8783087.0,"num_valid_corr":9585839}}}
CUDA_VISIBLE_DEVICES=${CUDA} python -u vis_attn.py \
 --dataset hpatches \
 --model croco \
 --pre_trained_models croco \
 --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
 --save_dir /media/dataset3/jinlovespho/ckpt/DenseMatching/eval \
 --image_shape 224 224 \
 --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching/train_settings/croco/train_croco_static_cats__pho_gpu5_dped_img224_bs2_croco_hierarchical_conv4d_cats_level_4stage/CroCoNet_model_best.pth.tar \
 --cost_agg hierarchical_conv4d_cats_level_4stage \
 --cost_transformer \
 --reciprocity \
 --correlation \
 --softmaxattn \
 --cats_depth 4 \



## "all":{"AEPE":5.119423127376427,"PCK_1_per_image":0.7514371957668272,"PCK_3_per_image":0.8835462150002151,"PCK_5_per_image":0.9115146303770401,"PCK_1_per_dataset":0.8102755533448872,"PCK_3_per_dataset":0.9307224959651419,"PCK_5_per_dataset":0.9541762593759399,"num_pixels_pck_1":7767171.0,"num_pixels_pck_3":8921756.0,"num_pixels_pck_5":9146580.0,"num_valid_corr":9585839}}}
# CUDA_VISIBLE_DEVICES=${CUDA} python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/dataset3/jinlovespho/ckpt/DenseMatching/eval \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_cats__reciprocity_lr1e4_aftersoftmax_correlation/CroCoNet_model_best.pth.tar \
#  --cost_agg cats \
#  --cost_transformer \
#  --reciprocity \
#  --correlation \
#  --softmaxattn \



# CUDA_VISIBLE_DEVICES=3 python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/data1/hg_log/densematching \
#  --image_shape 224 224 \
#  --path_to_pre_trained_models /media/data1/hg_log/densematching/train_settings/croco/train_croco_static_cats__uncertainty/CroCoNet_model_best.pth.tar \
#  --softmaxattn \
#  --reciprocity \
#  --correlation \
#  --cost_agg cats \
#  --cost_transformer \
#  --uncertainty
 

#  CUDA_VISIBLE_DEVICES=3 python -u vis_attn.py \
#  --dataset hpatches \
#  --model croco \
#  --pre_trained_models croco \
#  --pretrain_croco_path ./CroCo_V2_ViTLarge_BaseDecoder.pth \
#  --save_dir /media/data1/hg_log/densematching \
#  --image_shape 224 224 \
#  --softmaxattn \
#  --cost_agg cats \
#  --cost_transformer

#  --path_to_pre_trained_models /media/data1/hg_log/densematching/train_settings/croco/train_croco_static_cats__reciprocity_lr1e4_aftersoftmax_correlation/CroCoNet_model_best.pth.tar \
#  --reciprocity \
#  --correlation \
 