#!/bin/bash



CUDA=3

# lr 1e-3
# /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1 
# lr 1e-4
# /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1

for epoch in 20; do 

    CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
        --seed 1997 \
        --dataset hp \
        --model_img_size 224 224 \
        --model croco_catseg \
        --pre_trained_models croco \
        --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
        --path_to_pre_trained_models /home/cvlab08/projects/data/jinlovespho/dm_final/train_settings/croco/train_croco_static_stage1_multigpu/server8_pho0,1,2_TRAIN_CVPR2025REBUTTAL_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1/CroCoNet_ep00${epoch}.pth.tar \
        --output_flow_interp \
        --output_ca_map \
        --softmax_camap \
        --correlation \
        --reciprocity \
        --save_dir ./vis/eval/hp_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep00${epoch} \
        --log_tool wandb \
        --wandb_path ./ \
        --wandb_proj_name matching_dped \
        --wandb_exp_name server8_pho${CUDA}_EVAL_CVPR2025REBUTTAL_hp_dpedmsk_img224_bs12_lr1e3_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep00${epoch} \
        --wandb_log_img \

done

# ep(57) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057.pth.tar
# Validation EPE: 4.867763, 1px: 0.750827, 3px: 0.936564, 5px: 0.962140

# (ep63) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0063.pth.tar
# Validation EPE: 4.926181, 1px: 0.749193, 3px: 0.934810, 5px: 0.960156

# (ep64) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0064.pth.tar
# Validation EPE: 4.923685, 1px: 0.745711, 3px: 0.937248, 5px: 0.962999

# (ep80) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0080.pth.tar
# Validation EPE: 4.744846, 1px: 0.789866, 3px: 0.943540, 5px: 0.964811

# (ep100) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0100.pth.tar
# Validation EPE: 4.754663, 1px: 0.789996, 3px: 0.943419, 5px: 0.964041
# zoom234 - Validation EPE: 14.698702, 1px: 0.472491, 3px: 0.833618, 5px: 0.911935
# zoom345 - Validation EPE: 14.950584, 1px: 0.490877, 3px: 0.833727, 5px: 0.909261

# (model_best) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_model_best.pth.tar
# Validation EPE: 4.761091, 1px: 0.792701, 3px: 0.943957, 5px: 0.964300
# zoom234 - Validation EPE: 14.699932, 1px: 0.477581, 3px: 0.834174, 5px: 0.912208
# zoom345 - Validation EPE: 14.984115, 1px: 0.493292, 3px: 0.834593, 5px: 0.910013



# ## 현재 제일 잘 나온 모델 (ep57)
# ## Validation EPE: 4.867763, 1px: 0.750827, 3px: 0.936564, 5px: 0.962140
# CUDA=3
# CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
#     --seed 1997 \
#     --dataset hp-224 \
#     --eval_img_size 224 224 \
#     --model croco_catseg \
#     --pre_trained_models croco \
#     --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
#     --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057.pth.tar \
#     --output_flow_interp \
#     --output_ca_map \
#     --softmax_camap \
#     --correlation \
#     --reciprocity \
#     --save_dir ./vis/eval/hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
#     --log_tool wandb \
#     --wandb_path ./ \
#     --wandb_proj_name matching_dped \
#     --wandb_exp_name pho_EVAL_hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
#     --wandb_log_img \
#     # --compute_metrics_uncertainty \
#     # --plot \
#     # --plot_100 \
#     # --plot_individual_images \


# 홍규님껄로 돌려본 결과
# Validation EPE: 17.412396, 1px: 0.010136, 3px: 0.076416, 5px: 0.170543
# CUDA=3
# CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
#     --seed 1997 \
#     --dataset hp-224 \
#     --eval_img_size 224 224 \
#     --model croco_catseg \
#     --pre_trained_models croco \
#     --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
#     --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho2_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_CroCoNet_ep0057.pth.tar \
#     --output_flow_interp \
#     --output_ca_map \
#     --softmax_camap \
#     --correlation \
#     --reciprocity \
#     --save_dir ./vis/eval/hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_CroCoNet_ep0057_fineflow \
#     --log_tool wandb \
#     --wandb_path ./ \
#     --wandb_proj_name matching_dped \
#     --wandb_exp_name pho_EVAL_hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_CroCoNet_ep0057_fineflow \
#     --wandb_log_img \
#     # --compute_metrics_uncertainty \
#     # --plot \
#     # --plot_100 \
#     # --plot_individual_images \


## 현재 제일 잘 나온 모델
## Validation EPE: 4.867763, 1px: 0.750827, 3px: 0.936564, 5px: 0.962140
# CUDA=3
# CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
#     --seed 1997 \
#     --dataset hp-224 \
#     --eval_img_size 224 224 \
#     --model croco_catseg \
#     --pre_trained_models croco \
#     --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
#     --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057.pth.tar \
#     --output_flow_interp \
#     --output_ca_map \
#     --softmax_camap \
#     --correlation \
#     --reciprocity \
#     --save_dir ./vis/eval/hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
#     --log_tool wandb \
#     --wandb_path ./ \
#     --wandb_proj_name matching_dped \
#     --wandb_exp_name pho_EVAL_hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
#     --wandb_log_img \
#     # --compute_metrics_uncertainty \
#     # --plot \
#     # --plot_100 \
#     # --plot_individual_images \


# 1. For evaluating full fine tuned crocoflow on dpedmsk
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho4_TRAIN_dpedmsk_img224_bs14_lr2e5_crocoflow_baseline_fullfinetuning/CroCoDownstreamBinocular_ep0003.pth.tar \

# 2. For evaluating  fine tuned crocoflow(frozen croco encoder) on dpedmsk
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_static/pho5_TRAIN_dpedmsk_img224_bs14_lr2e5_crocoflow_baseline_freezeCrocoEnc/CroCoDownstreamBinocular_ep0003.pth.tar \


# 3. catseg_freezeCrocoAll_CroCoNet_ep0027
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho2_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_CroCoNet_ep0027.pth.tar \


# 4. catseg_freezeCrocoAll_try1_CroCoNet_ep0027
# --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0027.pth.tar \



