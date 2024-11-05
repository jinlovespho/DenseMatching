#!/bin/bash


## 현재 제일 잘 나온 모델
## Validation EPE: 4.867763, 1px: 0.750827, 3px: 0.936564, 5px: 0.962140
CUDA=3
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-224 \
    --eval_img_size 224 224 \
    --model croco_catseg \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057.pth.tar \
    --output_flow_interp \
    --output_ca_map \
    --softmax_camap \
    --correlation \
    --reciprocity \
    --save_dir ./vis/eval/hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp224_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057_fineflow \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plot_individual_images \


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





