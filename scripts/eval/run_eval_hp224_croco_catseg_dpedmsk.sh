#!/bin/bash

# ep(57) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0057.pth.tar
# hp224 - Validation EPE: 4.867763, 1px: 0.750827, 3px: 0.936564, 5px: 0.962140
# eth3d - zoom23 - 


# (ep63) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0063.pth.tar
# hp224 - Validation EPE: 4.926181, 1px: 0.749193, 3px: 0.934810, 5px: 0.960156

# (ep64) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0064.pth.tar
# hp224 - Validation EPE: 4.923685, 1px: 0.745711, 3px: 0.937248, 5px: 0.962999

# (ep80) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0080.pth.tar
# hp224 - Validation EPE: 4.744846, 1px: 0.789866, 3px: 0.943540, 5px: 0.964811
# eth3d - zoom23

# (ep100) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_ep0100.pth.tar
# hp224 - Validation EPE: 4.754663, 1px: 0.789996, 3px: 0.943419, 5px: 0.964041
# hp - zoom234 - Validation EPE: 14.698702, 1px: 0.472491, 3px: 0.833618, 5px: 0.911935
# hp - zoom345 - Validation EPE: 14.950584, 1px: 0.490877, 3px: 0.833727, 5px: 0.909261
# eth3d - zoom

# (model_best) - /media/dataset3/jinlovespho/ckpt/server8/dm_final/server8_pho3_TRAIN_dpedmsk_img224_bs12_lr1e4_croco_catseg_freezeCrocoAll_try1_CroCoNet_model_best.pth.tar
# hp224 - Validation EPE: 4.761091, 1px: 0.792701, 3px: 0.943957, 5px: 0.964300
# hp -zoom234 - Validation EPE: 14.699932, 1px: 0.477581, 3px: 0.834174, 5px: 0.912208
# hp -zoom345 - Validation EPE: 14.984115, 1px: 0.493292, 3px: 0.834593, 5px: 0.910013



# <stage2_freezeCrocoEnc>
    # (ep15) - /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_dynamic_stage2/pho2,3,4,5_TRAIN_stage2_dpedcocomega_img224_bs8_lr5e5_croco_catseg_freezeCrocoEnc_uncertainty_newweight/CroCoNet_ep0015.pth.tar
        # hp224 - 
        # hp - zoom23 - Validation EPE: 17.144649, 1px: 0.384079, 3px: 0.804783, 5px: 0.900036
        # eth3d - zoom

    # (ep30) - /media/dataset3/jinlovespho/ckpt/DenseMatching_final/train_settings/croco/train_croco_dynamic_stage2/pho2,3,4,5_TRAIN_stage2_dpedcocomega_img224_bs8_lr5e5_croco_catseg_freezeCrocoEnc_uncertainty_newweight/CroCoNet_ep0030.pth.tar
        # hp224 - 
        # hp - zoom23 - Validation EPE: 15.624423, 1px: 0.412170, 3px: 0.815895, 5px: 0.905636
        # eth3d - zoom

# <stage2_freezeAll>
    # (ep15) - /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_multigpu_2stage/hg0,1,6,7_TRAIN_dpedcocomega_img224_bs12_lr1e3_croco_catseg_uncertainty1e4_newweight/CroCoNet_ep0015.pth.tar
        # hp224 -
        # hp - zoom23 - Validation EPE: 43.310516, 1px: 0.001570, 3px: 0.013406, 5px: 0.035765
        # eth3d - zoom

    # (ep30) - /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_multigpu_2stage/hg0,1,6,7_TRAIN_dpedcocomega_img224_bs12_lr1e3_croco_catseg_uncertainty1e4_newweight/CroCoNet_ep0030.pth.tar
        # hp224 - 
        # hp - zoom23 - Validation EPE: 41.693769, 1px: 0.001253, 3px: 0.011057, 5px: 0.030490
        # eth3d - zoom
    # (ep45) - 
        # hp224 - 
        # hp - zoom23 - EPE: 34.080723, 1px: 0.001648, 3px: 0.016158, 5px: 0.047997


CUDA=5
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp-224 \
    --eval_img_size 224 224 \
    --model_img_size 224 224 \
    --model croco_catseg \
    --pre_trained_models croco \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --path_to_pre_trained_models /media/dataset3/honggyu_log/train_settings/croco/train_croco_static_multigpu_2stage/hg0,1,6,7_TRAIN_dpedcocomega_img224_bs12_lr1e3_croco_catseg_uncertainty1e4_newweight/CroCoNet_ep0015.pth.tar \
    --output_flow_interp \
    --output_ca_map \
    --softmax_camap \
    --correlation \
    --reciprocity \
    --uncertainty \
    --save_dir ./vis/eval/hp224_stage2_dpedcocomega_img224_bs12_lr1e3_croco_catseg_uncertainty1e4_newweight_CroCoNet_ep0015 \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --wandb_exp_name pho_EVAL_hp224_stage2_dpedcocomega_img224_bs12_lr1e3_croco_catseg_uncertainty1e4_newweight_CroCoNet_ep0015 \
    --wandb_log_img \
    # --compute_metrics_uncertainty \
    # --plot \
    # --plot_100 \
    # --plo



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





