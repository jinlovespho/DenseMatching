#!/bin/bash

# CUDA=7
# CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
#     --seed 1997 \
#     --dataset hp \
#     --model_img_size 224 224 \
#     --model glunet \
#     --pre_trained_models croco \
#     --croco_ckpt ./pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
#     --output_mode ca_map \
#     --save_dir ./vis/eval/hp_suppl_zeroshot_dust3r_512_dpt \
#     --log_tool wandba \
#     --wandb_path ./ \
#     --wandb_proj_name matching_dped \
#     --wandb_exp_name pho_SUPPL_hp_zeroshot_dust3r_512_dpt \
#     --wandb_log_img \
#     # --compute_metrics_uncertainty \
#     # --plot \
#     # --plot_100 \
#     # --plot_individual_images \

CUDA=7
CUDA_VISIBLE_DEVICES=${CUDA} python eval_matching.py    --dataset hp \
                                                        --model PDCNet_plus \
                                                        --pre_trained_models dynamic \
                                                        --optim_iter 3 \
                                                        --local_optim_iter 7 \
                                                        --save_dir ./vis/suppl/hp/ \
                                                        --path_to_pre_trained_models ./pretrained_weights/PDCNet_plus_megadepth.pth.tar