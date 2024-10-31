#!/bin/bash
CUDA=7
BATCH_SIZE=10

# try1
# changed from  self.block1(swin) -> self.attn_multi1 -> self.block2(swin) -> self.attn_multi2
# to self.block1(swin) -> self.block2(swin) -> self.attn_multi

# try2
# changed cats_swin.py: line 580
# JLP
# decfeats_tgt = [encfeat_last_tgt, encfeat_last_tgt] + decfeats_tgt[1:]
# decfeats_src = [encfeat_last_src, encfeat_last_src] + decfeats_src[1:]

CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_dynamic_cats' \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_${CUDA}_dpedcoco_applymsk_img224_bs${BATCH_SIZE}_croco_cats_swin_try2 \
    --img_size 224 224 \
    --batch_size ${BATCH_SIZE} \
    --softmaxattn \
    --reciprocity \
    --cost_agg cats_swin \
    --cost_transformer \
    --correlation \
    --apply_coco_msk \
#  --multi_gpu \



# CUDA_VISIBLE_DEVICES=${CUDA} python run_training.py 'croco' 'train_croco_static_cats' \
#  --tag hierarchical_conv4d_cats_level_4stage_again \
#  --img_size 224 224 \
#  --softmaxattn \
#  --reciprocity \
#  --cost_agg hierarchical_conv4d_cats_level_4stage \
#  --cost_transformer \
#  --correlation \
#  --hierarchical \
#  --batch_size 1 \
#  --cats_depth 2