CUDA_VISIBLE_DEVICES=4 python run_training.py PDCNet train_PDCNet_plus_stage1 \
    --log_tool wandb \
    --wandb_path ./ \
    --wandb_proj_name matching_dped \
    --tag pho_gpu4_dpedcoco_img520_bs10_pdcnetplus_baseline \
    --img_size 520 520 \
    --batch_size 10 \
    # --multi_gpu