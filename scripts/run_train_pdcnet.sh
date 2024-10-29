CUDA_VISIBLE_DEVICES=6 python run_training.py PDCNet train_PDCNet_stage1 \
    --tag pho_gpu4_testasdf \
    --img_size 520 520 \
    --batch_size 16 \
    # --multi_gpu