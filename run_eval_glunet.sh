CUDA_VISIBLE_DEVICES=0 python -u eval_matching.py   --dataset hpatches \
                                                    --model GLUNet_GOCor \
                                                    --pre_trained_models static \
                                                    --optim_iter 3 \
                                                    --local_optim_iter 7 \
                                                    --save_dir /home/cvlab05/project/jinlovespho/github/monodepth/DenseMatching/pre_trained_models/GLUNet_GOCor_static.pth \
                                                    


                                                    # --path_to_pre_trained_models /home/cvlab05/project/jinlovespho/github/monodepth/pho_mfd4/MaskingDepth/pretrained_weights/crocoflow.pth \
                                                    # --save_dir /media/dataset1/jinlovespho/log/dense_matching/eval \
                                                    # --image_shape 384 320 \
                                                    # --batch_size 1 \

                                                    # --optim_iter 3 --local_optim_iter 7 --save_dir path_to_save_dir