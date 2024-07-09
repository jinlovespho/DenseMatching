CUDA_VISIBLE_DEVICES=0  python eval_matching.py     --dataset hp \
                                                    --model GLUNet_GOCor \
                                                    --pre_trained_models static \
                                                    --optim_iter 3 \
                                                    --local_optim_iter 7 \
                                                    --save_dir ./save_dir/ \
                                                    --path_to_pre_trained_models /home/cvlab05/project/jinlovespho/github/monodepth/DenseMatching/pre_trained_models/GLUNet_GOCor_static.pth \