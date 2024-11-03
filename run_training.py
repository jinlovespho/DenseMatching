import os
import argparse
import importlib
import cv2 as cv
import torch
import torch.backends.cudnn
import random
import numpy as np
from shutil import copyfile
from datetime import date

import admin.settings as ws_settings

# JLP
import wandb


def run_training(train_module, train_name, seed, cudnn_benchmark=True, args=None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    # dd/mm/YY
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    # show training args for experiment in a pretty format
    print(f'Training from:  {train_module}/{train_name}.py')
    print(f'Date: {d1}')
    print("-" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("-" * 50 + "\n")

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.copy_project_path = f'train_settings/{train_module}/{train_name}'
    settings.project_path = f'{settings.copy_project_path}/{args.wandb_exp_name}'
    settings.seed = seed

    # will save the checkpoints there

    save_dir = os.path.join(settings.env.workspace_dir, settings.project_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(settings.copy_project_path + '.py', os.path.join(save_dir, settings.script_name + '.py'))

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module.replace('/', '.'),
                                                                        train_name.replace('/', '.')))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings, args=args)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    

    # data_args
    parser.add_argument('--dataset', type=str, default='dped')
    parser.add_argument('--apply_coco_msk', action='store_true', help='Apply coco mask')

    # train_args
    parser.add_argument('--seed', type=int, default=1992, help='Pseudo-RNG seed')
    parser.add_argument('--img_size', nargs='+', type=int )
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epoch', type=int)

    # model_args
    parser.add_argument('--model', type=str, default='crocoflow')
    parser.add_argument('--croco_ckpt', type=str, default=None)
    parser.add_argument('--freeze', type=str, default='none')

    parser.add_argument('--output_flow_interp', action='store_true', help='Output flow interpolation')
    parser.add_argument('--output_ca_map', action='store_true', help='Output decoder cross attention map')
    parser.add_argument('--softmax_camap', action='store_true', help='apply Softmax to the output cross attention map')
    parser.add_argument('--correlation', action='store_true', help='Correlation')
    parser.add_argument('--reciprocity', action='store_true', help='Reciprocity')

    # log_args
    parser.add_argument('--log_tool', type=str, default=None)
    parser.add_argument('--wandb_path', type=str, default=None)
    parser.add_argument('--wandb_proj_name', type=str, default=None)
    parser.add_argument('--wandb_exp_name', type=str, default='no_tag_assigned')

    # etc_args
    parser.add_argument('--multi_gpu', action='store_true', help='Multi GPU')   # default is False

    args = parser.parse_args()

    # args.seed = random.randint(0, 3000000)
    # args.seed = torch.initial_seed() & (2 ** 32 - 1)
    print('Seed is {}'.format(args.seed))
    random.seed(int(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # wandb
    if args.log_tool == 'wandb':
        wandb.init( project = args.wandb_proj_name,
                    name = args.wandb_exp_name,
                    config = args,
                    dir=args.wandb_path)

    run_training(args.train_module, args.train_name, cudnn_benchmark=args.cudnn_benchmark, seed=args.seed, args=args)


if __name__ == '__main__':
    main()