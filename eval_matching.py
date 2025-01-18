import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import datasets
import json
import wandb

from utils_data.image_transforms import ArrayToTensor
from validation.flow_evaluation.evaluate_per_dataset import run_evaluation_generic, run_evaluation_eth3d, run_evaluation_semantic, run_evaluation_semantic_dift, run_evaluation_semantic_joint
from model_selection import select_model
import admin.settings as ws_settings
from admin.stats import merge_dictionaries

def main(args, settings):
    
    # set up log tool
    if args.log_tool == 'wandb':
        wandb.init( project = args.wandb_proj_name,
                    name = args.wandb_exp_name,
                    config = args,
                    dir=args.wandb_path)
        
    # image transformations for the dataset
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_to_save = args.save_dir.split('/')[-1]
    save_dict = {}
    path_to_save = name_to_save
    # define the network to use
    network, estimate_uncertainty = select_model(args.model, args.path_to_pre_trained_models, args)
    if args.dense_zoom_in:
        print('DENSE_ZOOM_IN_RATIO: ', args.dense_zoom_ratio)
    print('CROCO_CKPT: ', args.croco_ckpt)
    print('OUTPUT_CORRELATION: ', args.output_correlation)
    print('SOFT_ARGMAX_BETA: ', args.softargmax_beta)

    if args.dataset == 'hp':
        if args.eval_img_size is None:
            original_size = True
            print('Hpatches Eval Img Size: ', 'Original size')
        else:
            original_size = False
            print('Hpatches Eval Img Size: ', args.eval_img_size)
        number_of_scenes = 5 + 1
        list_of_outputs = []
        # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
        for id, k in enumerate(range(2, number_of_scenes + 2)):
            if id == 5:
                _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                        os.path.join('assets', 'hpatches_all.csv'.format(k)),
                                                        input_transform, target_transform, co_transform,
                                                        use_original_size=original_size, split=0,
                                                        image_size=args.eval_img_size)   # JLP eval_img_size
            else:
                _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                        os.path.join('assets', 'hpatches_1_{}.csv'.format(k)),
                                                        input_transform, target_transform, co_transform,
                                                        use_original_size=original_size, split=0,
                                                        image_size=args.eval_img_size)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output_scene = run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty, curr_id=id, args=args)
            list_of_outputs.append(output_scene)

        output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2], 'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[5]}

    elif args.dataset == 'eth3d':
        output = run_evaluation_eth3d(network, settings.env.eth3d, input_transform, target_transform, co_transform, device, estimate_uncertainty=estimate_uncertainty, args=args)
    
    elif args.dataset == 'TSS':
        output = {}
        for sub_data in ['FG3DCar', 'JODS', 'PASCAL']:
            print('TSS evaluating on subdata: ', sub_data)
            path_to_save_ = os.path.join(path_to_save, sub_data)
            if not os.path.exists(path_to_save_) and (args.plot or args.plot_100):
                os.makedirs(path_to_save_)
            test_set = datasets.TSSDataset(os.path.join(settings.env.tss, sub_data),
                                            source_image_transform=input_transform,
                                            target_image_transform=input_transform, flow_transform=target_transform,
                                            co_transform=co_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            results = run_evaluation_semantic(network, test_dataloader, device,
                                                estimate_uncertainty=estimate_uncertainty,
                                                flipping_condition=args.flipping_condition,
                                                path_to_save=path_to_save_, plot=args.plot, plot_100=args.plot_100,
                                                plot_ind_images=args.plot_individual_images, tss_subdata=sub_data, args=args)
            output[sub_data] = results

    elif args.dataset == 'PFPascal':
        test_set = datasets.PFPascalDataset(settings.env.PFPascal, source_image_transform=input_transform,
                                            target_image_transform=input_transform, split='test',
                                            flow_transform=target_transform)
        test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
        output = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                            flipping_condition=args.flipping_condition,
                                            path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                            plot_ind_images=args.plot_individual_images, args=args)

    elif args.dataset == 'PFWillow':
        test_set = datasets.PFWillowDataset(settings.env.PFWillow, source_image_transform=input_transform,
                                            target_image_transform=input_transform, split='test',
                                            flow_transform=target_transform)
        test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
        output = run_evaluation_semantic(network, test_dataloader, device,
                                            estimate_uncertainty=estimate_uncertainty,
                                            flipping_condition=args.flipping_condition,
                                            path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                            plot_ind_images=args.plot_individual_images, args=args)
    elif args.dataset == 'spair':
        if args.model == 'dift_sd' or args.model == 'sd3_single' or args.model == 'dit_single' or args.model == 'cogvid_single':
            output = run_evaluation_semantic_dift(network=network, dataset_path=settings.env.spair, args=args)
        elif args.model == 'sd3_joint':
            output = run_evaluation_semantic_joint(network=network, dataset_path=settings.env.spair, args=args)
        else:
            test_set = datasets.SPairDataset(settings.env.spair, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)
            output = run_evaluation_semantic(network, test_dataloader, device,
                                                estimate_uncertainty=estimate_uncertainty,
                                                flipping_condition=args.flipping_condition,
                                                path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                                plot_ind_images=args.plot_individual_images, args=args)
    
    else:
        raise ValueError('Unknown dataset, {}'.format(args.dataset))

    if output is None:
        return 0
    
    else:
        save_dict[f'{name_to_save}'] = output
        name_save_metrics = 'metrics_{}'.format(name_to_save)

        path_file = '{}/{}.txt'.format(save_dir, name_save_metrics)
        if os.path.exists(path_file):
            with open(path_file, 'r') as outfile:
                save_dict_existing = json.load(outfile)
            save_dict = merge_dictionaries([save_dict_existing, save_dict])

        with open(path_file, 'w') as outfile:
            json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
            print('written to file ')
        
        if os.path.exists(args.feat_save_path) and output[f'per_image_pck@0.1']['aeroplane'] < 70.0:
            print(f'DELETING FEATURES FOR {name_to_save}')
            del_files = os.listdir(args.feat_save_path)
            for file in del_files:
                if file.endswith('.pth'):
                    os.remove(os.path.join(args.feat_save_path, file))
        print(f'FINISHED EXPERIMENT: {name_to_save}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)

    parser.add_argument('--log_warped_images', action='store_true', help='log warped images? default is False')
    parser.add_argument('--save_dir', type=str )
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--path_to_pre_trained_models', type=str, default=None)

    parser.add_argument('--croco_ckpt', type=str, help='path to pretrained crocoflow checkpoint')
    parser.add_argument('--eval_img_size', nargs='+', type=int, help='evaluation image size')

    # model
    parser.add_argument('--output_flow_interp', action='store_true', help='output flow interpolation? default is False')
    parser.add_argument('--output_ca_map', action='store_true', help='output confidence map? default is False')
    parser.add_argument('--softmax_camap', action='store_true', help='softmax confidence map? default is False')
    parser.add_argument('--correlation', action='store_true', help='compute correlation? default is False')
    parser.add_argument('--reciprocity', action='store_true', help='compute reciprocity? default is False')
    parser.add_argument('--uncertainty', action='store_true', help='compute uncertainty? default is False')
    parser.add_argument('--model_img_size', nargs='+', type=int, help='model image size')
    parser.add_argument('--heuristic_attn_map_refine', action='store_true', help='heuristic attention map refine? default is False')
    parser.add_argument('--output_correlation', type=str, help='correlation type')
    parser.add_argument('--softargmax_beta', type=float, help='softargmax beta')
    parser.add_argument('--lora_dec', action='store_true', help='lora decoder? default is False')
    parser.add_argument('--lora_dec_rank', type=int, help='lora decoder rank')

    parser.add_argument('--flipping_condition', action='store_true', help='flipping condition? default is False')

    # inference 
    parser.add_argument('--dense_zoom_in', action='store_true', help='dense zoom in default is False')
    parser.add_argument('--dense_zoom_ratio', nargs='+', type=int, help='dense zoom ratio')

    parser.add_argument('--compute_metrics_uncertainty', action='store_true', help='compute metrics uncertainty? default is False')
    parser.add_argument('--plot', action='store_true', help='plot? default is False')
    parser.add_argument('--plot_100', action='store_true', help='plot 100 first images? default is False')
    parser.add_argument('--plot_individual_images', action='store_true', help='plot individual images? default is False')
    
    # log_args
    parser.add_argument('--log_tool', type=str, default=None)
    parser.add_argument('--wandb_path', type=str, default=None)
    parser.add_argument('--wandb_proj_name', type=str, default=None)
    parser.add_argument('--wandb_exp_name', type=str, default='no_tag_assigned')
    
    # dift args 
    parser.add_argument('--feat_save_path', type=str, default='./extracted_feats', help='path to save features')
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--feat_already_extracted', action='store_true')

    
    parser.add_argument('--joint_full_attn', action='store_true')
    parser.add_argument('--CONCAT_WIDTH', action='store_true')
    parser.add_argument('--ACTUALLY_SINGLE', action='store_true')
    parser.add_argument('--FLOW_CAMAP', action='store_true')
    
    parser.add_argument('--inf_max_step', type=int, default=28, help='max steps for inference')
    parser.add_argument('--inf_stop_step', type=int, default=25, help='stop step for inference')
    
    parser.add_argument('--output_feat_type', type=str, default='query', help='output feature type')
    parser.add_argument('--output_layer', type=int, default=0, help='output layer for sd3')
    
    # VIS_ARGS 
    parser.add_argument('--WANDB_LOG_FREQ', type=int, default=50, help='wandb log frequency')
    parser.add_argument('--VIS_PCA_SINGLE_IMG', action='store_true')
    parser.add_argument('--VIS_PCA_JOINT_IMG', action='store_true')
    parser.add_argument('--VIS_KPTS_PREDICTION', action='store_true')
    
    # VIS_ATTN_MAP
    parser.add_argument('--VIS_ATTN_MAP', action='store_true')
    parser.add_argument('--VIS_ATTN_SRC_TO_TRG', action='store_true')
    parser.add_argument('--VIS_ATTN_TRG_TO_SRC', action='store_true')
    parser.add_argument('--VIS_ATTN_SRC_TO_SRC', action='store_true')
    parser.add_argument('--VIS_ATTN_TRG_TO_TRG', action='store_true')
    
    # ETC_ARGS
    parser.add_argument('--EVAL_SAMPLE_NUM', type=int, default=-1, help='evaluation sample number')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu

    # settings containing paths to datasets
    settings = ws_settings.Settings()
    main(args=args, settings=settings)
