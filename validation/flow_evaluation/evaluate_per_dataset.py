import numpy as np
import torch
import os
import json 
import sys 
import cv2 
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import wandb
from einops import rearrange
import gc

from validation.flow_evaluation.metrics_uncertainty import (compute_average_of_uncertainty_metrics, compute_aucs,
                                                            compute_uncertainty_per_image)
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from validation.plot import plot_sparse_keypoints, plot_flow_and_uncertainty, plot_individual_images
from .metrics_segmentation_matching import poly_str_to_mask, intersection_over_union, label_transfer_accuracy
from utils_flow.pixel_wise_mapping import warp

import torch.nn.functional as F
from torchvision.utils import save_image
import torch.nn as nn 
from models.modules.mod import unnormalise_and_convert_mapping_to_flow
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from validation.sd3_utils import prepare_spair, extract_and_save_feats, extract_and_save_feats_joint

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)
    
def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02, x_normal=None, y_normal=None):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    b,_,h,w = corr.size()
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def resize_images_to_min_resolution(min_size, img, x, y, stride_net=16):  # for consistency with RANSAC-Flow
    """
    Function that resizes the image according to the minsize, at the same time resize the x,y coordinate.
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here use exactly the same function that they used, for fair comparison. Even through the index_valid could
    also theoretically include the lower bound x = 0 or y = 0.
    """
    # Is is source image resized
    # Xs contains the keypoint x coordinate in source image
    # Ys contains the keypoints y coordinate in source image
    # valids is bool on wheter the keypoint is contained in the source image
    x = np.array(list(map(float, x.split(';')))).astype(np.float32)  # contains all the x coordinate
    y = np.array(list(map(float, y.split(';')))).astype(np.float32)

    w, h = img.size
    ratio = min(w / float(min_size), h / float(min_size))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // stride_net * stride_net, new_h // stride_net * stride_net

    ratioW, ratioH = new_w / float(w), new_h / float(h)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    x, y = x * ratioW, y * ratioH  # put coordinate in proper size after resizing the images
    index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)

    return img, x, y, index_valid


def run_evaluation_kitti(network, test_dataloader, device, estimate_uncertainty=False,
                         path_to_save=None, plot=False, plot_100=False, plot_ind_images=False, args=None):
    out_list, epe_list = [], []
    dict_list_uncertainties = {}
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']     # 1 3 376 1241
        target_img = mini_batch['target_image']     # 1 3 376 1241
        flow_gt = mini_batch['flow_map'].to(device)  # 1 2 376 1241
        mask_valid = mini_batch['correspondence_mask'].to(device)   # 1 376 1241
        
        breakpoint()
        source_img = source_img.float().to(device)
        target_img = target_img.float().to(device)
        _, _, orig_H, orig_W = source_img.shape

        if args.model == 'crocov2':
            # in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            # in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)

            img_s = source_img.clone().float() / 255.0
            img_t = target_img.clone().float() / 255.0

            # img_s = (img_s - in1k_mean) / in1k_std
            # img_t = (img_t - in1k_mean) / in1k_std

            img_s = img_s.to(device)
            img_t = img_t.to(device) 

            if args.eval_img_size is not None:
                H, W = args.eval_img_size
                H_32, W_32 = (H//32)*32, (W//32)*32
                img_s = F.interpolate(img_s, size=(H_32, W_32), mode='bilinear', align_corners=False)
                img_t = F.interpolate(img_t, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            # mask_valid = F.interpolate(mask_valid.float().unsqueeze(0), size=(H_32, W_32), mode='nearest').squeeze(0).bool()
            # flow_gt_h,flow_gt_w = flow_gt.size(2),flow_gt.size(3)
            # flow_gt = F.interpolate(flow_gt, size=(H_32, W_32), mode='bilinear', align_corners=False).to(device)
            # flow_gt[:,0,:,:] *= W_32/flow_gt_w
            # flow_gt[:,1,:,:] *= H_32/flow_gt_h

        breakpoint()
        if args.dense_zoom_in:
            flow_est, uncertainty_est = network.zoom_in_batch(img_s, img_t, zoom_ratio=args.dense_zoom_ratio, optimize=False, homo_only=False, batch_size=1)
            print('dense zoom in')
            print('input img_shape: ', img_s.shape)
            print('output flow_est shape: ', flow_est.shape)
            print('gt_flow shape: ', flow_gt.shape)
        else:
            coarse_flow = network(img_t, img_s)    # b 2 14 14
            orig_size = (orig_H, orig_W)    # 600 800
            flow_est = F.interpolate(coarse_flow, size=orig_size, mode='bilinear', align_corners=False)
            flow_est[:, 0] *= float(orig_W) / float(14)
            flow_est[:, 1] *= float(orig_H) / float(14)
            print('no dense zoom in')
            print('input img_shape: ', img_s.shape)
            print('output coarse_flow shape: ', coarse_flow.shape)
            print('output upsampled flow_est shape: ', flow_est.shape)
            print('gt_flow shape: ', flow_gt.shape)

        save_image(img_s, f'kitti2012_img_s.jpg', normalize=True)
        save_image(img_t, f'kitti2012_img_t.jpg', normalize=True)
        save_image(mask_valid.float(), f'kitti2012_img_mask.jpg', normalize=True)
        warped_source_gt = warp(img_s, flow_gt.float())  
        save_image(warped_source_gt, f'kitti2012_img_warped_src_gt.jpg', normalize=True)

        breakpoint()

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            flow_est = network.estimate_flow(source_img, target_img)

        if plot or (plot_100 and i_batch < 100):
            plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                      flow_gt, flow_est, compute_rgb_flow=True)
        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe.mean().item())
        out_list.append(out.cpu().numpy())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)  # AEPE is per image, and then averaged over the dataset.
    fl = 100 * np.mean(out_list)  # fl is over the whole dataset
    print("Validation KITTI: aepe: %f, fl: %f" % (epe, fl))
    output = {'AEPE': epe, 'kitti-fl': fl}
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False, name_dataset=None, rate=None, curr_id=0, args=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        mask_valid_orig = mask_valid.clone()

        b, _, H_orig, W_orig = source_img.shape  

        source_img_orig = source_img.clone() / 255.0
        target_img_orig = target_img.clone() / 255.0

        source_img_orig = source_img_orig.to(device)
        target_img_orig = target_img_orig.to(device)

        # # check if the flow_gt is correctly resized
        # save_image(source_img, f'./img_src.jpg', normalize=True)
        # save_image(target_img, f'./img_tgt.jpg', normalize=True)
        # save_image(mask_valid.float(), f'./img_mask.jpg', normalize=True)
        # warped_source_gt = warp(source_img, flow_gt)  
        # save_image(warped_source_gt, f'./img_warped_src_gt.jpg', normalize=True)

        if args.dataset == 'eth3d' and args.eval_img_size is not None:
            eval_h, eval_w = args.eval_img_size
            # H_32, W_32 = (H//32)*32, (W//32)*32
            source_img = F.interpolate(source_img, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            target_img = F.interpolate(target_img, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            mask_valid = F.interpolate(mask_valid.float().unsqueeze(0), size=(eval_h, eval_w), mode='nearest').squeeze(0).bool().to(device)
            flow_gt_h,flow_gt_w = flow_gt.size(2),flow_gt.size(3)
            flow_gt = F.interpolate(flow_gt, size=(eval_h, eval_w), mode='bilinear', align_corners=True).to(device)
            flow_gt[:,0,:,:] *= eval_w/flow_gt_w
            flow_gt[:,1,:,:] *= eval_h/flow_gt_h   

        if args.model == 'crocov2':
            H_224, W_224 = args.model_img_size
            source_img = source_img / 255.0
            target_img = target_img / 255.0
            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std
            source_img = source_img.to(device)
            target_img = target_img.to(device)

            if args.dense_zoom_in:
                flow_est, uncertainty_est = network.zoom_in_batch(source_img, target_img, zoom_ratio=args.dense_zoom_ratio, batch_size=b)
            else:
                source_img = F.interpolate(source_img, size=(H_224, W_224), mode='bilinear', align_corners=False)   # b 3 224 224
                target_img = F.interpolate(target_img, size=(H_224, W_224), mode='bilinear', align_corners=False)
                output_correlation = args.output_correlation    # correlation: enc_feat, dec_feat, camap
                flow_est = network(target_img, source_img, output_correlation=output_correlation) # b 2 14 14

                if args.uncertainty:
                    flow_est = flow_est['flow_estimates'][0]
                    
                flow_est = F.interpolate(flow_est, size=(H_orig, W_orig), mode='bilinear', align_corners=False)  
                # corr_size = H_224//16
                flow_est[:,0,:,:] *= W_orig/224
                flow_est[:,1,:,:] *= H_orig/224


            if args.log_warped_images:
                if args.dataset == 'eth3d':
                    save_path = f'{args.save_dir}/rate{rate}'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    warped_source_gt = warp(source_img_orig, flow_gt)  
                    warped_source_est = warp(source_img_orig, flow_est)
                    save_image(source_img_orig, f'{save_path}/{name_dataset}_{i_batch}_img_src.jpg', normalize=True)
                    save_image(target_img_orig, f'{save_path}/{name_dataset}_{i_batch}_img_tgt.jpg', normalize=True)
                    save_image(warped_source_gt, f'{save_path}/{name_dataset}_{i_batch}_img_warped_src_gt.jpg', normalize=True)
                    save_image(warped_source_est, f'{save_path}/{name_dataset}_{i_batch}_img_warped_src_est.jpg', normalize=True)
            

                elif args.dataset == 'hp' and curr_id < 5:
                    save_path = f'{args.save_dir}/{curr_id}'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    warped_source_gt = warp(source_img_orig, flow_gt)  
                    warped_source_est = warp(source_img_orig, flow_est)
                    # Create a 2x2 grid of images
                    grid = torch.cat([
                        torch.cat([source_img_orig, target_img_orig], dim=3),
                        torch.cat([warped_source_gt*mask_valid, warped_source_est*mask_valid], dim=3)
                    ], dim=2)
                    
                    save_image(grid, f'{save_path}/{i_batch}_combined.jpg')

        elif args.model == 'add more models':
            pass 
        
        else:
            if estimate_uncertainty:
                flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)    # flow_est: 1 2 224 224, uncertainty_est: pdcnet 
            else:
                # source_img = source_img.float()/255.
                # target_img = target_img.float()/255.
                # source_img_orig = source_img.clone()
                # target_img_orig = target_img.clone()

                
                flow_est = network.estimate_flow(source_img, target_img)    

            save_path = f'./vis/tmp/hp/{args.model}/{curr_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            
            # vis_size = (H, W)
            # source_img_orig = F.interpolate(source_img_orig, size=vis_size, mode='bilinear', align_corners=True)
            # target_img_orig = F.interpolate(target_img_orig, size=vis_size, mode='bilinear', align_corners=True)
        
            # mask_valid_orig = F.interpolate(mask_valid_orig.float().unsqueeze(0), size=vis_size, mode='bilinear', align_corners=True)
            save_image(source_img, f'{save_path}/{i_batch}_img_src.jpg')
            save_image(target_img, f'{save_path}/{i_batch}_img_tgt.jpg')
            # save_image(mask_valid.float(), f'{save_path}/{curr_id}/{i_batch}_img_mask.jpg', normalize=True)

            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            # warped_source_gt = F.interpolate(warped_source_gt, size=vis_size, mode='bilinear', align_corners=True)
            # warped_source_est = F.interpolate(warped_source_est, size=vis_size, mode='bilinear', align_corners=True)
            save_image(warped_source_gt, f'{save_path}/{i_batch}_img_warped_src_gt.jpg')
            save_image(warped_source_est*mask_valid_orig, f'{save_path}/{i_batch}_img_warped_src_est.jpg')

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

    epe_all = np.concatenate(epe_all_list)
    pck1_dataset = np.mean(epe_all <= 1)
    pck3_dataset = np.mean(epe_all <= 3)
    pck5_dataset = np.mean(epe_all <= 5)
    output = {'AEPE': np.mean(mean_epe_list), 'PCK_1_per_image': np.mean(pck_1_list),
              'PCK_3_per_image': np.mean(pck_3_list), 'PCK_5_per_image': np.mean(pck_5_list),
              'PCK_1_per_dataset': pck1_dataset, 'PCK_3_per_dataset': pck3_dataset,
              'PCK_5_per_dataset': pck5_dataset, 'num_pixels_pck_1': np.sum(epe_all <= 1).astype(np.float64),
              'num_pixels_pck_3': np.sum(epe_all <= 3).astype(np.float64),
              'num_pixels_pck_5': np.sum(epe_all <= 5).astype(np.float64),
              'num_valid_corr': len(epe_all)
              }
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(mean_epe_list), pck1_dataset, pck3_dataset, pck5_dataset))

    return output


def run_evaluation_eth3d(network, data_dir, input_images_transform, gt_flow_transform, co_transform, device,
                         estimate_uncertainty, args=None):
    # ETH3D dataset information
    dataset_names = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 'delivery_area', 'electro',
                     'forest', 'playground', 'terrains']
    rates = list(range(3, 16, 2))
    dict_results = {}
    
    for rate in rates:
        print('Computing results for interval {}...'.format(rate))
        dict_results['rate_{}'.format(rate)] = {}
        list_of_outputs_per_rate = []
        num_pck_1 = 0.0
        num_pck_3 = 0.0
        num_pck_5 = 0.0
        num_valid_correspondences = 0.0
        for name_dataset in dataset_names:
            print('looking at dataset {}...'.format(name_dataset))

            test_set = ETHInterval(root=data_dir,
                                   path_list=os.path.join(data_dir, 'info_ETH3D_files',
                                                          '{}_every_5_rate_of_{}'.format(name_dataset, rate)),
                                   source_image_transform=input_images_transform,
                                   target_image_transform=input_images_transform,
                                   flow_transform=gt_flow_transform,
                                   co_transform=co_transform)  # only test
            test_dataloader = DataLoader(test_set,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)
            print(test_set.__len__())
            output = run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty, name_dataset=name_dataset, rate=rate, args=args)
            # to save the intermediate results
            # dict_results['rate_{}'.format(rate)][name_dataset] = output
            list_of_outputs_per_rate.append(output)
            num_pck_1 += output['num_pixels_pck_1']
            num_pck_3 += output['num_pixels_pck_3']
            num_pck_5 += output['num_pixels_pck_5']
            num_valid_correspondences += output['num_valid_corr']

        # average over all datasets for this particular rate of interval
        avg = {'AEPE': np.mean([list_of_outputs_per_rate[i]['AEPE'] for i in range(len(dataset_names))]),
               'PCK_1_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_1_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_3_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_3_per_image'] for i in
                                           range(len(dataset_names))]),
               'PCK_5_per_image': np.mean([list_of_outputs_per_rate[i]['PCK_5_per_image'] for i in
                                           range(len(dataset_names))]),
               'pck-1-per-rate': num_pck_1 / (num_valid_correspondences + 1e-6),
               'pck-3-per-rate': num_pck_3 / (num_valid_correspondences + 1e-6),
               'pck-5-per-rate': num_pck_5 / (num_valid_correspondences + 1e-6),
               'num_valid_corr': num_valid_correspondences
               }
        dict_results['rate_{}'.format(rate)] = avg

    avg_rates = {'AEPE': np.mean([dict_results['rate_{}'.format(rate)]['AEPE'] for rate in rates]),
                 'PCK_1_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_1_per_image'] for rate in rates]),
                 'PCK_3_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_3_per_image'] for rate in rates]),
                 'PCK_5_per_image': np.mean(
                     [dict_results['rate_{}'.format(rate)]['PCK_5_per_image'] for rate in rates]),
                 'pck-1-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-1-per-rate'] for rate in rates]),
                 'pck-3-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-3-per-rate'] for rate in rates]),
                 'pck-5-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-5-per-rate'] for rate in rates]),
                 }
    dict_results['avg'] = avg_rates
    print(f"eth3d - Validation EPE: {avg_rates['AEPE']}, rate_3_AEPE: {dict_results['rate_3']['AEPE']}, rate_5_AEPE: {dict_results['rate_5']['AEPE']}, rate_7_AEPE: {dict_results['rate_7']['AEPE']}, rate_9_AEPE: {dict_results['rate_9']['AEPE']}, rate_11_AEPE: {dict_results['rate_11']['AEPE']}, rate_13_AEPE: {dict_results['rate_13']['AEPE']}, rate_15_AEPE: {dict_results['rate_15']['AEPE']}")    
    return dict_results

def run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=False,
                            path_to_save=None, plot=False, plot_100=False, plot_ind_images=False, tss_subdata=None, args=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list = [], [], [], [], [], []
    dict_list_uncertainties = {}
    eval_buf = {'cls_pck': dict(), 'vpvar': dict(), 'scvar': dict(), 'trncn': dict(), 'occln': dict()}

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())
    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        cat = mini_batch['category'][0]

        b, c, H_orig, W_orig = source_img.shape

        breakpoint()
        
        source_img_orig = source_img.clone() / 255.0
        target_img_orig = target_img.clone() / 255.0

        source_img_orig = source_img_orig.to(device)
        target_img_orig = target_img_orig.to(device)

        source_img = source_img / 255.0
        target_img = target_img / 255.0

        in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        source_img = (source_img - in1k_mean) / in1k_std
        target_img = (target_img - in1k_mean) / in1k_std

        source_img = F.interpolate(source_img, size=(224, 224), mode='bilinear', align_corners=False)
        target_img = F.interpolate(target_img, size=(224, 224), mode='bilinear', align_corners=False)

        source_img = source_img.to(device)
        target_img = target_img.to(device)


        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()
        else:
            raise ValueError('No pck threshold in mini_batch')

        estimate_uncertainty=False
        uncertainty_est=None
        output_correlation = args.output_correlation    # correlation: enc_feat, dec_feat, camap
        flow_est = network(target_img, source_img, output_correlation=output_correlation) # b 2 14 14

        if args.uncertainty:
            flow_est = flow_est['flow_estimates'][0]

        flow_est = F.interpolate(flow_est, size=(H_orig, W_orig), mode='bilinear', align_corners=False)   # 1 2 240 240
        flow_est[:,0,:,:] *= W_orig/224
        flow_est[:,1,:,:] *= H_orig/224

        if args.log_warped_images:
            if args.dataset == 'TSS':
                save_path = f'{args.save_dir}/{tss_subdata}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                warped_source_gt = warp(source_img_orig, flow_gt)  
                warped_source_est = warp(source_img_orig, flow_est)
                # Create a 2x2 grid of images
                grid = torch.cat([
                    torch.cat([source_img_orig, target_img_orig], dim=3),
                    torch.cat([warped_source_gt*mask_valid, warped_source_est*mask_valid], dim=3)
                ], dim=2)

            else:
                save_path = f'{args.save_dir}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                warped_source_gt = warp(source_img_orig, flow_gt)  
                warped_source_est = warp(source_img_orig, flow_est)
                # Create a 2x2 grid of images
                grid = torch.cat([
                    torch.cat([source_img_orig, target_img_orig], dim=3),
                    torch.cat([warped_source_gt, warped_source_est], dim=3)
                ], dim=2)
                
            save_image(grid, f'{save_path}/{i_batch}_combined.jpg')

        plot_ind_images=False
        if plot_ind_images:
            source_img = source_img_orig * 255.
            target_img = target_img_orig * 255.
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        plot=False
        if plot or (plot_100 and i_batch < 100):
            source_img = source_img_orig * 255.
            target_img = target_img_orig * 255.
            if 'source_kps' in list(mini_batch.keys()):
                # I = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map_padded, R=1.0)
                plot_sparse_keypoints(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est,
                                      mini_batch['source_kps'][0][:, 0], mini_batch['source_kps'][0][:, 1],
                                      mini_batch['target_kps'][0][:, 0], mini_batch['target_kps'][0][:, 1],
                                      uncertainty_comp_est=uncertainty_est)
            else:
                plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_gt, flow_est)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_0_05_list.append(epe.le(0.05*L_pck).float().mean().item())
        pck_0_01_list.append(epe.le(0.01*L_pck).float().mean().item())
        pck_0_1_list.append(epe.le(0.1*L_pck).float().mean().item())
        pck_0_15_list.append(epe.le(0.15*L_pck).float().mean().item())
        for t in range(len(pck_thresholds)):
            pck_per_image_curve[t, i_batch] = epe.le(pck_thresholds[t]*L_pck).float().mean().item()

        if 'category' in mini_batch.keys():
            if eval_buf['cls_pck'].get(mini_batch['category'][0]) is None:
                eval_buf['cls_pck'][mini_batch['category'][0]] = []
            eval_buf['cls_pck'][mini_batch['category'][0]].append(epe.le(0.1 * L_pck).float().mean().item())

        if 'vpvar' in mini_batch.keys():
            for name in ['vpvar', 'scvar', 'trncn', 'occln']:
                # different difficulties
                # means it is spair
                if eval_buf[name].get('{}'.format(mini_batch[name][0])) is None:
                    eval_buf[name]['{}'.format(mini_batch[name][0])] = []
                eval_buf[name]['{}'.format(mini_batch[name][0])].append(epe.le(0.1 * L_pck).float().mean().item())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    breakpoint()
    epe_all = np.concatenate(epe_all_list)
    pck_0_05_dataset = np.mean(epe_all <= 0.05 * L_pck)
    pck_0_01_dataset = np.mean(epe_all <= 0.01 * L_pck)
    pck_0_1_dataset = np.mean(epe_all <= 0.1 * L_pck)
    pck_0_15_dataset = np.mean(epe_all <= 0.15 * L_pck)

    output = {'AEPE': np.mean(mean_epe_list), 'PCK_0_05_per_image': np.mean(pck_0_05_list),
              'PCK_0_01_per_image': np.mean(pck_0_01_list), 'PCK_0_1_per_image': np.mean(pck_0_1_list),
              'PCK_0_15_per_image': np.mean(pck_0_15_list),
              'PCK_0_01_per_dataset': pck_0_01_dataset, 'PCK_0_05_per_dataset': pck_0_05_dataset,
              'PCK_0_1_per_dataset': pck_0_1_dataset, 'PCK_0_15_per_dataset': pck_0_15_dataset,
              'pck_threshold_alpha': pck_thresholds, 'pck_curve_per_image': np.mean(pck_per_image_curve, axis=1).tolist()
              }
    print(f'Validation EPE: {output["AEPE"]:6f}, alpha=0.01: {output["PCK_0_01_per_image"]:6f}, alpha=0.05: {output["PCK_0_05_per_image"]:6f}, alpha=0.1: {output["PCK_0_1_per_image"]:6f}, alpha=0.15: {output["PCK_0_15_per_image"]:6f}')
    # print("Validation EPE: %f, alpha=0_01: %f, alpha=0.05: %f" % (output['AEPE'], output['PCK_0_01_per_image'],
    #                                                               output['PCK_0_05_per_image']))

    for name in eval_buf.keys():
        output[name] = {}
        for cls in eval_buf[name]:
            if eval_buf[name] is not None:
                cls_avg = sum(eval_buf[name][cls]) / len(eval_buf[name][cls])
                output[name][cls] = cls_avg

    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output

def print_exp_info(args):
    
    if args.model=='sd3_single' or args.model=='sd3_joint':
        # inf_max_step=28
        timesteps=[ 1000.0000,  987.3806,  974.1077,  960.1293,  945.3875,  929.8179,
                    913.3490,  895.9003,  877.3819,  857.6923,  836.7167,  814.3248,
                    790.3683,  764.6771,  737.0558,  707.2785,  675.0823,  640.1602,
                    602.1506,  560.6250,  515.0721,  464.8760,  409.2888,  347.3926,
                    278.0488,  199.8270,  110.9057,    8.9286]
    elif args.model=='dit_single':
        # inf_max_step=25 
        timesteps=[ 999, 959, 919, 879, 839, 799, 759, 719, 679, 639, 599, 559, 519, 480,
                    440, 400, 360, 320, 280, 240, 200, 160, 120,  80,  40]
    elif args.model=='cogvid_single':
        # inf_max_step=50
        timesteps=[ 999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739,
                    719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459,
                    439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179,
                    159, 139, 119,  99,  79,  59,  39,  19]
    
    print('='*50)
    print(f'Extracting {args.model} DIFT features!!!')
    
    print('-'*50)
    print(f'FEATURE SAVE PATH: {args.feat_save_path}')
    print(f'VIS SAVE DIR: {args.save_dir}')
    print(f'EVAL SAMPLE NUM: {args.EVAL_SAMPLE_NUM}')
    
    print('-'*50)
    if args.model == 'dift_sd':
        print(f'ENSEMBLE SIZE: {args.ensemble_size}')
        print(f'CURRENT TIMESTEP: {args.t}')
        print(f'UP SAMPLING BLOCK INDEX: {args.up_ft_index}')
        
    else:  
        print(f'INF_STOP_STEP: {args.inf_stop_step}/{args.inf_max_step}')
        print(f'CURRENT TIMESTEP: {timesteps[args.inf_stop_step]}')
        print(f'OUTPUT FEATURE TYPE: {args.output_feat_type}')
        print(f'OUTPUT LAYER: {args.output_layer}')
        print('-'*50)
        print(f'VIS_PCA_SINGLE_IMG: {args.VIS_PCA_SINGLE_IMG}')
        print(f'VIS_PCA_JOINT_IMG: {args.VIS_PCA_JOINT_IMG}')
        print(f'VIS_KPTS_PREDICTION: {args.VIS_KPTS_PREDICTION}')
        print('-'*50)
        print(f'VIS_ATTN_MAP: {args.VIS_ATTN_MAP}')
        print(f'VIS ATTN SRC TO TRG: {args.VIS_ATTN_SRC_TO_TRG}')
        print(f'VIS ATTN TRG TO SRC: {args.VIS_ATTN_TRG_TO_SRC}')
        print(f'VIS ATTN SRC TO SRC: {args.VIS_ATTN_SRC_TO_SRC}')
        print(f'VIS ATTN TRG TO TRG: {args.VIS_ATTN_TRG_TO_TRG}')
    print('='*50)


def run_evaluation_semantic_dift(network, dataset_path, args):
    
    all_cats, cat2json, cat2img = prepare_spair(dataset_path, args)
    
    if not args.feat_already_extracted:
        print_exp_info(args)
        extract_and_save_feats(network, dataset_path, all_cats, cat2img, args)
    else:
        print(f'{args.model} DIFT features already extracted')
    
    # Prepare evaluation
    output={}
    output[f'per_image_pck@0.1']={}
    output[f'per_point_pck@0.1']={}
    total_pck = []
    all_correct = 0
    all_total = 0
    
    '''
        all_cats: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json.keys(): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json['aeroplane']: ['000250-2008_008607-2009_002388:aeroplane.json', '000540-2010_004817-2008_008607:aeroplane.json', ... ]
        len(cat2json['aeroplane']): 690
    '''
    
    # remove model for inference to save GPU memory 
    if args.model == 'sd3_single':  
        network.pipe.to('cpu')
        del network
        gc.collect()
        torch.cuda.empty_cache() 
        
    for cat in all_cats:
        cat_list = cat2json[cat]
        output_dict = torch.load(os.path.join(args.feat_save_path, f'{cat}.pth'), weights_only=True)

        cat_pck = []
        cat_correct = 0
        cat_total = 0
        
        # Define interpolation size for PCA visualization
        interp_size = (768, 768)
        
        # EVAL For a subset of categories (to save time)
        if args.EVAL_SAMPLE_NUM != -1:
            cat_list = cat_list[:args.EVAL_SAMPLE_NUM]

        print(f'MSG: Evaluating for category ==> {cat}')
        for i, json_path in enumerate(tqdm(cat_list)):

            with open(os.path.join(dataset_path, 'PairAnnotation/test', json_path)) as temp_f:
                data = json.load(temp_f)
            
            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]

            src_ft = output_dict[data['src_imname']]
            trg_ft = output_dict[data['trg_imname']]
            
            src_ft = src_ft.cuda()
            trg_ft = trg_ft.cuda()
            
            if len(src_ft.shape) == 3:
                b,n,d = src_ft.shape 
        
                if args.model == 'cogvid_single':
                    h = args.eval_img_size[0]//16 
                    w = args.eval_img_size[1]//16 
                    src_ft = rearrange(src_ft, 'b (h w) d -> b d h w', h=h, w=w)
                    trg_ft = rearrange(trg_ft, 'b (h w) d -> b d h w', h=h, w=w)
                    
                else:
                    h = int(n ** 0.5)
                    src_ft = rearrange(src_ft, 'b (h w) d -> b d h w', h=h)
                    trg_ft = rearrange(trg_ft, 'b (h w) d -> b d h w', h=h)
                
            # ======================== VISUALIZE JOINT IMG PCA ========================
            if args.VIS_PCA_JOINT_IMG:
                
                SAVE_PATH_JOINT_PCA = f'{args.save_dir}/pca_joint/{cat}'
                if not os.path.exists(SAVE_PATH_JOINT_PCA):
                    os.makedirs(SAVE_PATH_JOINT_PCA)
                
                _, _, H, W = src_ft.shape 
                N = H*W
                
                src_feat = src_ft.clone() 
                trg_feat = trg_ft.clone() 
                
                src_feat = rearrange(src_feat, 'b d h w -> b (h w) d')  # 1 2304 1280
                src_feat = src_feat.squeeze(0)  # hw d
                
                trg_feat = rearrange(trg_feat, 'b d h w -> b (h w) d')  # 1 2304 1280
                trg_feat = trg_feat.squeeze(0)  # hw d 
                
                joint_feat = torch.cat([src_feat, trg_feat], dim=0)
                joint_feat = joint_feat.to(torch.float32).cuda()
                _,_,V = torch.pca_lowrank(joint_feat)
                pca1 = torch.matmul(joint_feat, V[:, :1])
                
                def minmax_norm(x):
                    """Min-max normalization along the token dimension (n,d) dim=n"""
                    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)

                pca1_norm = minmax_norm(pca1)
                
                # Segment foreground/background based on first PCA component
                foreground = pca1_norm.squeeze() > 0.4
                background = pca1_norm.squeeze() <= 0.4
                
                # Get 3 PCA components for foreground visualization
                _, _, V = torch.pca_lowrank(joint_feat[foreground])
                pca3_fg = torch.matmul(joint_feat[foreground], V[:, :3])
                pca3_fg_norm = minmax_norm(pca3_fg)
                
                # Get 3 PCA components for full feature visualization
                _, _, V_full = torch.pca_lowrank(joint_feat)
                pca3_full = torch.matmul(joint_feat, V_full[:, :3])
                pca3_full_norm = minmax_norm(pca3_full)
                
                # Foreground only visualization
                pca_vis_joint = torch.zeros(2*N, 3).cuda()
                pca_vis_joint[foreground] = pca3_fg_norm 
                
                pca_joint_vis = rearrange(pca_vis_joint, '(b n) d -> b n d', b=2) 
                src_pca = pca_joint_vis[0]
                trg_pca = pca_joint_vis[1] 
                
                src_pca = rearrange(src_pca, '(h w) d -> d h w', h=H)
                trg_pca = rearrange(trg_pca, '(h w) d -> d h w', h=H)
                
                up_src_pca = F.interpolate(src_pca.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                up_trg_pca = F.interpolate(trg_pca.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                joint_imgs_pca_fg = torch.cat([up_src_pca, up_trg_pca], dim=2)
                
                # Full feature visualization including background
                pca_joint_vis_full = rearrange(pca3_full_norm, '(b n) d -> b n d', b=2)
                src_pca_full = pca_joint_vis_full[0]
                trg_pca_full = pca_joint_vis_full[1]
                
                src_pca_full = rearrange(src_pca_full, '(h w) d -> d h w', h=H)
                trg_pca_full = rearrange(trg_pca_full, '(h w) d -> d h w', h=H)
                
                up_src_pca_full = F.interpolate(src_pca_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                up_trg_pca_full = F.interpolate(trg_pca_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                joint_imgs_pca_full = torch.cat([up_src_pca_full, up_trg_pca_full], dim=2)

                # Load and resize original image
                src_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                src_img = src_img.resize(interp_size)
                trg_img = trg_img.resize(interp_size)
                src_img_tensor = transforms.ToTensor()(src_img).cuda()
                trg_img_tensor = transforms.ToTensor()(trg_img).cuda()
                
                joint_imgs = torch.cat([src_img_tensor, trg_img_tensor], dim=2)
                
                # Stack original images, foreground PCA and full PCA vertically
                vis_joint = torch.cat([joint_imgs, joint_imgs_pca_full, joint_imgs_pca_fg], dim=1)
                
                if args.log_tool == 'wandb':
                    # log frequency
                    if i % args.WANDB_LOG_FREQ == 0:
                        wandb.log({f"vis_PCA_JOINT_{cat}/pca_joint_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg": wandb.Image(vis_joint) })
                else:
                    # Save visualization using torchvision
                    save_image(vis_joint, f"{SAVE_PATH_JOINT_PCA}/pca_joint_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg")

            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0
            
            # ======================== VISUALIZE KEYPOINTS PREDICTION ========================
            if args.VIS_KPTS_PREDICTION:
                
                SAVE_PATH_KPTS = f'{args.save_dir}/kpts/{cat}'
                if not os.path.exists(SAVE_PATH_KPTS):
                    os.makedirs(SAVE_PATH_KPTS)
                    
                # Load source and target images
                src_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                
                # Get original dimensions
                src_h, src_w = src_img.shape[:2]
                trg_h, trg_w = trg_img.shape[:2]
                
                # Calculate scale factors
                scale_x = src_w / trg_w
                scale_y = src_h / trg_h
                
                # Resize target image and adjust target points for visualization only
                trg_img = cv2.resize(trg_img, (src_w, src_h))
                vis_trg_kpts = [[int(kp[0] * scale_x), int(kp[1] * scale_y)] for kp in data['trg_kps']]
                
                # Create combined visualization
                combined_vis = np.hstack((src_img.copy(), trg_img.copy()))

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1
                    
                if args.VIS_KPTS_PREDICTION:
                    circle_color = (255,0,0)    # BGR
                    # Draw source keypoint
                    src_pt = (int(src_point[0]), int(src_point[1]))
                    cv2.circle(combined_vis, src_pt, 5, circle_color, -1)
                    
                    # Draw predicted target keypoint (with src_w offset)
                    vis_pred_x = int(max_yx[1] * scale_x)
                    vis_pred_y = int(max_yx[0] * scale_y)
                    pred_pt = (vis_pred_x + src_w, vis_pred_y)
                    cv2.circle(combined_vis, pred_pt, 5, circle_color, -1)
                    
                    # Draw line - green for correct matches, red for incorrect
                    line_color = (0,255,0) if (dist / threshold) <= 0.1 else (0,0,255)
                    cv2.line(combined_vis, src_pt, pred_pt, line_color, 1)
            
            if args.VIS_KPTS_PREDICTION: 
                if args.log_tool == 'wandb':
                    # log frequency
                    if i % args.WANDB_LOG_FREQ == 0:
                        wandb.log({f"vis_KPTS_{cat}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}": wandb.Image(combined_vis[..., ::-1]) })
                else:
                    cv2.imwrite(f"{SAVE_PATH_KPTS}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg", combined_vis)    
            
            cat_pck.append(correct / total)
        total_pck.extend(cat_pck)

        output[f'per_image_pck@0.1'][cat] = np.mean(cat_pck) * 100
        output[f'per_point_pck@0.1'][cat] = cat_correct / cat_total * 100
        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        
        if args.log_tool == 'wandb':
            wandb.log({f'per image PCK@0.1/{cat}': output[f'per_image_pck@0.1'][cat]})
            wandb.log({f'per point PCK@0.1/{cat}': output[f'per_point_pck@0.1'][cat]})
        
        # if cat == 'cat':
        #     break 

    output[f'per_image_pck@0.1']['All'] = np.mean(total_pck) * 100
    output[f'per_point_pck@0.1']['All'] = all_correct / all_total * 100
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
    if args.log_tool == 'wandb':
        wandb.log({f'per image PCK@0.1/All': output['per_image_pck@0.1']['All']})
        wandb.log({f'per point PCK@0.1/All': output['per_point_pck@0.1']['All']})
    
    return output 




def run_evaluation_semantic_joint(network, dataset_path, args):
    
    all_cats, cat2json, cat2img = prepare_spair(dataset_path, args)
    print_exp_info(args)
    
    # if not args.feat_already_extracted:
        
    #     breakpoint()
    #     extract_and_save_feats_joint(network, dataset_path, all_cats, cat2json, cat2img, args)
    # else:
    #     print(f'{args.model} DIFT features already extracted')
    
    # Prepare evaluation
    output={}
    output[f'per_image_pck@0.1']={}
    output[f'per_point_pck@0.1']={}
    total_pck = []
    all_correct = 0
    all_total = 0
    
    '''
        all_cats: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json.keys(): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json['aeroplane']: ['000250-2008_008607-2009_002388:aeroplane.json', '000540-2010_004817-2008_008607:aeroplane.json', ... ]
        len(cat2json['aeroplane']): 690
    '''
     
    for cat in all_cats:
        cat_list = cat2json[cat]
        # output_dict = torch.load(os.path.join(args.feat_save_path, f'{cat}.pth'), weights_only=True)

        cat_pck = []
        cat_correct = 0
        cat_total = 0
        
        # Define interpolation size for PCA visualization
        interp_size = (768, 768)
        
        # EVAL For a subset of categories (to save time)
        if args.EVAL_SAMPLE_NUM != -1:
            cat_list = cat_list[:args.EVAL_SAMPLE_NUM]

        print(f'Evaluating for category ==> {cat}')
        for i, json_path in enumerate(tqdm(cat_list)):

            with open(os.path.join(dataset_path, 'PairAnnotation/test', json_path)) as temp_f:
                data = json.load(temp_f)
            
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            
            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]
            
            img_src = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, src_imname))
            img_trg = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, trg_imname))
            
            prompt = f"a photo of a {cat}"
                
            img1_info = {
                'img1': img_src,
                'img1_cat': cat,
                'img1_name': src_imname.split('.')[0]
            }
            
            img2_info = {
                'img2': img_trg,
                'img2_cat': cat,
                'img2_name': trg_imname.split('.')[0]
            }
            
            
            if args.model == 'sd3_joint':
                extracted_feat = network.forward(   img1_info=img1_info,
                                                    img2_info=img2_info,
                                                    prompt=prompt,
                                                    negative_prompt="",
                                                    num_inference_steps=args.inf_max_step,
                                                    height=args.eval_img_size[0],
                                                    width=args.eval_img_size[1],
                                                    guidance_scale=7.0,
                                                    do_classifier_free_guidance=False)
                
                ''' extracted_feat: 24 2 2304 1536
                '''
            
            # breakpoint()
            if args.VIS_ATTN_MAP:
                
                # OUTPUT FEATURE MUST BE ATTENTION MAPS! 
                assert args.output_feat_type == 'attn_map'

                attn_maps12 = extracted_feat[:,0]   # src->trg: 24 2304 2304 
                attn_maps21 = extracted_feat[:,1]   # trg->src: 24 2304 2304 
                attn_maps11 = extracted_feat[:,2]   # src->src: 24 2304 2304 
                attn_maps22 = extracted_feat[:,3]   # trg->trg: 24 2304 2304 
                
                # 0. SELECT ATTN MAP TO VISUALIZE
                if args.VIS_ATTN_SRC_TO_TRG:
                    attn_maps_img = attn_maps12
                    attn_map_direction = 'attn_src_to_trg'  
                elif args.VIS_ATTN_TRG_TO_SRC:
                    attn_maps_img = attn_maps21
                    attn_map_direction = 'attn_trg_to_src'
                elif args.VIS_ATTN_SRC_TO_SRC:
                    attn_maps_img = attn_maps11
                    attn_map_direction = 'attn_src_to_src'
                elif args.VIS_ATTN_TRG_TO_TRG:
                    attn_maps_img = attn_maps22
                    attn_map_direction = 'attn_trg_to_trg'
                else:
                    print('ERROR!!!! VIS_ATTN_MAP')
                
                if args.CONCAT_WIDTH:
                    vis_h = args.eval_img_size[0] // 2
                    vis_w = args.eval_img_size[1] 
                else:
                    vis_h = args.eval_img_size[0]
                    vis_w = args.eval_img_size[1]
                    
                # 1. prepare images
                img1_tensor = network.pipe.image_processor.preprocess(img1_info['img1'], vis_h, vis_w)     # 1 3 768 768
                img2_tensor = network.pipe.image_processor.preprocess(img2_info['img2'], vis_h, vis_w)     # 1 3 768 768
                
                img1_tensor_re = (img1_tensor + 1) / 2  # [0,1]
                img1_np = (img1_tensor_re.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) # 768 768 3 
                img1_np = np.ascontiguousarray(img1_np) # 768 768 3 
                
                img2_tensor_re = (img2_tensor + 1) / 2  # [0,1]
                img2_np = (img2_tensor_re.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) # 768 768 3 
                img2_np = np.ascontiguousarray(img2_np) # 768 768 3 
                
                # 2. prepare size infos
                ps=16   # not real patch size, but patch size for visualization
                pH = vis_h // ps
                pW = vis_w // ps
                N = pH * pW

                # 3. set vis points 
                num_vis=30
                vis_points = torch.rand(N).argsort()[:num_vis].tolist()
                
                # breakpoint()
                vis_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
                for l in vis_layers: 
                    for point in vis_points:
                        
                        src_name = img1_info['img1_name']
                        trg_name = img2_info['img2_name']
                        
                        vis_save_path = f"{args.save_dir}/{attn_map_direction}/{img1_info['img1_cat']}/src{src_name}_trg{trg_name}/layer{l}"                        
                        # set save path 
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        
                        img1_np_vis = img1_np.copy()
                        img2_np_vis = img2_np.copy()
                        
                        # get l-th layer attention map with query_point
                        attn_mask = attn_maps_img[l][point].view(pH,pW)
                        attn_mask = F.interpolate(attn_mask[None, None], size=(vis_h, vis_w), mode='bilinear', align_corners=False).squeeze()
                        attn_mask = (attn_mask-attn_mask.min())/(attn_mask.max()-attn_mask.min())

                        idx_h = point // pW 
                        idx_w = point % pW 
                        
                        # Draw the query point as a circle
                        center = (idx_w*ps + ps//2, idx_h*ps + ps//2)
                        cv2.circle(img1_np_vis, center, ps//2, (0,0,255), -1)
                        cv2.circle(img1_np_vis, center, ps//2, (255,255,255), 2)
                        
                        attn_heatmap = cv2.applyColorMap(np.uint8(255*attn_mask), cv2.COLORMAP_JET)
                        
                        if args.VIS_ATTN_SRC_TO_SRC:
                            img2_tmp = img2_np_vis.copy()
                            img2_np_vis = img1_np_vis
                        
                        masked_img = img2_np_vis/255. + attn_heatmap/255.
                        masked_img = masked_img / masked_img.max()
                        
                        if args.VIS_ATTN_SRC_TO_SRC:
                            combined_img = np.concatenate([img1_np_vis, np.uint8(255*masked_img), img2_tmp], axis=1)
                        else:     
                            # Combine source and target images side by side
                            combined_img = np.concatenate([img1_np_vis, np.uint8(255*masked_img)], axis=1)
                        cv2.imwrite(f"{vis_save_path}/src{src_name}_trg{trg_name}_point{point}.jpg", combined_img)
                continue
            
            feat = extracted_feat[args.output_layer]    # 2 2304 1536
            src_ft = feat[0].unsqueeze(0)   # 1 2304 64
            trg_ft = feat[1].unsqueeze(0)   # 1 2304 64
            
            src_ft = src_ft.cuda()  # 1 2304 64
            trg_ft = trg_ft.cuda()  # 1 2304 64
            
            if len(src_ft.shape) == 3:
                b,n,d = src_ft.shape 
                if args.CONCAT_WIDTH:
                    h = args.eval_img_size[0] // 2 // 16
                    w = args.eval_img_size[1] // 16
                    src_ft = rearrange(src_ft, 'b (h w) d -> b d h w', h=h, w=w)
                    trg_ft = rearrange(trg_ft, 'b (h w) d -> b d h w', h=h, w=w)
                else: 
                    h = args.eval_img_size[0] // 16
                    w = args.eval_img_size[1] // 16
                    src_ft = rearrange(src_ft, 'b (h w) d -> b d h w', h=h, w=w)
                    trg_ft = rearrange(trg_ft, 'b (h w) d -> b d h w', h=h, w=w)
                
            # ======================== VISUALIZE JOINT IMG PCA ========================
            if args.VIS_PCA_JOINT_IMG:
                
                SAVE_PATH_JOINT_PCA = f'{args.save_dir}/pca_joint/{cat}'
                if not os.path.exists(SAVE_PATH_JOINT_PCA):
                    os.makedirs(SAVE_PATH_JOINT_PCA)
                
                _, _, H, W = src_ft.shape 
                N = H*W
                
                src_feat = src_ft.clone() 
                trg_feat = trg_ft.clone() 
                
                src_feat = rearrange(src_feat, 'b d h w -> b (h w) d')  # 1 2304 1280
                src_feat = src_feat.squeeze(0)  # hw d
                
                trg_feat = rearrange(trg_feat, 'b d h w -> b (h w) d')  # 1 2304 1280
                trg_feat = trg_feat.squeeze(0)  # hw d 
                
                joint_feat = torch.cat([src_feat, trg_feat], dim=0)
                joint_feat = joint_feat.to(torch.float32).cuda()
                _,_,V = torch.pca_lowrank(joint_feat)
                pca1 = torch.matmul(joint_feat, V[:, :1])
                
                def minmax_norm(x):
                    """Min-max normalization along the token dimension (n,d) dim=n"""
                    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)

                pca1_norm = minmax_norm(pca1)
                
                # Segment foreground/background based on first PCA component
                foreground = pca1_norm.squeeze() > 0.4
                background = pca1_norm.squeeze() <= 0.4
                
                # Get 3 PCA components for foreground visualization
                _, _, V = torch.pca_lowrank(joint_feat[foreground])
                pca3_fg = torch.matmul(joint_feat[foreground], V[:, :3])
                pca3_fg_norm = minmax_norm(pca3_fg)
                
                # Get 3 PCA components for full feature visualization
                _, _, V_full = torch.pca_lowrank(joint_feat)
                pca3_full = torch.matmul(joint_feat, V_full[:, :3])
                pca3_full_norm = minmax_norm(pca3_full)
                
                # Foreground only visualization
                pca_vis_joint = torch.zeros(2*N, 3).cuda()
                pca_vis_joint[foreground] = pca3_fg_norm 
                
                pca_joint_vis = rearrange(pca_vis_joint, '(b n) d -> b n d', b=2) 
                src_pca = pca_joint_vis[0]
                trg_pca = pca_joint_vis[1] 
                
                src_pca = rearrange(src_pca, '(h w) d -> d h w', h=H)
                trg_pca = rearrange(trg_pca, '(h w) d -> d h w', h=H)
                
                up_src_pca = F.interpolate(src_pca.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                up_trg_pca = F.interpolate(trg_pca.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                joint_imgs_pca_fg = torch.cat([up_src_pca, up_trg_pca], dim=2)
                
                # Full feature visualization including background
                pca_joint_vis_full = rearrange(pca3_full_norm, '(b n) d -> b n d', b=2)
                src_pca_full = pca_joint_vis_full[0]
                trg_pca_full = pca_joint_vis_full[1]
                
                src_pca_full = rearrange(src_pca_full, '(h w) d -> d h w', h=H)
                trg_pca_full = rearrange(trg_pca_full, '(h w) d -> d h w', h=H)
                
                up_src_pca_full = F.interpolate(src_pca_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                up_trg_pca_full = F.interpolate(trg_pca_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                joint_imgs_pca_full = torch.cat([up_src_pca_full, up_trg_pca_full], dim=2)

                # Load and resize original image
                src_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                src_img = src_img.resize(interp_size)
                trg_img = trg_img.resize(interp_size)
                src_img_tensor = transforms.ToTensor()(src_img).cuda()
                trg_img_tensor = transforms.ToTensor()(trg_img).cuda()
                
                joint_imgs = torch.cat([src_img_tensor, trg_img_tensor], dim=2)
                
                # Stack original images, foreground PCA and full PCA vertically
                vis_joint = torch.cat([joint_imgs, joint_imgs_pca_full, joint_imgs_pca_fg], dim=1)
                
                if args.log_tool == 'wandb':
                    # log frequency
                    if i % args.WANDB_LOG_FREQ == 0:
                        wandb.log({f"vis_PCA_JOINT_{cat}/pca_joint_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg": wandb.Image(vis_joint) })
                else:
                    # Save visualization using torchvision
                    save_image(vis_joint, f"{SAVE_PATH_JOINT_PCA}/pca_joint_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg")

            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0
            
            # ======================== VISUALIZE KEYPOINTS PREDICTION ========================
            if args.VIS_KPTS_PREDICTION:
                
                SAVE_PATH_KPTS = f'{args.save_dir}/kpts/{cat}'
                if not os.path.exists(SAVE_PATH_KPTS):
                    os.makedirs(SAVE_PATH_KPTS)
                    
                # Load source and target images
                src_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                
                # Get original dimensions
                src_h, src_w = src_img.shape[:2]
                trg_h, trg_w = trg_img.shape[:2]
                
                # Calculate scale factors
                scale_x = src_w / trg_w
                scale_y = src_h / trg_h
                
                # Resize target image and adjust target points for visualization only
                trg_img = cv2.resize(trg_img, (src_w, src_h))
                vis_trg_kpts = [[int(kp[0] * scale_x), int(kp[1] * scale_y)] for kp in data['trg_kps']]
                
                # Create combined visualization
                combined_vis = np.hstack((src_img.copy(), trg_img.copy()))

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1
                    
                if args.VIS_KPTS_PREDICTION:
                    circle_color = (255,0,0)    # BGR
                    # Draw source keypoint
                    src_pt = (int(src_point[0]), int(src_point[1]))
                    cv2.circle(combined_vis, src_pt, 5, circle_color, -1)
                    
                    # Draw predicted target keypoint (with src_w offset)
                    vis_pred_x = int(max_yx[1] * scale_x)
                    vis_pred_y = int(max_yx[0] * scale_y)
                    pred_pt = (vis_pred_x + src_w, vis_pred_y)
                    cv2.circle(combined_vis, pred_pt, 5, circle_color, -1)
                    
                    # Draw line - green for correct matches, red for incorrect
                    line_color = (0,255,0) if (dist / threshold) <= 0.1 else (0,0,255)
                    cv2.line(combined_vis, src_pt, pred_pt, line_color, 1)
            
            if args.VIS_KPTS_PREDICTION: 
                if args.log_tool == 'wandb':
                    # log frequency
                    if i % args.WANDB_LOG_FREQ == 0:
                        wandb.log({f"vis_KPTS_{cat}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}": wandb.Image(combined_vis[..., ::-1]) })
                else:
                    cv2.imwrite(f"{SAVE_PATH_KPTS}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg", combined_vis)    
            
            cat_pck.append(correct / total)
        
        if args.VIS_ATTN_MAP:
            continue
        
        total_pck.extend(cat_pck)

        output[f'per_image_pck@0.1'][cat] = np.mean(cat_pck) * 100
        output[f'per_point_pck@0.1'][cat] = cat_correct / cat_total * 100
        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        
        if args.log_tool == 'wandb':
            wandb.log({f'per image PCK@0.1/{cat}': output[f'per_image_pck@0.1'][cat]})
            wandb.log({f'per point PCK@0.1/{cat}': output[f'per_point_pck@0.1'][cat]})
        
        # if output[f'per_image_pck@0.1']['aeroplane'] < 50.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
        
        # if output[f'per_image_pck@0.1']['bicycle'] < 40.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
    
        # if output[f'per_image_pck@0.1']['bird'] < 70.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
        
        # if output[f'per_image_pck@0.1']['boat'] < 20.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
        
        # if output[f'per_image_pck@0.1']['bottle'] < 20.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
        
        # if output[f'per_image_pck@0.1']['cat'] < 70.0:
        #     print(f"BREAK!! for {args.save_dir.split('/')[-1]}, due to LOW PCK for {cat}: {output[f'per_image_pck@0.1'][cat]}")
        #     break
        
    if args.VIS_ATTN_MAP:
        output = None

    else:
        output[f'per_image_pck@0.1']['All'] = np.mean(total_pck) * 100
        output[f'per_point_pck@0.1']['All'] = all_correct / all_total * 100
        print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
        print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
        if args.log_tool == 'wandb':
            wandb.log({f'per image PCK@0.1/All': output['per_image_pck@0.1']['All']})
            wandb.log({f'per point PCK@0.1/All': output['per_point_pck@0.1']['All']})
    return output 


def run_evaluation_semantic_joint_before(pipe, dataset_path, args):
    
    # copied code from dift 
    feat_save_path = args.feat_save_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    all_cats.sort()
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list

    # get test image path for all cats
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)
                
    print('-'*100)
    print('< cat2img info >')
    for key, value in cat2img.items():
        count=len(cat2img[key])
        print(f'{key}: {count}')
    print('-'*100)
    print('< cat2json info >')
    for key, value in cat2json.items():
        print(f'{key}: {len(value)}')
    print('-'*100)
    
    if not args.feat_already_extracted:
        print(f'MSG: Extracting all test images {args.model} features...')
        os.makedirs(feat_save_path, exist_ok=True)
        
        
        for cat in tqdm(all_cats):
            cat_list = cat2json[cat]
            
            output_dict = {}
            for i, json_path in enumerate(tqdm(cat_list)):
                with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                
                print('current cat: ', cat)
                print('current json_path: ', json_path)
                
                src_imname = data['src_imname']
                trg_imname = data['trg_imname']
                
                src_img_size = data['src_imsize'][:2][::-1]
                trg_img_size = data['trg_imsize'][:2][::-1]
                
                img_src = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, src_imname))
                img_trg = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, trg_imname))
                
                prompt = f"a photo of a {cat}"
                    
                img1_info = {
                    'img1': img_src,
                    'img1_cat': cat,
                    'img1_name': src_imname.split('.')[0]
                }
                
                img2_info = {
                    'img2': img_trg,
                    'img2_cat': cat,
                    'img2_name': trg_imname.split('.')[0]
                }
                
                breakpoint()
                # print(f'saving attn maps for {cat} {image_path}')
                output_dict[image_path] = pipe( img1_info=img1_info,
                                                img2_info=img2_info,
                                                prompt=prompt,
                                                negative_prompt="",
                                                num_inference_steps=28,
                                                height=args.eval_img_size[0],
                                                width=args.eval_img_size[1],
                                                guidance_scale=7.0,)
                
                breakpoint()
            
        
        
        
        
        
        
        for cat in tqdm(all_cats):
            output_dict = {}
            image_list = cat2img[cat]
            
            for image_path in image_list:
                img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
                
                if args.model == 'dift_sd':
                    # print(f'MSG: Extracting feats for dift_sd')
                    output_dict[image_path] = pipe.forward(img, # 1 1280 48 48
                                                            category=cat,
                                                            img_size=args.eval_img_size,
                                                            t=args.t,
                                                            up_ft_index=args.up_ft_index,
                                                            ensemble_size=args.ensemble_size)
                    # breakpoint()
                    
                elif args.model == 'sd3_baseline':
                    breakpoint()
                    # print(f'MSG: Extracting feats for sd3_baseline')
                    prompt = f"a photo of a {cat}"
                    
                    img1_info = {
                        'img1': img,
                        'img1_cat': cat,
                        'img1_name': image_path.split('.')[0]
                    }
                    
                    # print(f'saving attn maps for {cat} {image_path}')
                    output_dict[image_path] = pipe(img1_info=img1_info,
                                 prompt=prompt,
                                 negative_prompt="",
                                 num_inference_steps=28,
                                 height=args.eval_img_size[0],
                                 width=args.eval_img_size[1],
                                 guidance_scale=7.0,)

            torch.save(output_dict, os.path.join(feat_save_path, f'{cat}.pth'))
            print(f'MSG: saved feats for {cat}, in {feat_save_path}')
                            
    else:
        print(f'MSG: {args.model} features already extracted')
    print('-'*100)
    
    breakpoint()
    # Prepare evaluation
    output={}
    output[f'per_image_pck@0.1']={}
    output[f'per_point_pck@0.1']={}
    total_pck = []
    all_correct = 0
    all_total = 0
    
    '''
        all_cats: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json.keys(): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        cat2json['aeroplane']: ['000250-2008_008607-2009_002388:aeroplane.json', '000540-2010_004817-2008_008607:aeroplane.json', ... ]
        len(cat2json['aeroplane']): 690
    '''
    
    # remove model to save GPU memory 
    import gc
    pipe.to('cpu')
    del pipe
    gc.collect()
    torch.cuda.empty_cache() 
        
    for cat in all_cats:
        cat_list = cat2json[cat]
        output_dict = torch.load(os.path.join(feat_save_path, f'{cat}.pth'), weights_only=True)

        cat_pck = []
        cat_correct = 0
        cat_total = 0
        
        if args.vis_pca:
            
            for key, value in output_dict.items():
                
                img_name = key.split(".")[0]
                vis_pca_save_path = f'{args.save_dir}/pca/{cat}'
                if not os.path.exists(vis_pca_save_path):
                    os.makedirs(vis_pca_save_path)
                    
                feat = output_dict[key]     # 1 1280 48 48 
                
                if len(feat.shape) == 3:
                    b,n,d = feat.shape 
                    h = int(n ** 0.5)
                    feat = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                
                _, _, H, W = feat.shape
                
                feat = rearrange(feat, 'b d h w -> b (h w) d')  # 1 2304 1280
                feat = feat.squeeze(0)  # hw d
                
                # Get first PCA component to separate foreground/background
                feat = feat.to(torch.float32)
                _,_,V = torch.pca_lowrank(feat)
                pca1 = torch.matmul(feat, V[:, :1])
                
                def minmax_norm(x):
                    """Min-max normalization along the token dimension (n,d) dim=n"""
                    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)

                pca1_norm = minmax_norm(pca1)
                
                # Segment foreground/background based on first PCA component
                foreground = pca1_norm.squeeze() > 0.4
                background = pca1_norm.squeeze() <= 0.4
                
                # Get 3 PCA components for foreground visualization
                _, _, V = torch.pca_lowrank(feat[foreground])
                pca3_fg = torch.matmul(feat[foreground], V[:, :3])
                pca3_fg_norm = minmax_norm(pca3_fg)
                
                # Get 3 PCA components for full feature visualization
                _, _, V_full = torch.pca_lowrank(feat)
                pca3_full = torch.matmul(feat, V_full[:, :3])
                pca3_full_norm = minmax_norm(pca3_full)
                
                # Define interpolation size
                interp_size = (768, 768)
                
                # Reshape PCA components back to spatial dimensions
                pca_vis_fg = torch.zeros((H*W, 3), device=feat.device)
                pca_vis_fg[foreground] = pca3_fg_norm
                pca_vis_fg = pca_vis_fg.reshape(H, W, 3).permute(2,0,1)
                pca_vis_fg = F.interpolate(pca_vis_fg.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                pca_vis_full = pca3_full_norm.reshape(H, W, 3).permute(2,0,1)
                pca_vis_full = F.interpolate(pca_vis_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
                
                # Load and resize original image
                img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, key))
                img = img.resize(interp_size)
                img_tensor = transforms.ToTensor()(img).cuda()
                
                # Concatenate original and both PCA visualizations horizontally
                vis = torch.cat([img_tensor, pca_vis_fg, pca_vis_full], dim=2)
                # Save visualization using torchvision
                save_image(vis, f'{vis_pca_save_path}/pca_{img_name}.jpg')

        print(f'MSG: Evaluating for category ==> {cat}')
        for i, json_path in enumerate(tqdm(cat_list)):

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
            
            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]

            src_ft = output_dict[data['src_imname']]
            trg_ft = output_dict[data['trg_imname']]
            
            if len(src_ft.shape) == 3:
                b,n,d = src_ft.shape 
                h = int(n ** 0.5)
                src_ft = rearrange(src_ft, 'b (h w) d -> b d h w', h=h)
                trg_ft = rearrange(trg_ft, 'b (h w) d -> b d h w', h=h)

            src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0
            
            SAVE_PATH = f'{args.save_dir}/{cat}'
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
                    
            VISUALIZE=args.vis_pred_kpts
            ## PHO_VISUALIZE BBOX and KEYPOINTS IN TARGET AND SOURCE
            if VISUALIZE:
                # Load source and target images
                src_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
                trg_img = cv2.imread(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
                
                # Get original dimensions
                src_h, src_w = src_img.shape[:2]
                trg_h, trg_w = trg_img.shape[:2]
                
                # Calculate scale factors
                scale_x = src_w / trg_w
                scale_y = src_h / trg_h
                
                # Resize target image and adjust target points for visualization only
                trg_img = cv2.resize(trg_img, (src_w, src_h))
                vis_trg_kpts = [[int(kp[0] * scale_x), int(kp[1] * scale_y)] for kp in data['trg_kps']]
                
                # Create combined visualization
                combined_vis = np.hstack((src_img.copy(), trg_img.copy()))

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                num_channel = src_ft.size(1)
                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                trg_vec = F.normalize(trg_vec) # HW, c
                cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy() # H, W

                max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1
                    
                if VISUALIZE:
                    circle_color = (0,0,255) if args.log_tool == 'wandb' else (255,0,0)
                    # Draw source keypoint
                    src_pt = (int(src_point[0]), int(src_point[1]))
                    cv2.circle(combined_vis, src_pt, 5, circle_color, -1)
                    
                    # Draw predicted target keypoint (with src_w offset)
                    vis_pred_x = int(max_yx[1] * scale_x)
                    vis_pred_y = int(max_yx[0] * scale_y)
                    pred_pt = (vis_pred_x + src_w, vis_pred_y)
                    cv2.circle(combined_vis, pred_pt, 5, circle_color, -1)
                    
                    # Draw line - green for correct matches, red for incorrect
                    if args.log_tool == 'wandb':
                        line_color = (0,255,0) if (dist / threshold) <= 0.1 else (255,0,0)
                    else:
                        line_color = (0,255,0) if (dist / threshold) <= 0.1 else (0,0,255)
                    cv2.line(combined_vis, src_pt, pred_pt, line_color, 1)
            
            if VISUALIZE: 
                if args.log_tool == 'wandb':
                    # log frequency
                    if i % 75 == 0:
                        wandb.log({f"vis_{cat}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}": wandb.Image(combined_vis) })
                
                else:
                    # Save visualization
                    cv2.imwrite(f"{SAVE_PATH}/pred_src{data['src_imname'].split('.')[0]}_trg{data['trg_imname'].split('.')[0]}.jpg", combined_vis)    
            
            cat_pck.append(correct / total)
        total_pck.extend(cat_pck)

        output[f'per_image_pck@0.1'][cat] = np.mean(cat_pck) * 100
        output[f'per_point_pck@0.1'][cat] = cat_correct / cat_total * 100
        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        if args.log_tool == 'wandb':
            wandb.log({f'per image PCK@0.1/{cat}': output[f'per_image_pck@0.1'][cat]})
            wandb.log({f'per point PCK@0.1/{cat}': output[f'per_point_pck@0.1'][cat]})
        # breakpoint()

    output[f'per_image_pck@0.1']['All'] = np.mean(total_pck) * 100
    output[f'per_point_pck@0.1']['All'] = all_correct / all_total * 100
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
    if args.log_tool == 'wandb':
        wandb.log({f'per image PCK@0.1/All': output['per_image_pck@0.1']['All']})
        wandb.log({f'per point PCK@0.1/All': output['per_point_pck@0.1']['All']})
    
    return output 