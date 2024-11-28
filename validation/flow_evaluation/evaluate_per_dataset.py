import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader


from utils_flow.img_processing_utils import pad_to_same_shape
from validation.flow_evaluation.metrics_uncertainty import (compute_average_of_uncertainty_metrics, compute_aucs,
                                                            compute_uncertainty_per_image)
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from validation.plot import plot_sparse_keypoints, plot_flow_and_uncertainty, plot_individual_images
from .metrics_segmentation_matching import poly_str_to_mask, intersection_over_union, label_transfer_accuracy
from utils_flow.pixel_wise_mapping import warp

# JLP
import wandb
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn as nn 
from models.modules.mod import unnormalise_and_convert_mapping_to_flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def warp_image(image, flow):
#     """Warp image using flow field"""
#     B, C, H, W = image.size()
#     # Create mesh grid
#     xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
#     yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
#     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     grid = torch.cat((xx, yy), 1).float().to(device)
        
#     # Add flow to grid
#     vgrid = grid + flow
        
#     # Scale grid to [-1,1]
#     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
#     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        
#     # Reshape for grid_sample
#     vgrid = vgrid.permute(0, 2, 3, 1)
        
#     # Warp
#     output = torch.nn.functional.grid_sample(image, vgrid, align_corners=True)
#     return output
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


def compute_pck_sparse_data(x_s, y_s, x_r, y_r, flow, pck_thresholds, dict_list_uncertainties, uncertainty_est=None):

    flow_x = flow[0, 0].cpu().numpy()
    flow_y = flow[0, 1].cpu().numpy()

    # remove points for which xB, yB are outside of the image
    h, w = flow_x.shape
    index_valid = (np.int32(np.round(x_r)) >= 0) * (np.int32(np.round(x_r)) < w) * \
                  (np.int32(np.round(y_r)) >= 0) * (np.int32(np.round(y_r)) < h)
    x_s, y_s, x_r, y_r = x_s[index_valid], y_s[index_valid], x_r[index_valid], y_r[index_valid]
    nbr_valid_corr = index_valid.sum()

    # calculates the PCK
    if nbr_valid_corr > 0:
        # more accurate to compute the flow like this, instead of rounding both coordinates as in RANSAC-Flow
        flow_gt_x = x_s - x_r
        flow_gt_y = y_s - y_r
        flow_est_x = flow_x[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        flow_est_y = flow_y[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        EPE = ((flow_gt_x - flow_est_x) ** 2 + (flow_gt_y - flow_est_y) ** 2) ** 0.5
        EPE = EPE.reshape((-1, 1))
        AEPE = np.mean(EPE)
        count_pck = np.sum(EPE <= pck_thresholds, axis=0)
        # here compares the EPE of the pixels to be inferior to some value pixelGrid
    else:
        count_pck = np.zeros(pck_thresholds.shape[1])
        AEPE = np.nan

    results = {'count_pck': count_pck, 'nbr_valid_corr': nbr_valid_corr, 'aepe': AEPE}

    # calculates sparsification plot information
    if uncertainty_est is not None:
        flow_est = torch.from_numpy(np.concatenate((flow_est_x.reshape(-1, 1), flow_est_y.reshape(-1, 1)), axis=1))
        flow_gt = torch.from_numpy(np.concatenate((flow_gt_x.reshape(-1, 1), flow_gt_y.reshape(-1, 1)), axis=1))

        # uncert shape is #number_of_elements
        for uncertainty_name in uncertainty_est.keys():
            if uncertainty_name == 'inference_parameters' or uncertainty_name == 'log_var_map' or \
                    uncertainty_name == 'weight_map' or uncertainty_name == 'warping_mask':
                continue

            if 'p_r' == uncertainty_name:
                # convert confidence map to uncertainty
                uncert = (1.0 / (uncertainty_est['p_r'] + 1e-6)).squeeze()[np.int32(np.round(y_r)),
                                                                           np.int32(np.round(x_r))]
            else:
                uncert = uncertainty_est[uncertainty_name].squeeze()[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
            # compute metrics based on uncertainty
            uncertainty_metric_dict = compute_aucs(flow_gt, flow_est, uncert, intervals=50)
            if uncertainty_name not in dict_list_uncertainties.keys():
                # for first image, create the list for each possible uncertainty type
                dict_list_uncertainties[uncertainty_name] = []
            dict_list_uncertainties[uncertainty_name].append(uncertainty_metric_dict)
    return results, dict_list_uncertainties


def run_evaluation_megadepth_or_robotcar(network, root, path_to_csv, estimate_uncertainty=False,
                                         min_size=480, stride_net=16, pre_processing=None,
                                         path_to_save=None, plot=False, plot_100=False,
                                         plot_ind_images=False):
    """
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here recreate the same functions that they used, for fair comparison, but add additional metrics.
    """

    df = pd.read_csv(path_to_csv, dtype=str)
    nbImg = len(df)

    # pixelGrid = np.around(np.logspace(0, np.log10(36), 8).reshape(-1, 8))
    # looks at different distances for the keypoint
    pixelGrid = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 13.0, 22.0, 36.0]).reshape(-1, 9)
    # for metric calculation
    count_pck = np.zeros_like(pixelGrid)
    nbr_valid_corr = 0
    aepe_list = []
    dict_list_uncertainties = {}

    for i in tqdm(range(nbImg)):
        scene = df['scene'][i]

        # read and open the source and target image
        Is_original = Image.open(os.path.join(os.path.join(root, scene), df['source_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(root, df['source_image'][i])).convert('RGB')
        It_original = Image.open(os.path.join(os.path.join(root, scene), df['target_image'][i])).convert('RGB') \
            if scene != '/' else Image.open(os.path.join(root, df['target_image'][i])).convert('RGB')

        # resize images and scale corresponding keypoints
        Is_original, Xs, Ys, valids = resize_images_to_min_resolution(min_size, Is_original, df['XA'][i],
                                                                      df['YA'][i], stride_net)
        It_original, Xt, Yt, validt = resize_images_to_min_resolution(min_size, It_original, df['XB'][i],
                                                                      df['YB'][i], stride_net)
        It_original = np.array(It_original)
        Is_original = np.array(Is_original)

        # removes points that are not contained in the source or the target
        index_valid = valids * validt
        Xs, Ys, Xt, Yt = Xs[index_valid], Ys[index_valid], Xt[index_valid], Yt[index_valid]

        # padd the images to the same shape to be fed to network + convert them to Tensors
        Is_original_padded_numpy, It_original_padded_numpy = pad_to_same_shape(Is_original, It_original)
        Is = torch.Tensor(Is_original_padded_numpy).permute(2, 0, 1).unsqueeze(0)
        It = torch.Tensor(It_original_padded_numpy).permute(2, 0, 1).unsqueeze(0)

        if pre_processing is not None:
            uncertainty_est = None
            flow_estimated = pre_processing.combine_with_est_flow_field(i, Is_original_padded_numpy,
                                                                        It_original_padded_numpy, Is, It, network)
        else:
            if estimate_uncertainty:
                flow_estimated, uncertainty_est = network.estimate_flow_and_confidence_map(Is, It)
            else:
                uncertainty_est = None
                flow_estimated = network.estimate_flow(Is, It)

        dict_results, dict_list_uncertainties = compute_pck_sparse_data(Xs, Ys, Xt, Yt, flow_estimated, pixelGrid,
                                                                        uncertainty_est=uncertainty_est,
                                                                        dict_list_uncertainties=dict_list_uncertainties)
        count_pck = count_pck + dict_results['count_pck']
        if dict_results['aepe'] != np.nan:
            aepe_list.append(dict_results['aepe'])
        nbr_valid_corr += dict_results['nbr_valid_corr']

        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i), Is, It, flow_estimated)
        if plot or (plot_100 and i < 100):
            plot_sparse_keypoints(path_to_save, 'image_{}'.format(i), Is, It, flow_estimated, Xs, Ys, Xt, Yt,
                                  uncertainty_comp_est=uncertainty_est)

    # Note that the PCK is over the whole dataset, for consistency with RANSAC-Flow computation.
    output = {'pixel-threshold': pixelGrid.tolist(), 'PCK': (count_pck / (nbr_valid_corr + 1e-6)).tolist(),
              'AEPE': np.mean(aepe_list).astype(np.float64)}
    print("Validation MegaDepth: {}".format(output['PCK']))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


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


def run_evaluation_sintel(network, test_dataloader, device, estimate_uncertainty=False):
    epe_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], []
    dict_list_uncertainties = {}
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            flow_est = network.estimate_flow(source_img, target_img)

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()
        epe_list.append(epe.view(-1).cpu().numpy())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

        if estimate_uncertainty:
            dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                    dict_list_uncertainties)

    epe_all = np.concatenate(epe_list).astype(np.float64)
    epe = np.mean(epe_all)
    pck1 = np.mean(epe_all <= 1)
    pck3 = np.mean(epe_all <= 3)
    pck5 = np.mean(epe_all <= 5)

    output = {'AEPE': epe, 'PCK_1': pck1, 'PCK_3': pck3, 'PCK5': pck5,
              'PCK_1_per_image': np.mean(pck_1_list),
              'PCK_3_per_image': np.mean(pck_3_list), 'PCK_5_per_image': np.mean(pck_5_list),
              }
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, pck1, pck3, pck5))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
    return output


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False, name_dataset=None, rate=None, curr_id=None, args=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []
    dict_list_uncertainties = {}

    # number of images to log to wandb
    wandb_num_log_img = 24


    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image'] # source, target, flow_gt, mask_valid ALL resized to args.eval_img_size
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device)
        mask_valid = mini_batch['correspondence_mask'].to(device)
        mask_valid_orig = mask_valid.clone()

        b, _, H, W = source_img.shape

        source_img = source_img.float().to(device) # 1 3 h w
        target_img = target_img.float().to(device) # 1 3 h w

        ## check if the flow_gt is correctly resized
        # save_image(source_img, f'./suppl_tmp_img_src.jpg', normalize=True)
        # save_image(target_img, f'./suppl_tmp_img_tgt.jpg', normalize=True)
        # save_image(mask_valid.float(), f'./suppl_tmp_img_mask.jpg', normalize=True)
        # warped_source_gt = warp(source_img, flow_gt)  
        # save_image(warped_source_gt, f'./suppl_tmp_img_warped_src_gt.jpg', normalize=True)

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

            # save_image(source_img, f'.tmp1_img_src.jpg', normalize=True)
            # save_image(target_img, f'.tmp1_img_tgt.jpg', normalize=True)
            # save_image(mask_valid.float(), f'.tmp1_img_mask.jpg', normalize=True)
            # warped_source_gt = warp(source_img, flow_gt)  
            # save_image(warped_source_gt, f'.tmp1_img_warped_src_gt.jpg', normalize=True)

        # crocoflow, croco_catseg, 
        if args.model == 'croco':
            source_img = source_img / 255.0
            target_img = target_img / 255.0

            # for crocoflow as it predicts uncertainty
            if estimate_uncertainty:
                output = network(target_img, source_img)
                flow_est = output[:,:-1,:,:]
                conf = output[:,-1,:,:]
            
            # for other croco models that doesnt predict uncertainty
            else:
                if args.model =='croco_catseg':
                    if args.dense_zoom_in:
                        flow_est, uncertainty_est = network.zoom_in_batch(source_img, target_img, zoom_ratio=args.dense_zoom_ratio, optimize=False, homo_only=False, batch_size=b)
                    else:
                        output = network(target_img, source_img)   
                        if isinstance(output, dict):
                            flow_est = output['flow_estimates'][0]
                        elif isinstance(output, list):
                            flow_est = output[0]

                elif args.model == 'future croco models':
                    pass
            
        
        elif args.model == 'crocov2' or args.model == 'crocov1':

            H_32, W_32 = args.model_img_size

            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.

            source_img_orig = source_img.clone()
            target_img_orig = target_img.clone()

            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std

            source_img = F.interpolate(source_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            # breakpoint()
            output_mode = args.output_mode    # enc_feat, dec_feat, camap
            outputs = network(target_img, source_img, output_mode=output_mode)

            if output_mode == 'enc_feat':
                feats1, feats2 = outputs[0], outputs[1]

                feat1 = feats1[-1]  # b n d 
                feat2 = feats2[-1]

                l2norm = FeatureL2Norm()    # normalizes along the feature 

                ## from (b, n, d) normalize along the n dimension.
                # feat1 = l2norm(feat1)
                # feat2 = l2norm(feat2)

                # from (b, n, d) normalize along the d dimension.
                feat1 = l2norm(feat1.permute(0,2,1)).permute(0,2,1)    
                feat2 = l2norm(feat2.permute(0,2,1)).permute(0,2,1)

                corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)    # b 196 196
                
            elif output_mode == 'dec_feat':
                dec_feats1, dec_feats2 = outputs[0], outputs[1]

                # use the first decoder feature
                dec_feat1 = dec_feats1[0]
                dec_feat2 = dec_feats2[0]

                ## use last decoder feature
                # dec_feat1 = dec_feats1[-1]
                # dec_feat2 = dec_feats2[-1]

                ## use the average of all decoder features
                # dec_feat1 = torch.stack(dec_feats1, dim=1).mean(dim=1)
                # dec_feat2 = torch.stack(dec_feats2, dim=1).mean(dim=1)

                l2norm = FeatureL2Norm()

                ## from (b, n, d) normalize along the n dimension.
                dec_feat1 = l2norm(dec_feat1)
                dec_feat2 = l2norm(dec_feat2)

                # from (b, n, d) normalize along the d dimension.
                # dec_feat1 = l2norm(dec_feat1.permute(0,2,1)).permute(0,2,1)    # b n d
                # dec_feat2 = l2norm(dec_feat2.permute(0,2,1)).permute(0,2,1)

                corr = torch.einsum('bnd, bmd -> bnm', dec_feat1, dec_feat2)    # b 196 196

            elif output_mode == 'ca_map':
                camap1, camap2 = outputs[0], outputs[1]   # b 12 196 196

                camap1 = [attn.mean(dim=1).detach() for attn in camap1]   # b 196 196
                camap2 = [attn.mean(dim=1).detach() for attn in camap2]   # avg heads

                ## heuristic attention visualize
                # for j in range(len(camap1)):
                #     print(camap1[j].argmax(dim=-1))
                # print('-'*50)
                # for j in range(len(camap2)):
                #     print(camap2[j].argmax(dim=-1))
                # breakpoint()

                for i in range(len(camap1)):
                    camap1[i][:,:,0]=camap1[i].min()
                for i in range(len(camap2)):
                    camap2[i][:,:,0]=camap2[i].min()

                
                # for j in range(len(camap1)):
                #     print(camap1[j].argmax(dim=-1))
                # print('-'*50)
                # for j in range(len(camap2)):
                #     print(camap2[j].argmax(dim=-1))
                # breakpoint()


                camap1 = torch.stack(camap1, dim=1)
                camap2 = torch.stack(camap2, dim=1)
                corr = (camap1.mean(dim=1) + camap2.mean(dim=1).transpose(-1,-2))/2.

                # # heuristic attention refine
                # for i in range(len(camap1)):
                #     camap1[i][:,:,0]=0
                # for i in range(len(camap2)):
                #     camap2[i][:,:,0]=0

                # corr = [ (camap1[i] + camap2[i].transpose(-1,-2))/2 for i in range(len(camap1))]    # b 196 196
                # # avg across layers
                # corr = torch.stack(corr, dim=1).mean(dim=1)    # b 196 196

                
            else:
                pass

            feature_size = H_32 // 16
            x_normal = np.linspace(-1,1,feature_size)
            x_normal = nn.Parameter(torch.tensor(x_normal, dtype=torch.float, requires_grad=False)).cuda()
            y_normal = np.linspace(-1,1,feature_size)
            y_normal = nn.Parameter(torch.tensor(y_normal, dtype=torch.float, requires_grad=False)).cuda()

            grid_x, grid_y = soft_argmax(corr.transpose(-1,-2).view(b, -1, feature_size, feature_size), beta=1e-4, x_normal=x_normal, y_normal=y_normal)
            coarse_flow = torch.cat((grid_x, grid_y), dim=1)
            flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow)  # b 2 14 14 = b 2 self.feature_size self.feature_size
            flow_est = F.interpolate(flow_est, size=(H, W), mode='bilinear', align_corners=True)
            flow_est[:,0,:,:] *= W/feature_size
            flow_est[:,1,:,:] *= H/feature_size 

            save_path = f'./vis/suppl/hp/{args.model}/{curr_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            vis_size = (224, 224)
            source_img_orig = F.interpolate(source_img_orig, size=vis_size, mode='bilinear', align_corners=True)
            target_img_orig = F.interpolate(target_img_orig, size=vis_size, mode='bilinear', align_corners=True)
        
            mask_valid_orig = F.interpolate(mask_valid_orig.float().unsqueeze(0), size=vis_size, mode='bilinear', align_corners=True)
            save_image(source_img_orig, f'{save_path}/{i_batch}_img_src.jpg', normalize=True)
            save_image(target_img_orig, f'{save_path}/{i_batch}_img_tgt.jpg', normalize=True)
            # save_image(mask_valid.float(), f'{save_path}/{curr_id}/{i_batch}_img_mask.jpg', normalize=True)

            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            warped_source_gt = F.interpolate(warped_source_gt, size=vis_size, mode='bilinear', align_corners=True)
            warped_source_est = F.interpolate(warped_source_est, size=vis_size, mode='bilinear', align_corners=True)
            save_image(warped_source_gt, f'{save_path}/{i_batch}_img_warped_src_gt.jpg', normalize=True)
            save_image(warped_source_est*mask_valid_orig, f'{save_path}/{i_batch}_img_warped_src_est.jpg', normalize=True)


        # evaluation protocol
        # eval_img_size -> resize input img to 224 -> model's output_flow 224 -> 
        elif args.model == 'dust3r' or args.model == 'mast3r':

            H_32, W_32 = args.model_img_size
            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.
            source_img_orig = source_img.clone()
            target_img_orig = target_img.clone()
            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std
            source_img = F.interpolate(source_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            output_mode = args.output_mode    # enc_feat, dec_feat, camap
            outputs = network(target_img, source_img, output_mode=output_mode)

            if output_mode == 'enc_feat':
                feats1, feats2 = outputs[0], outputs[1]
                feat1 = feats1[-1]  # b n d 
                feat2 = feats2[-1]
                feat1 = feat1.unsqueeze(0)
                feat2 = feat2.unsqueeze(0)

                l2norm = FeatureL2Norm()    # normalizes along the feature 
                ## from (b, n, d) normalize along the n dimension.
                # feat1 = l2norm(feat1)
                # feat2 = l2norm(feat2)
                # from (b, n, d) normalize along the d dimension.
                feat1 = l2norm(feat1.permute(0,2,1)).permute(0,2,1)    
                feat2 = l2norm(feat2.permute(0,2,1)).permute(0,2,1)
                corr = torch.einsum('bnd, bmd -> bnm', feat1, feat2)

            elif output_mode == 'dec_feat':
                dec_feats1 = [outputs[i][0] for i in range(len(outputs))]
                dec_feats2 = [outputs[i][1] for i in range(len(outputs))]
                dec_feats1.pop(0)
                dec_feats2.pop(0)

                # use the first decoder feature
                dec_feat1 = dec_feats1[0]
                dec_feat2 = dec_feats2[0]
                ## use last decoder feature
                # dec_feat1 = dec_feats1[-1]
                # dec_feat2 = dec_feats2[-1]
                ## use the average of all decoder features
                # dec_feat1 = torch.stack(dec_feats1, dim=1).mean(dim=1)
                # dec_feat2 = torch.stack(dec_feats2, dim=1).mean(dim=1)

                l2norm = FeatureL2Norm()
                ## from (b, n, d) normalize along the n dimension.
                dec_feat1 = l2norm(dec_feat1)
                dec_feat2 = l2norm(dec_feat2)

                # from (b, n, d) normalize along the d dimension.
                # dec_feat1 = l2norm(dec_feat1.permute(0,2,1)).permute(0,2,1)    # b n d
                # dec_feat2 = l2norm(dec_feat2.permute(0,2,1)).permute(0,2,1)
                corr = torch.einsum('bnd, bmd -> bnm', dec_feat1, dec_feat2)    # b 196 196

            elif output_mode == 'ca_map':
                camap1, camap2 = outputs[0], outputs[1]   # b 12 196 196
                camap1 = [attn.mean(dim=1).detach() for attn in camap1]   # b 196 196
                camap2 = [attn.mean(dim=1).detach() for attn in camap2]   # avg heads

                ## heuristic attention visualize
                # for j in range(len(camap1)):
                #     print(camap1[j].argmax(dim=-1))
                # print('-'*50)
                # for j in range(len(camap2)):
                #     print(camap2[j].argmax(dim=-1))
                # breakpoint()

                for i in range(len(camap1)):
                    camap1[i][:,:,0]=camap1[i].min()
                for i in range(len(camap2)):
                    camap2[i][:,:,0]=camap2[i].min()

                # for j in range(len(camap1)):
                #     print(camap1[j].argmax(dim=-1))
                # print('-'*50)
                # for j in range(len(camap2)):
                #     print(camap2[j].argmax(dim=-1))
                # breakpoint()

                camap1 = torch.stack(camap1, dim=1)
                camap2 = torch.stack(camap2, dim=1)
                corr = (camap1.mean(dim=1) + camap2.mean(dim=1).transpose(-1,-2))/2.

                # # heuristic attention refine
                # for i in range(len(camap1)):
                #     camap1[i][:,:,0]=0
                # for i in range(len(camap2)):
                #     camap2[i][:,:,0]=0

                # corr = [ (camap1[i] + camap2[i].transpose(-1,-2))/2 for i in range(len(camap1))]    # b 196 196
                # # avg across layers
                # corr = torch.stack(corr, dim=1).mean(dim=1)    # b 196 196

            else:
                pass

            feature_size = H_32 // 16
            x_normal = np.linspace(-1,1,feature_size)
            x_normal = nn.Parameter(torch.tensor(x_normal, dtype=torch.float, requires_grad=False)).cuda()
            y_normal = np.linspace(-1,1,feature_size)
            y_normal = nn.Parameter(torch.tensor(y_normal, dtype=torch.float, requires_grad=False)).cuda()

            grid_x, grid_y = soft_argmax(corr.transpose(-1,-2).view(b, -1, feature_size, feature_size), beta=1e-4, x_normal=x_normal, y_normal=y_normal)
            coarse_flow = torch.cat((grid_x, grid_y), dim=1)
            flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow)  # b 2 14 14 = b 2 self.feature_size self.feature_size
            flow_est = F.interpolate(flow_est, size=(H, W), mode='bilinear', align_corners=True)
            flow_est[:,0,:,:] *= W/feature_size
            flow_est[:,1,:,:] *= H/feature_size   
            
            save_path = f'./vis/suppl/hp/{args.model}/{curr_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            vis_size = (224, 224)
            source_img_orig = F.interpolate(source_img_orig, size=vis_size, mode='bilinear', align_corners=True)
            target_img_orig = F.interpolate(target_img_orig, size=vis_size, mode='bilinear', align_corners=True)
        
            mask_valid_orig = F.interpolate(mask_valid_orig.float().unsqueeze(0), size=vis_size, mode='bilinear', align_corners=True)
            save_image(source_img_orig, f'{save_path}/{i_batch}_img_src.jpg', normalize=True)
            save_image(target_img_orig, f'{save_path}/{i_batch}_img_tgt.jpg', normalize=True)
            # save_image(mask_valid.float(), f'{save_path}/{curr_id}/{i_batch}_img_mask.jpg', normalize=True)

            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            warped_source_gt = F.interpolate(warped_source_gt, size=vis_size, mode='bilinear', align_corners=True)
            warped_source_est = F.interpolate(warped_source_est, size=vis_size, mode='bilinear', align_corners=True)
            save_image(warped_source_gt, f'{save_path}/{i_batch}_img_warped_src_gt.jpg', normalize=True)
            save_image(warped_source_est*mask_valid_orig, f'{save_path}/{i_batch}_img_warped_src_est.jpg', normalize=True)


        elif args.model == 'crocoflow':

            H_32, W_32 = args.model_img_size

            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.

            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std

            source_img = F.interpolate(source_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(H_32, W_32), mode='bilinear', align_corners=False)
            
            # breakpoint()
            output = network(target_img, source_img)
            flow_est = output[:,:-1,:,:]
            conf = output[:,-1,:,:]

            flow_est = F.interpolate(flow_est, size=(H, W), mode='bilinear', align_corners=True)
            flow_est[:,0,:,:] *= W/224.
            flow_est[:,1,:,:] *= H/224.   

            save_image(source_img, f'./suppl_crocoflow_img_src.jpg', normalize=True)
            save_image(target_img, f'./suppl_crocoflow_img_tgt.jpg', normalize=True)
            save_image(mask_valid.float(), f'./suppl_crocoflow_img_mask.jpg', normalize=True)
            warped_source_gt = warp(source_img, flow_gt)  
            warped_source_est = warp(source_img, flow_est)
            save_image(warped_source_gt, f'./suppl_crocoflow_img_warped_src_gt.jpg', normalize=True)
            save_image(warped_source_est*mask_valid.unsqueeze(1), f'./suppl_crocoflow_img_warped_src_est.jpg', normalize=True)

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

            breakpoint()

        # flow_est = F.interpolate(flow_est, size=source_img.shape[-2:], mode='bilinear', align_corners=False)

        # =========================== log warped imgs to wandb (cursor) ==================================
        if args.dataset == 'eth3d':
            curr_id = 0
            # tmp = source_img.clone()
            # source_img = target_img.clone()
            # target_img = tmp
            
        if args.log_tool == 'wandb' and args.wandb_log_img and curr_id < 5:
            # Log warped images to wandb for first few batches
            if i_batch < wandb_num_log_img and args.log_tool is not None:
                # Warp source image using ground truth and estimated flows
                warped_source_gt = warp(source_img, flow_gt)  
                warped_source_est = warp(source_img, flow_est)
                # Apply mask to warped estimated flow
                warped_source_est_masked = warped_source_est * mask_valid.unsqueeze(1)

                # Create grid of images for visualization
                img_grid = torch.cat([
                    torch.cat([source_img[0], target_img[0]], dim=2),
                    torch.cat([warped_source_gt[0], warped_source_est[0]], dim=2),  # warped_source는 최종적으로 Tgt이미지가 나와야하는 것!
                    torch.cat([mask_valid[0].unsqueeze(0).repeat(3,1,1), # Repeat mask 3 times for RGB channels
                            warped_source_est_masked[0]], dim=2) # Show masked warped estimate in last column
                ], dim=1)
                # Save image grid locally
                # save_image(img_grid.cpu(), f'./tmp.png')

                if args.dataset == 'eth3d':
                    wandb.log({
                        f"vis_warped_flow_{name_dataset}_rate{rate}/img_{i_batch}": wandb.Image(
                            img_grid.cpu(),
                            caption=f"Top: Query | Reference, Middle: Warped (GT) | Warped (Est), Bottom: Valid Mask | Masked Warped (Est), Img_size: {h}x{w}")})
                else:
                    wandb.log({
                        f"vis_warped_flow_{curr_id+1}/img_{i_batch}": wandb.Image(
                            img_grid.cpu(),
                            caption=f"Top: Source | Target, Middle: Warped (GT) | Warped (Est), Bottom: Valid Mask | Masked Warped (Est), Img_size: {h}x{w}")})
            # =========================== Cursor ==================================

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

        if 'crocoflow' in args.model:
            pass
        else:
            if estimate_uncertainty:
                dict_list_uncertainties = compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid,
                                                                        dict_list_uncertainties)

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
    print("Validation EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(mean_epe_list), pck1_dataset,
                                                             pck3_dataset, pck5_dataset))
    if estimate_uncertainty:
        for uncertainty_name in dict_list_uncertainties.keys():
            output['uncertainty_dict_{}'.format(uncertainty_name)] = compute_average_of_uncertainty_metrics(
                dict_list_uncertainties[uncertainty_name])
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
                            path_to_save=None, plot=False, plot_100=False, plot_ind_images=False, sub_data=None, args=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_0_05_list, pck_0_01_list, pck_0_1_list, pck_0_15_list = [], [], [], [], [], []
    dict_list_uncertainties = {}
    eval_buf = {'cls_pck': dict(), 'vpvar': dict(), 'scvar': dict(), 'trncn': dict(), 'occln': dict()}

    # pck curve per image
    pck_thresholds = [0.01]
    pck_thresholds.extend(np.arange(0.05, 0.4, 0.05).tolist())
    pck_per_image_curve = np.zeros((len(pck_thresholds), len(test_dataloader)), np.float32)

    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image'] # b=1 3 600 800 [0,255]
        target_img = mini_batch['target_image'] # b=1 3 600 800
        flow_gt = mini_batch['flow_map'].to(device) # b=1 2 600 800
        mask_valid = mini_batch['correspondence_mask'].to(device) # b=1 600 800

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


        # save_image(img_s, f'img_s.jpg', normalize=True)
        # save_image(img_t, f'img_t.jpg', normalize=True)
        # save_image(mask_valid.float(), f'img_mask.jpg', normalize=True)
        # warped_source_gt = warp(img_s, flow_gt.float())  
        # save_image(warped_source_gt, f'img_warped_src_gt.jpg', normalize=True)
        

        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()    # L_pck=800
        else:
            raise ValueError('No pck threshold in mini_batch')

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

        # =========================== log warped imgs to wandb (cursor) ==================================
        wandb_num_log_img=20
        if args.log_tool == 'wandb':
            # Log warped images to wandb for first few batches
            if i_batch < wandb_num_log_img:
                # Warp source image using ground truth and estimated flows
                source_img = source_img / 255.0     # 1 3 600 800 [0,1]
                target_img = target_img / 255.0
                '''
                    flow_gt: 1 2 600 800
                    flow_est: 1 2 600 800
                '''
                warped_source_gt = warp(source_img, flow_gt)  
                warped_source_est = warp(source_img, flow_est)
                # Apply mask to warped estimated flow
                warped_source_est_masked = warped_source_est * mask_valid.unsqueeze(1)

                # save_image(source_img, './img_s.jpg', normalize=True)
                # save_image(target_img, './img_t.jpg', normalize=True)
                # save_image(warped_source_gt, './img_warped_src_gt.jpg', normalize=True)
                # save_image(warped_source_est, './img_warped_src_est.jpg')

                # breakpoint()

                # Create grid of images for visualization
                img_grid = torch.cat([
                    torch.cat([source_img[0], target_img[0]], dim=2),
                    torch.cat([warped_source_gt[0], warped_source_est[0]], dim=2),  # warped_source는 최종적으로 Tgt이미지가 나와야하는 것!
                    torch.cat([mask_valid[0].unsqueeze(0).repeat(3,1,1), # Repeat mask 3 times for RGB channels
                            warped_source_est_masked[0]], dim=2) # Show masked warped estimate in last column
                ], dim=1)
                # Save image grid locally
                # save_image(img_grid.cpu(), f'./tmp.png')

                wandb.log({
                    f"vis_warped_flow_{sub_data}/img_{i_batch}": wandb.Image(
                        img_grid.cpu(),
                        caption=f"Top: Source | Target, Middle: Warped (GT) | Warped (Est), Bottom: Valid Mask | Masked Warped (Est), Img_size: {orig_H}x{orig_W}")})
            # =========================== Cursor ==================================
        

        if plot_ind_images: # false
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        if plot or (plot_100 and i_batch < 100):# false
            if 'source_kps' in list(mini_batch.keys()):
                # I = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map_padded, R=1.0)
                plot_sparse_keypoints(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est,
                                      mini_batch['source_kps'][0][:, 0], mini_batch['source_kps'][0][:, 1],
                                      mini_batch['target_kps'][0][:, 0], mini_batch['target_kps'][0][:, 1],
                                      uncertainty_comp_est=uncertainty_est)
            else:
                plot_flow_and_uncertainty(path_to_save, 'image_{}'.format(i_batch), source_img, target_img,
                                          flow_gt, flow_est)

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
    print("Validation EPE: %f, alpha=0_01: %f, alpha=0.05: %f" % (output['AEPE'], output['PCK_0_01_per_image'],
                                                                  output['PCK_0_05_per_image']))

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


def run_evaluation_caltech(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=False,
                           path_to_save=None, plot_ind_images=False,):

    def compute_mean(results):
        good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
        filtered_results = np.float64(results)[good_idx]
        return np.mean(filtered_results)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    list_intersection_over_union, list_label_transfer_accuracy, list_localization_error = [], [], []

    for i_batch, mini_batch in pbar:
        mini_batch['nbr'] = i_batch
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        h_src, w_src = mini_batch['source_image_size'][0]
        h_tgt, w_tgt = mini_batch['source_image_size'][0]

        target_mask_np, target_mask = poly_str_to_mask(
            mini_batch['target_kps'][0, :mini_batch['n_pts'][0], 0],
            mini_batch['target_kps'][0, :mini_batch['n_pts'][0], 1], h_tgt, w_tgt)

        source_mask_np, source_mask = poly_str_to_mask(
            mini_batch['source_kps'][0, :mini_batch['n_pts'][0], 0],
            mini_batch['source_kps'][0, :mini_batch['n_pts'][0], 1], h_src, w_src)

        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            if flipping_condition:
                flow_est = network.estimate_flow_with_flipping_condition(source_img, target_img)
            else:
                flow_est = network.estimate_flow(source_img, target_img)

        flow_est = flow_est[:, :, :h_tgt, :w_tgt]  # remove the padding, to original images.

        warped_mask_1 = warp(source_mask, flow_est)

        list_intersection_over_union.append(intersection_over_union(warped_mask_1, target_mask).item())
        list_label_transfer_accuracy.append(label_transfer_accuracy(warped_mask_1, target_mask).item())

        if plot_ind_images:
            mask = None
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est, mask)

    output = {'intersection_over_union': compute_mean(list_intersection_over_union),
              'label_transfer_accuracy': compute_mean(list_label_transfer_accuracy)
              }
    print("Validation IoU: %f, transfer Acc: %f" % (output['intersection_over_union'],
                                                    output['label_transfer_accuracy']))
    return output
