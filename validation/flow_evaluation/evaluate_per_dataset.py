import numpy as np
import torch
import torch.nn as nn 
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms 


from utils_flow.img_processing_utils import pad_to_same_shape
from validation.flow_evaluation.metrics_uncertainty import (compute_average_of_uncertainty_metrics, compute_aucs,
                                                            compute_uncertainty_per_image)
from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from validation.plot import plot_sparse_keypoints, plot_flow_and_uncertainty, plot_individual_images
from .metrics_segmentation_matching import poly_str_to_mask, intersection_over_union, label_transfer_accuracy
from utils_flow.pixel_wise_mapping import warp
from models.modules.mod import unnormalise_and_convert_mapping_to_flow

from torchvision.utils import save_image 
import torch.nn.functional as F
from einops import rearrange 
import cv2 


def softmax_with_temperature(x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum
    
    
def soft_argmax(corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        # corr: b HsWs Ht Wt 
        b,_,h,w = corr.size()   # b c h w   # b source tg_h tg_w
        
        x_normal = np.linspace(-1,1,w)
        x_normal = nn.Parameter(torch.tensor(x_normal, dtype=torch.float, requires_grad=False)).to('cuda')
        y_normal = np.linspace(-1,1,h)
        y_normal = nn.Parameter(torch.tensor(y_normal, dtype=torch.float, requires_grad=False)).to('cuda')

        corr = softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (tgt hxw) x (src hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y


def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf



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
                         path_to_save=None, plot=False, plot_100=False, plot_ind_images=False):
    out_list, epe_list = [], []
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


def run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty=False, args=None, id=None, k=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []
    dict_list_uncertainties = {}
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image'] # b 3 914 1380
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device) # b 2 914 1380  # target 기준 ? 
        mask_valid = mini_batch['correspondence_mask'].to(device)
        

        # # PHO WARP TEST
        # img_s = mini_batch['source_image'].unsqueeze(dim=0).detach().cpu().float()    #   b 3 914 1380
        # img_t = mini_batch['target_image'].unsqueeze(dim=0).detach().cpu().float()
        # gt_flow = mini_batch['flow_map'].unsqueeze(dim=0).detach().cpu()    # b 2 914 1380
        # save_image(img_s,'./img_s.jpg', normalize=True)
        # save_image(img_t,'./img_t.jpg', normalize=True)
        # b,_,h,w=img_t.shape
        # yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w))
        # grid=torch.stack([xx,yy]).repeat(b,1,1,1)
        # mapping=grid+gt_flow
        # mapping[:,0,:,:]=mapping[:,0,:,:]*2.0/max(w-1,1)-1.0
        # mapping[:,1,:,:]=mapping[:,1,:,:]*2.0/max(h-1,1)-1.0
        # img_t_warped = F.grid_sample(img_s, mapping.permute(0,2,3,1))
        # save_image(img_t_warped, './img_t_warped.jpg', normalize=True)
        
        # # PHO RESIZED TEST
        # trans_resize = transforms.Resize((224,224))
        # img_s224=trans_resize(img_s)
        # img_t224=trans_resize(img_t)
        # save_image(img_s224, './img_s224.jpg', normalize=True)
        # save_image(img_t224, './img_t224.jpg', normalize=True)
        # b,_,h224,w224=img_t224.shape
        # yy,xx=torch.meshgrid(torch.arange(h224), torch.arange(w224))
        # grid224=torch.stack([xx,yy]).repeat(b,1,1,1)
        # flow224 = F.interpolate(gt_flow, (h224,w224), mode='bilinear', align_corners=True)
        # flow224[:,0,:,:] *= (w224/w)
        # flow224[:,1,:,:] *= (h224/h)
        # mapping224=grid224+flow224
        # mapping224[:,0,:,:]=mapping224[:,0,:,:]*2.0/max(w224-1,1)-1.0
        # mapping224[:,1,:,:]=mapping224[:,1,:,:]*2.0/max(h224-1,1)-1.0
        # img_t224_warped=F.grid_sample(img_s224, mapping224.permute(0,2,3,1))
        # save_image(img_t224_warped, './img_t224_warped.jpg', normalize=True)
        # breakpoint()
         
        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        elif 'croco' in args.model or 'dust3r' in args.model:
            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            source_img = source_img.float()/255.
            target_img = target_img.float()/255.

            source_img = (source_img - in1k_mean) / in1k_std
            target_img = (target_img - in1k_mean) / in1k_std

            H_orig, W_orig = flow_gt.shape[2:]
            H_384, W_320 = args.image_shape

            source_img = F.interpolate(source_img, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)
            target_img = F.interpolate(target_img, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)
            
            mask_valid = F.interpolate(mask_valid.float().unsqueeze(0), size=(H_384, W_320), mode='nearest').squeeze(0).bool().to(device)
            flow_gt = F.interpolate(flow_gt, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)    # b 2 380 384
            
            flow_gt[:,0,:,:] *= (W_320/W_orig)
            flow_gt[:,1,:,:] *= (H_384/H_orig)
            
            flow_est, feature1, feature2 = network(source_img, target_img)
            flow_est, flow_est_conf = split_prediction_conf(flow_est, with_conf=True) # b 2 320 384
            
            # flow_est[:,0,:,:] *= (W_320/W_orig)   # 이미 network의 input은 (320,384)에 대해서 받았기에 flow도 이미 이 값들로 scaling되어 있음 
            # flow_est[:,1,:,:] *= (H_384/H_orig)
            flow_est *= -1    
            
            warped_tgt = warp(source_img.detach().cpu(), flow_est.detach().cpu())  
            gt_warped_tgt = warp(source_img.detach().cpu(), flow_gt.detach().cpu())
            
            if id < 5:
                log_img_path=f'{args.save_dir}/ca_out/frame1{k}'
                if not os.path.exists(log_img_path):
                    os.makedirs(log_img_path)
            
                save_image(source_img.detach().cpu(), f'{log_img_path}/{i_batch}_img_src.jpg', normalize=True)
                save_image(target_img, f'{log_img_path}/{i_batch}_img_tgt.jpg', normalize=True)
                save_image(warped_tgt, f'{log_img_path}/{i_batch}_img_wrp_tgt.jpg',              normalize=True)
                save_image(gt_warped_tgt, f'{log_img_path}/{i_batch}_img_wrp_tgt_GT.jpg',        normalize=True)
            
            # def corr(src, trg):
            #     return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
            
            # def l2norm(feature, dim=1):
            #     epsilon = 1e-6
            #     norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
            #     return torch.div(feature, norm)
            # corr_map = corr(l2norm(feature1[-1].permute(0,2,1)), l2norm(feature2[-1].permute(0,2,1))).squeeze()
        
        else:
            flow_est = network.estimate_flow(source_img.unsqueeze(dim=0), target_img.unsqueeze(dim=0))

        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

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


def run_evaluation_generic_camap(network, test_dataloader, device, estimate_uncertainty=False, args=None, id=None, k=None):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    mean_epe_list, epe_all_list, pck_1_list, pck_3_list, pck_5_list = [], [], [], [], []
    dict_list_uncertainties = {}
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image'] # b 3 914 1380
        target_img = mini_batch['target_image']
        flow_gt = mini_batch['flow_map'].to(device) # b 2 914 1380
        mask_valid = mini_batch['correspondence_mask'].to(device)
          
        if estimate_uncertainty:
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        elif 'croco' in args.model or 'dust3r' in args.model:
            in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img_s = source_img.float()/255.
            img_t = target_img.float()/255.

            img_s_orig = (img_s - in1k_mean) / in1k_std      # b 3 H_orig W_orig
            img_t_orig = (img_t - in1k_mean) / in1k_std

            H_orig, W_orig = flow_gt.shape[2:]
            H_384, W_320 = args.image_shape
            
            # vis original src and tgt image 
            save_image(img_s_orig, './img_s_orig.jpg',normalize=True)
            save_image(img_t_orig, './img_t_orig.jpg',normalize=True)
            
            # resize original image to model input size
            img_s = F.interpolate(img_s_orig, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)  # b 3 H_384 W_320
            img_t = F.interpolate(img_t_orig, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)
            mask_valid = F.interpolate(mask_valid.float().unsqueeze(0), size=(H_384, W_320), mode='nearest').squeeze(0).bool().to(device)
            flow_gt = F.interpolate(flow_gt, size=(H_384, W_320), mode='bilinear', align_corners=True).to(device)    # b 2 380 384
            
            flow_gt_H384_W320=flow_gt.clone()
            flow_gt_H384_W320[:,0,:,:] *= (W_320/W_orig)  
            flow_gt_H384_W320[:,1,:,:] *= (H_384/H_orig)
            
            # forward pass
            flow_est, feature1, feature2 = network(img_t, img_s)
            flow_est, flow_est_conf = split_prediction_conf(flow_est, with_conf=True) # b 2 320 384
                    
            ########################## VIS CROSS ATTN MAPS (START) ###############################
            
            p_size=16
            H_24=H_384//p_size  # num patch H
            W_20=W_320//p_size  # num patch W 
            
            sa_maps=[]
            num_enc_blks=24
            for i in range(num_enc_blks):
                sa_maps.append(network.enc_blocks[i].sa_map)
               
            ca_maps=[]
            num_dec_blks=12
            for i in range(num_dec_blks):
                ca_maps.append(network.dec_blocks[i].ca_map)
            
            
            ca_maps = torch.cat(ca_maps)    # num_maps num_heads 480 480
            head_map = ca_maps.mean(dim=0)  # averaged layers
            layer_map = ca_maps.mean(dim=1) # averaged heads
            
            # hook ca_maps that have not gone through softmax
            ca_map = ca_maps.mean(dim=(0,1))    # averaged layers and heads
            ca_map_sft=ca_map.softmax(dim=-1)   
            
            # convert to numpy to use cv2 library (ex.heatmap)
            np_img_s = (img_s-img_s.min()) / (img_s.max()-img_s.min()) * 255.0 # [0,255]
            np_img_t = (img_t-img_t.min())/(img_t.max()-img_t.min()) * 255.0   # [0,255]
            np_img_s = np_img_s.squeeze().permute(1,2,0).detach().cpu().numpy() # 384 320 3
            np_img_t = np_img_t.squeeze().permute(1,2,0).detach().cpu().numpy()
            
            # H_24: 0~23    4 10 16 22
            # W_20: 0~19    7 12 17 23 
            vis_points=[(4,4),(4,12),(10,4),(10,12),(16,4),(16,12)]  # manually assign four points to vis cross attention map
            for points in vis_points:
                idx_h=points[0]     # to vis idx_h
                idx_w=points[1]     # to vis idx_w
                idx_n=idx_h*W_20+idx_w  # to vis token idx
                
                # plot white pixel to vis tkn location
                vis_np_img_t = np_img_t.copy()  # same as clone()
                vis_np_img_t[idx_h*p_size:(idx_h+1)*p_size, idx_w*p_size:(idx_w+1)*p_size,:]=255  
                 
                # generate attn heat map
                attn_msk=ca_map_sft[idx_n]  # 480
                attn_msk=attn_msk.view(1,1,H_24,W_20)
                attn_msk=F.interpolate(attn_msk, size=(H_384,W_320), mode='bilinear', align_corners=True)
                attn_msk=(attn_msk-attn_msk.min())/(attn_msk.max()-attn_msk.min())  # [0,1]
                attn_msk=attn_msk.squeeze().detach().cpu().numpy()*255  # [0,255]
                heat_mask=cv2.applyColorMap(attn_msk.astype(np.uint8), cv2.COLORMAP_JET)
                
                # overlap heat_mask to source image
                img_s_attn_msked = np_img_s[:,:,[2,1,0]] + heat_mask 
                img_s_attn_msked = (img_s_attn_msked-img_s_attn_msked.min())/(img_s_attn_msked.max()-img_s_attn_msked.min())*255.0
                
                if id < 5:
                    log_img_path=f'{args.save_dir}/vis_ca_map/frame1{k}/h{idx_h}_w{idx_w}'
                    if not os.path.exists(log_img_path):
                        os.makedirs(log_img_path)
                        
                    cv2.imwrite(f'{log_img_path}/{i_batch}_img_s.jpg', np_img_s[:,:,[2,1,0]])
                    cv2.imwrite(f'{log_img_path}/{i_batch}_img_t.jpg', vis_np_img_t[:,:,[2,1,0]])  
                    cv2.imwrite(f'{log_img_path}/{i_batch}_img_s_attn_msked.jpg', img_s_attn_msked)
                    
            ########################## VIS CROSS ATTN MAPS (END) ###############################
            
            
            
            ########################## WARP FLOW (START) ###############################
            
            b=1
            yy,xx=torch.meshgrid(torch.arange(H_384),torch.arange(W_320))
            grid=torch.stack([xx,yy]).repeat(b,1,1,1).cuda()
            
            # warped_t using gt_flow
            map_t_gt = grid + flow_gt_H384_W320
            map_t_gt[:,0,:,:]=map_t_gt[:,0,:,:]*2.0/max(W_320,1)-1.0
            map_t_gt[:,1,:,:]=map_t_gt[:,1,:,:]*2.0/max(H_384-1,1)-1.0
            warped_t_gt = F.grid_sample(img_s, map_t_gt.permute(0,2,3,1))
                   
            # warped_t using pred_flow
            map_t_pred = grid + flow_est
            map_t_pred[:,0,:,:]=map_t_pred[:,0,:,:]*2.0/max(W_320-1,1)-1.0
            map_t_pred[:,1,:,:]=map_t_pred[:,1,:,:]*2.0/max(H_384-1,1)-1.0
            warped_t_pred=F.grid_sample(img_s, map_t_pred.permute(0,2,3,1))
                 
            # warped_t using ca_map
            ca_map[:,0]=ca_map.min()
            # ca_map_sft=ca_map.softmax(dim=-1)
            ca_map_sft = rearrange(ca_map.unsqueeze(dim=0), 'b (Ht Wt) (Hs Ws) -> b (Hs Ws) Ht Wt', Hs=H_24, Ws=W_20, Ht=H_24, Wt=W_20)
            BETA=2e-2
            grid_x, grid_y = soft_argmax(ca_map_sft, beta=BETA)   # target aligned
            
            mapping_camap = torch.cat([grid_x,grid_y],dim=1)    # b 2 16 16  
            flow_camap = unnormalise_and_convert_mapping_to_flow(mapping_camap)
            flow_camap = F.interpolate(flow_camap, (H_384,W_320), mode='bilinear', align_corners=True)
            flow_camap[:,0,:,:] *= (W_320/W_20)
            flow_camap[:,1,:,:] *= (H_384/H_24)
            
            map_t_camap=grid+flow_camap
            map_t_camap[:,0,:,:]=map_t_camap[:,0,:,:]*2.0/max(W_320-1,1)-1.0
            map_t_camap[:,1,:,:]=map_t_camap[:,1,:,:]*2.0/max(H_384-1,1)-1.0
            warped_t_camap=F.grid_sample(img_s,map_t_camap.permute(0,2,3,1))
            
            
            if id < 5:
                log_img_path=f'{args.save_dir}/warp_ca_map/BETA{str(BETA)}/frame1{k}'
                if not os.path.exists(log_img_path):
                    os.makedirs(log_img_path)
                    
                save_image(img_s, f'{log_img_path}/{i_batch}_img_s.jpg', normalize=True)
                save_image(img_t, f'{log_img_path}/{i_batch}_img_t.jpg', normalize=True)
                save_image(warped_t_gt, f'{log_img_path}/{i_batch}_img_warped_t_gt.jpg', normalize=True)
                save_image(warped_t_pred, f'{log_img_path}/{i_batch}_img_warped_t_pred.jpg', normalize=True)
                save_image(warped_t_camap*mask_valid, f'{log_img_path}/{i_batch}_img_warped_t_camap.jpg',normalize=True)
 
            ########################## WARP FLOW (END) ###############################
            
            
        else:
            flow_est = network.estimate_flow(source_img, target_img)


        # 
        flow_est=flow_est
        flow_gt=flow_gt_H384_W320
        
 
        flow_est = flow_est.permute(0, 2, 3, 1)[mask_valid]
        flow_gt = flow_gt.permute(0, 2, 3, 1)[mask_valid]

        epe = torch.sum((flow_est - flow_gt) ** 2, dim=1).sqrt()

        epe_all_list.append(epe.view(-1).cpu().numpy())
        mean_epe_list.append(epe.mean().item())
        pck_1_list.append(epe.le(1.0).float().mean().item())
        pck_3_list.append(epe.le(3.0).float().mean().item())
        pck_5_list.append(epe.le(5.0).float().mean().item())

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
                         estimate_uncertainty):
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
            output = run_evaluation_generic(network, test_dataloader, device, estimate_uncertainty)
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
    return dict_results


def run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=False, flipping_condition=False,
                            path_to_save=None, plot=False, plot_100=False, plot_ind_images=False):
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

        if 'pckthres' in list(mini_batch.keys()):
            L_pck = mini_batch['pckthres'][0].float().item()
        else:
            raise ValueError('No pck threshold in mini_batch')

        if estimate_uncertainty:
            if flipping_condition:
                raise NotImplementedError('No flipping condition for PDC-Net yet')
            flow_est, uncertainty_est = network.estimate_flow_and_confidence_map(source_img, target_img)
        else:
            uncertainty_est = None
            if flipping_condition:
                flow_est = network.estimate_flow_with_flipping_condition(source_img, target_img)
            else:
                flow_est = network.estimate_flow(source_img, target_img)
        if plot_ind_images:
            plot_individual_images(path_to_save, 'image_{}'.format(i_batch), source_img, target_img, flow_est)

        if plot or (plot_100 and i_batch < 100):
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
