import os 
import json 
import wandb
import torch
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn 

from validation.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum


def soft_argmax(corr, beta=0.02):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    
    _, _, feat_H, feat_W = corr.shape
    
    x_normal = np.linspace(-1,1,feat_W)
    x_normal = nn.Parameter(torch.tensor(x_normal, dtype=torch.float, requires_grad=False))
    y_normal = np.linspace(-1,1,feat_H)
    y_normal = nn.Parameter(torch.tensor(y_normal, dtype=torch.float, requires_grad=False))
    
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


def prepare_spair(dataset_path, test_path,args):
    '''
        dataset_path='/media/dataset1/jinlovespho/github/DenseMatching/data/SPair-71k'
    '''
    
    # copied code from dift  
    all_cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    all_cats.sort()
    cat2json = {}
    
    # val_json_path = 'spair_71k_val-360.json'
    # img_path = f'{dataset_path}/JPEGImages'

    # with open(f'{dataset_path}/{val_json_path}') as f:
    #     val_anns = json.load(f) 
    #     f.close()
    
    # count = 0 
    # for js in json_list:
    #     to_del = True
    #     file=js.split('.')[0]
    #     src_js = file.split('-')[1]
    #     trg_js = file.split('-')[2].split(':')[0]
    #     cat_js = file.split('-')[2].split(':')[1] 
    #     for val_js in val_anns:
    #         src_val_js = val_js['source_path'].split('/')[-1].split('.')[0]
    #         trg_val_js = val_js['target_path'].split('/')[-1].split('.')[0]
    #         cat_val_js = val_js['category']
    #         if src_js == src_val_js and trg_js == trg_val_js and cat_js == cat_val_js:
    #             to_del = False
    #             print('match', count+1)
    #             count += 1
    #     if to_del:
    #         print('not match', count+1)
    #         os.remove(f'{dataset_path}/{test_path}/{js}')
    # print('count: ', count)    
            
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
    
    return all_cats, cat2json, cat2img

def minmax_norm(x):
    """Min-max normalization along the token dimension (n,d) dim=n"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
    
def vis_pca_single_img(dataset_path, img1_info, extracted_feat, img_idx, args):
    
    interp_size = (args.eval_img_size[0], args.eval_img_size[1])   
    img1_name = img1_info['img1_name']
    img1_cat = img1_info['img1_cat']
    
    SAVE_PATH_PCA_SINGLE = f'{args.save_dir}/pca_single/{img1_cat}'
    if not os.path.exists(SAVE_PATH_PCA_SINGLE):
        os.makedirs(SAVE_PATH_PCA_SINGLE)
        
    feat = extracted_feat   # 1 2304 64
    
    if len(feat.shape) == 3:
        b,n,d = feat.shape 
        
        if args.model == 'cogvid_single':
            h = args.eval_img_size[0]//16 
            w = args.eval_img_size[1]//16 
            feat = rearrange(feat, 'b (h w) d -> b d h w', h=h, w=w)
        else:
            h = int(n ** 0.5)
            feat = rearrange(feat, 'b (h w) d -> b d h w', h=h)
    
    _, _, H, W = feat.shape
    
    feat = rearrange(feat, 'b d h w -> b (h w) d')  # 1 2304 1280
    feat = feat.squeeze(0)  # hw d
    
    # Get first PCA component to separate foreground/background
    feat = feat.to(torch.float32).cuda()
    _,_,V = torch.pca_lowrank(feat)
    pca1 = torch.matmul(feat, V[:, :1])
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
    
    # Reshape PCA components back to spatial dimensions
    pca_vis_fg = torch.zeros((H*W, 3), device=feat.device)
    pca_vis_fg[foreground] = pca3_fg_norm
    pca_vis_fg = pca_vis_fg.reshape(H, W, 3).permute(2,0,1)
    pca_vis_fg = F.interpolate(pca_vis_fg.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
    
    pca_vis_full = pca3_full_norm.reshape(H, W, 3).permute(2,0,1)
    pca_vis_full = F.interpolate(pca_vis_full.unsqueeze(0), size=interp_size, mode='bilinear', align_corners=False).squeeze()
    
    # Load and resize original image
    img = Image.open(f'{dataset_path}/JPEGImages/{img1_cat}/{img1_name}.jpg')
    img = img.resize((interp_size[1], interp_size[0]))
    img_tensor = transforms.ToTensor()(img).cuda()
    
    # Concatenate original and both PCA visualizations horizontally
    pca_combined_vis = torch.cat([img_tensor, pca_vis_full, pca_vis_fg], dim=2)

    if args.log_tool == 'wandb':
        # log frequency
        if img_idx < 2:
            wandb.log({f"pca_single/{img1_cat}_{img1_name}": wandb.Image(pca_combined_vis) })
    else:
        # Save visualization using torchvision
        save_image(pca_combined_vis, f'{SAVE_PATH_PCA_SINGLE}/pca_single_{img1_name}.jpg')


def extract_and_save_feats(network, dataset_path, all_cats, cat2img, args):
    
    FEAT_SAVE_PATH = args.feat_save_path
    os.makedirs(FEAT_SAVE_PATH, exist_ok=True)
    
    for cat in tqdm(all_cats):
        output_dict = {}
        image_list = cat2img[cat]
        
        for img_idx, image_path in enumerate(image_list):
            img1 = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
            
            img1_info = {
                'img1': img1,
                'img1_cat': cat,
                'img1_name': image_path.split('.')[0]
                }
            
            if args.model == 'dift_sd':
                extracted_feat = network.forward(   img1, # 1 1280 48 48
                                                    category=cat,
                                                    img_size=args.eval_img_size,
                                                    t=args.t,
                                                    up_ft_index=args.up_ft_index,
                                                    ensemble_size=args.ensemble_size)
            # elif args.model == 'sd3_single':
            #     prompt = f"a photo of a {cat}"
            #     extracted_feat = network.forward(   img1_info=img1_info,
            #                                         img2_info=None,
            #                                         prompt=prompt,
            #                                         negative_prompt="",
            #                                         num_inference_steps=args.inf_max_step,
            #                                         height=args.eval_img_size[0],
            #                                         width=args.eval_img_size[1],
            #                                         guidance_scale=7.0,
            #                                         do_classifier_free_guidance=True)
            elif args.model == 'dit_single':
                extracted_feat = network.forward(   img1_info=img1_info,    
                                                    img2_info=None,
                                                    num_inference_steps=args.inf_max_step,
                                                    height=args.eval_img_size[0],
                                                    width=args.eval_img_size[1],
                                                    guidance_scale=7.0,
                                                    do_classifier_free_guidance=True)
                # extracted_feat: 1 1024 1152
            
            elif args.model == 'cogvid_single':
                prompt = f"a photo of a {cat}"
                extracted_feat = network.forward(   img1_info=img1_info,
                                                    img2_info=None,
                                                    num_frames=1,
                                                    prompt=prompt,
                                                    negative_prompt="",
                                                    num_inference_steps=args.inf_max_step,
                                                    height=args.eval_img_size[0],
                                                    width=args.eval_img_size[1],
                                                    guidance_scale=7.0,
                                                    do_classifier_free_guidance=True)
                # extracted_feat: 1 1350 1920
            else:
                raise ValueError(f'Unknown model: {args.model}')

            if args.VIS_PCA_SINGLE_IMG:
                vis_pca_single_img(dataset_path, img1_info, extracted_feat, img_idx, args)
            
            # save feats per categories
            output_dict[image_path] = extracted_feat
    
        torch.save(output_dict, os.path.join(FEAT_SAVE_PATH, f'{cat}.pth'))
        print(f'MSG: saved feats for {cat}, in {FEAT_SAVE_PATH}')

                         
logging = None
def log(s):
    global logging
    print(s)
    logging.write(s+'\n')
    logging.flush()     
                       
                            
def validate(network, dataset_path, args):

    val_json_path = 'spair_71k_val-360.json'
    img_path = f'{dataset_path}/JPEGImages'
    
    with open(f'{dataset_path}/{val_json_path}') as f:
        val_anns = json.load(f) 
        f.close()

    load_size = (args.eval_img_size[0], args.eval_img_size[1])
    plot_every_n_steps = -1
    pck_threshold = 0.1
    ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
    for j, ann in enumerate(tqdm(val_anns)):
        with torch.no_grad():

            source_points, target_points, src_path, trg_path, category = load_image_pair(ann, load_size, image_path=img_path)
            
            src_imname = f"{ann['source_path'].split('/')[-1].split('.')[0]}"
            trg_imname = f"{ann['target_path'].split('/')[-1].split('.')[0]}"
            
            source_size = ann["source_size"]
            target_size = ann["target_size"]
            
            img_src = Image.open(f"{dataset_path}/JPEGImages/{src_path}")
            img_trg = Image.open(f"{dataset_path}/JPEGImages/{trg_path}")
            
            prompt = f"a photo of a {category}"
                
            img1_info = {
                'img1': img_src,
                'img1_cat': category,
                'img1_name': src_imname
            }
            
            img2_info = {
                'img2': img_trg,
                'img2_cat': category,
                'img2_name': trg_imname
            }
            
            
            if args.model == 'sd3_joint':
                extracted_feat = network.forward(   img1_info=img1_info,
                                                    img2_info=img2_info,
                                                    prompt=prompt,
                                                    negative_prompt="",
                                                    num_inference_steps=args.inf_max_step,
                                                    height=load_size[0],
                                                    width=load_size[1],
                                                    guidance_scale=7.0,
                                                    do_classifier_free_guidance=False)
                
                ''' extracted_feat: 24 2 2304 1536
                '''
            
            feat = extracted_feat[args.output_layer]        # 2 4096 1536
            img1_hyperfeats = feat[0].unsqueeze(0).cuda()   # 1 4096 1536
            img2_hyperfeats = feat[1].unsqueeze(0).cuda()   # 1 4096 1536
            
            if len(img1_hyperfeats.shape) == 3:
                b,n,d = img1_hyperfeats.shape 
                if args.CONCAT_WIDTH:
                    h = load_size[0] // 2 // 16
                    w = load_size[1] // 16
                    img1_hyperfeats = rearrange(img1_hyperfeats, 'b (h w) d -> b d h w', h=h, w=w)
                    img2_hyperfeats = rearrange(img2_hyperfeats, 'b (h w) d -> b d h w', h=h, w=w)
                else: 
                    h = load_size[0] // 16
                    w = load_size[1] // 16
                    img1_hyperfeats = rearrange(img1_hyperfeats, 'b (h w) d -> b d h w', h=h, w=w)  # 1 1536 64 64 
                    img2_hyperfeats = rearrange(img2_hyperfeats, 'b (h w) d -> b d h w', h=h, w=w)
            
            output_size = img1_hyperfeats.shape[-2:]
            
            '''
                img1_hyperfeats: 1 50176 128 128 = 1 d output_size[0], output_size[1]
                img2_hyperfeats: 1 50176 128 128 = 1 d output_size[0], output_size[1]
            '''
            
            # Log NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            predicted_points = predicted_points.detach().cpu().numpy()
            # Rescale to the original image dimensions
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
            val_dist.append(dist)
            val_pck_img.append(pck_img)
            val_pck_bbox.append(pck_bbox)
            ids.append([j] * len(dist))
            
            # if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
            if args.VIS_KPTS_PREDICTION:
                
                SAVE_PATH_KPTS = f'{args.save_dir}/new_kpts/{category}'
                if not os.path.exists(SAVE_PATH_KPTS):
                    os.makedirs(SAVE_PATH_KPTS)
                
                title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
                title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
                source_points = rescale_points(source_points, load_size, source_size)
                draw_correspondences(source_points, target_points, ann, img1_info, img2_info, title=title, radius1=1, radius2=5, save_path=SAVE_PATH_KPTS, iter=j, args=args)

    breakpoint()
    ids = np.concatenate(ids)
    val_dist = np.concatenate(val_dist)
    val_pck_img = np.concatenate(val_pck_img)
    val_pck_bbox = np.concatenate(val_pck_bbox)
    # df = pd.DataFrame({
    #     "id": ids,
    #     "distances": val_dist,
    #     "pck_img": val_pck_img,
    #     "pck_bbox": val_pck_bbox,
    # })
    log(f"val/pck_img: {val_pck_img.sum() / len(val_pck_img)}")
    log(f"val/pck_bbox: {val_pck_bbox.sum() / len(val_pck_bbox)}")
    # wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})
    return val_pck_img.sum() / len(val_pck_img), val_pck_bbox.sum() / len(val_pck_bbox)

    
    
    breakpoint()