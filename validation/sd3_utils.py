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

def prepare_spair(dataset_path, args):
    
    # copied code from dift  
    all_cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
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
    img = img.resize(interp_size)
    img_tensor = transforms.ToTensor()(img).cuda()
    
    # Concatenate original and both PCA visualizations horizontally
    pca_combined_vis = torch.cat([img_tensor, pca_vis_fg, pca_vis_full], dim=2)

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
            elif args.model == 'sd3_single':
                prompt = f"a photo of a {cat}"
                extracted_feat = network.forward(   img1_info=img1_info,
                                                    img2_info=None,
                                                    prompt=prompt,
                                                    negative_prompt="",
                                                    num_inference_steps=args.inf_max_step,
                                                    height=args.eval_img_size[0],
                                                    width=args.eval_img_size[1],
                                                    guidance_scale=7.0,
                                                    do_classifier_free_guidance=True)
            else:
                raise ValueError(f'Unknown model: {args.model}')

            if args.VIS_PCA_SINGLE_IMG:
                vis_pca_single_img(dataset_path, img1_info, extracted_feat, img_idx, args)
            
            # save feats per categories
            output_dict[image_path] = extracted_feat
    
        torch.save(output_dict, os.path.join(FEAT_SAVE_PATH, f'{cat}.pth'))
        print(f'MSG: saved feats for {cat}, in {FEAT_SAVE_PATH}')
                            