import os  
import cv2
import torch
import torchvision
import inspect
from diffusers import StableDiffusion3Pipeline
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SD3Joint:
    def __init__(self, args):
        
        self.args = args
        
        # hf_token='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'
        # os.environ['HF_TOKEN'] = hf_token
        # self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_sequential_cpu_offload()
        # self.pipe.vae.enable_slicing()
        # self.pipe.vae.enable_tiling()

    @torch.no_grad()
    def forward(self, 
                img1_info=None,
                img2_info=None,
                prompt=None,
                negative_prompt="",
                num_inference_steps=None,
                model_H=None,
                model_W=None,
                guidance_scale=None,
                do_classifier_free_guidance=True):
        
        # self.pipe.args = self.args
        # extracted_feat = self.pipe( img1_info=img1_info,
        #                             img2_info=None,
        #                             prompt=prompt,
        #                             negative_prompt="",
        #                             num_inference_steps=num_inference_steps,
        #                             height=self.args.eval_img_size[0],
        #                             width=self.args.eval_img_size[1],
        #                             guidance_scale=7.0,)
        
        batch_size=1
        num_images_per_prompt=1
        max_sequence_length=77
        skip_guidance_layers=None
        skip_layer_guidance_scale=2.8
        skip_layer_guidance_stop=0.2
        skip_layer_guidance_start=0.01
        generator=torch.Generator(device=self.pipe._execution_device).manual_seed(self.args.seed)
        scheduler_kwargs = {}
        sigmas=None
        lora_scale=None
        device = self.pipe._execution_device
        
        (
            prompt_embeds,                      # 1 154 4096
            negative_prompt_embeds,             # 1 154 4096    # none for cfg=False
            pooled_prompt_embeds,               # 1 2048 
            negative_pooled_prompt_embeds,      # 1 2048        # none for cfg=False
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,           
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        
        if do_classifier_free_guidance:
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)    # 2 2048
        elif self.args.model == 'sd3_single':
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)    # 2 2048
        elif self.args.model == 'sd3_joint':
            pooled_prompt_embeds = pooled_prompt_embeds
        
        if self.args.model == 'sd3_joint':
            model_H = model_H // 2 
            model_W = model_W 
        else:
            model_H = model_H
            model_W = model_W
            
        # JLP - prepare image tensors 
        img1 = img1_info['img1']
        img1_cat = img1_info['img1_cat']
        img1_name = img1_info['img1_name']
        img1_tensor = self.pipe.image_processor.preprocess(img1, model_H, model_W).to(device=device, dtype=prompt_embeds.dtype)     # 1 3 h w 
        
        img2 = img2_info['img2']
        img2_cat = img2_info['img2_cat']
        img2_name = img2_info['img2_name']
        img2_tensor = self.pipe.image_processor.preprocess(img2, model_H, model_W).to(device=device, dtype=prompt_embeds.dtype)     # 1 3 h w 
        
        # Apply sam mask to features 
        if self.args.SAM_MASK_FEAT:
            img1_msk = img1_info['img1_msk'].resize((model_W, model_H))
            img2_msk = img2_info['img2_msk'].resize((model_W, model_H))
            
            img1_msk = torchvision.transforms.ToTensor()(img1_msk)
            img2_msk = torchvision.transforms.ToTensor()(img2_msk)
            
            # save_image(img1_msk, './img1_msk.jpg')
            # save_image(img2_msk, './img2_msk.jpg')
        else:
            img1_msk = None
            img2_msk = None

        
        if self.args.model == 'sd3_joint':
            img_cat = torch.cat([img1_tensor, img2_tensor], dim=-2)     # must concat along height dimension for proper flattening # 1 3 1024 1024 
            img_cat_latents = self.pipe.vae.encode(img_cat).latent_dist.sample(generator=generator)
            img_cat_latents = img_cat_latents * self.pipe.vae.config.scaling_factor
        
        elif self.args.model == 'sd3_single':
            img_stack = torch.cat([img1_tensor, img2_tensor], dim=0)    # 2 3 1024 1024
            img_stack_latents = self.pipe.vae.encode(img_stack).latent_dist.sample(generator=generator)     # 2 3 h w -> 2 16 h//8 w//8
            img_stack_latents = img_stack_latents * self.pipe.vae.config.scaling_factor   

        else:
            img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample(generator=generator)    # 1 16 96 96 
            img1_latents = img1_latents * self.pipe.vae.config.scaling_factor                           # 1 16 96 96 
            
            img2_latents = self.pipe.vae.encode(img2_tensor).latent_dist.sample(generator=generator)    # 1 16 96 96 
            img2_latents = img2_latents * self.pipe.vae.config.scaling_factor                           # 1 16 96 96 
        
        
        # JLP - prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            device,
            sigmas,
            **scheduler_kwargs,
        )
        
        # # JLP - check how noise works in latent space 
        # for i, t in enumerate(timesteps): 
            
        #     # unnorm
        #     save_image(img1_tensor, f'./vis/noisy_img/joint/unnorm/img_{img1_info["img1_name"]}.jpg')
        #     save_image(img2_tensor, f'./vis/noisy_img/joint/unnorm/img_{img2_info["img2_name"]}.jpg')
            
        #     # norm 
        #     save_image(img1_tensor, f'./vis/noisy_img/joint/norm/img_{img1_info["img1_name"]}.jpg', normalize=True)
        #     save_image(img2_tensor, f'./vis/noisy_img/joint/norm/img_{img2_info["img2_name"]}.jpg', normalize=True)
            
        #     # add noise 
        #     img_latents = torch.cat([img1_latents, img2_latents], dim=0)    # 2 16 96 96 
        #     noise = torch.randn_like(img_latents)                           # 2 16 96 96    
        #     timestep = t.expand(img_latents.shape[0])                       # 2 
        #     img_latents = self.pipe.scheduler.scale_noise(img_latents, timestep, noise)    # 2 16 96 96 
            
        #     # decode 
        #     img_latents = (img_latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        #     image = self.pipe.vae.decode(img_latents, return_dict=False)[0]     # 2 3 768 768 
        #     img1, img2 = image.chunk(2, dim=0)
        #     # image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
            
        #     # unnorm
        #     save_image(img1, f'./vis/noisy_img/joint/unnorm/img_{img1_info["img1_name"]}_step{i}_noise_t{round(t.item())}.jpg')
        #     save_image(img2, f'./vis/noisy_img/joint/unnorm/img_{img2_info["img2_name"]}_step{i}_noise_t{round(t.item())}.jpg')
        #     # norm
        #     save_image(img1, f'./vis/noisy_img/joint/norm/img_{img1_info["img1_name"]}_step{i}_noise_t{round(t.item())}.jpg', normalize=True)
        #     save_image(img2, f'./vis/noisy_img/joint/norm/img_{img2_info["img2_name"]}_step{i}_noise_t{round(t.item())}.jpg', normalize=True)


        '''  timesteps: tensor([1000.0000,  987.3806,  974.1077,  960.1293,  945.3875,  929.8179,
                                913.3490,  895.9003,  877.3819,  857.6923,  836.7167,  814.3248,
                                790.3683,  764.6771,  737.0558,  707.2785,  675.0823,  640.1602,
                                602.1506,  560.6250,  515.0721,  464.8760,  409.2888,  347.3926,
                                278.0488,  199.8270,  110.9057,    8.9286], device='cuda:0')
        '''    
        
        # breakpoint()
        if self.args.model == 'sd3_joint':
            img_latent_model_input = img_cat_latents    # 1 16 h//8 w//8
        elif self.args.model == 'sd3_single':
            img_latent_model_input = img_stack_latents  # 2 16 h//8 w//8
        else:
            img_latent_model_input = torch.cat([img1_latents, img2_latents], dim=0)
        
        # prepare noisy input
        t = timesteps[self.args.inf_stop_step]
        noise = torch.randn_like(img_latent_model_input)    # 1 16 h//8 w//8 
        timestep = t.expand(img_latent_model_input.shape[0])    # 1
        latent_model_input = self.pipe.scheduler.scale_noise(img_latent_model_input, timestep, noise)    # 1 16 h//8 w//8 
        
        # ADDED
        timesteps = timesteps[self.args.inf_stop_step:]
        if self.args.inf_step_count == -1:
            STOP_COUNT = len(timesteps)
        else:
            STOP_COUNT = self.args.inf_step_count
        
        latents = latent_model_input    # 1 16 h//8 w//8 
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for time_iter, t in enumerate(timesteps):
                
                print('Current timestep t: ', t.item())
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents   # 1 16 h//8 w//8 
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                trans_out = self.pipe.transformer(
                    hidden_states=latent_model_input,           # 1 16 h//8 w//8 
                    timestep=timestep,                          # 1
                    encoder_hidden_states=prompt_embeds,        # 1 333 4096
                    pooled_projections=pooled_prompt_embeds,    # 1 2048
                    joint_attention_kwargs=None,
                    return_dict=False,
                    args=self.args,
                    img1_msk=img1_msk,
                    img2_msk=img2_msk,
                )
                
                if self.args.VIS_TXT_TO_IMG:
                    
                    if self.args.model == 'sd3_single':     
                        txt_to_imgs = []
                        img_to_txts = [] 
                        
                        for i in range(24):
                            txt_to_img = self.pipe.transformer.transformer_blocks[i].attn.processor.txt_to_img   # 2 24 154 4096
                            img_to_txt = self.pipe.transformer.transformer_blocks[i].attn.processor.img_to_txt   # 2 24 4096 154
                            
                            txt_to_img = txt_to_img.mean(dim=1)     # 2 154 4096
                            img_to_txt = img_to_txt.mean(dim=1)     # 2 4096 154
                            
                            txt_to_imgs.append(txt_to_img)
                            img_to_txts.append(img_to_txt)
                        
                        txt_to_imgs = torch.stack(txt_to_imgs, dim=0)    # 24 2 154 4096
                        img_to_txts = torch.stack(img_to_txts, dim=0)    # 24 2 4096 154
                        
                        txt_to_imgs = txt_to_imgs.mean(dim=0)    # 2 154 4096
                        img_to_txts = img_to_txts.mean(dim=0)    # 2 4096 154
                        
                        txt_to_imgs = (txt_to_imgs + img_to_txts.transpose(-1, -2))/2.0    # 2 154 4096
                        txt_to_imgs = txt_to_imgs.softmax(dim=-1)   # 2 154 4096
                        
                        # set prompt token vis locations
                        clip_ids = self.pipe.tokenizer.encode(prompt)
                        t5_ids = self.pipe.tokenizer_3.encode(prompt)
                        
                        clip_tokens = self.pipe.tokenizer.convert_ids_to_tokens(clip_ids)
                        t5_tokens = self.pipe.tokenizer_3.convert_ids_to_tokens(t5_ids)
                        
                        clip_w1 = clip_tokens[5].split('<')[0]
                        clip_w2 = clip_tokens[6].split('<')[0]
                        
                        t5_w1 = t5_tokens[5].split('<')[0]
                        t5_w2 = t5_tokens[6].split('<')[0]
                        
                        
                        img_h, img_w = img1_tensor.shape[2:]
                        feat_h, feat_w = img_h//16, img_w//16
                        
                        clip_map1 = txt_to_imgs[:,5].view(-1, feat_h, feat_w).unsqueeze(1)   # 2 1 64 64
                        clip_map2 = txt_to_imgs[:,6].view(-1, feat_h, feat_w).unsqueeze(1)
                        
                        t5_map1 = txt_to_imgs[:,77+5].view(-1, feat_h, feat_w).unsqueeze(1)   # 2 1 64 64
                        t5_map2 = txt_to_imgs[:,77+6].view(-1, feat_h, feat_w).unsqueeze(1)
                        
                        clip_map1 = F.interpolate(clip_map1, size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        clip_map2 = F.interpolate(clip_map2, size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        
                        t5_map1 = F.interpolate(t5_map1, size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        t5_map2 = F.interpolate(t5_map2, size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        
                        save_image(clip_map1, f'./tmp_bilinear_{clip_w1}_clip.jpg', normalize=True)
                        save_image(clip_map2, f'./tmp_bilinear_{clip_w2}_clip.jpg', normalize=True)
                        save_image(t5_map1, f'./tmp_bilinear_{t5_w1}_t5.jpg', normalize=True)
                        save_image(t5_map2, f'./tmp_bilinear_{t5_w2}_t5.jpg', normalize=True)
                        
                        breakpoint()
                        
                        
                        map1 = txt_to_imgs[:,5].view(-1, feat_h, feat_w)      # 2 64 64
                        map2 = txt_to_imgs[:,6].view(-1, feat_h, feat_w)      # 2 64 64
                        
                        map1 = F.interpolate(map1.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        map2 = F.interpolate(map2.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 2 1 1024 1024
                        
                        tmp_save_path = f"./vis/spair_SORTED_VALSPLIT360/txt_to_img/{self.args.model}/no_softmax_try3/{self.args.save_dir.split('/')[-1]}/{img1_cat}"
                        if not os.path.exists(tmp_save_path):
                            os.makedirs(tmp_save_path)
                        
                        vis_img1 = (img1_tensor - img1_tensor.min()) / (img1_tensor.max() - img1_tensor.min())
                        vis_img2 = (img2_tensor - img2_tensor.min()) / (img2_tensor.max() - img2_tensor.min())
                        vis = torch.cat([vis_img1, vis_img2], dim=-1)
                        
                        map1[0] = (map1[0] - map1[0].min()) / (map1[0].max() - map1[0].min())
                        map1[1] = (map1[1] - map1[1].min()) / (map1[1].max() - map1[1].min())
                        
                        map2[0] = (map2[0] - map2[0].min()) / (map2[0].max() - map2[0].min())
                        map2[1] = (map2[1] - map2[1].min()) / (map2[1].max() - map2[1].min())
                        
                        map1 = map1.cuda().permute(1,2,0,3).reshape(1, 1, 1024, -1)
                        map2 = map2.cuda().permute(1,2,0,3).reshape(1, 1, 1024, -1)
                        
                        # Repeat map1 along channel dimension to match vis
                        map1 = map1.repeat(1, 3, 1, 1)
                        map2 = map2.repeat(1, 3, 1, 1)
                        
                        # Concatenate vis and map1 along height dimension
                        combined1 = torch.cat([vis, map1], dim=2)
                        combined2 = torch.cat([vis, map2], dim=2)
                        
                        save_image(combined1, f'{tmp_save_path}/{word1}_{img1_name}_{img2_name}.jpg')
                        save_image(combined2, f'{tmp_save_path}/{word2}_{img1_name}_{img2_name}.jpg')
                        
                    elif self.args.model == 'sd3_joint':
                        txt_to_imgs1 = []
                        txt_to_imgs2 = []
                        
                        img1_to_txts = [] 
                        img2_to_txts = [] 
                        
                        for i in range(24):
                            txt_to_img1 = self.pipe.transformer.transformer_blocks[i].attn.processor.txt_to_img1   # 1 24 154 2048
                            txt_to_img2 = self.pipe.transformer.transformer_blocks[i].attn.processor.txt_to_img2   # 1 24 154 2048
                            
                            img1_to_txt = self.pipe.transformer.transformer_blocks[i].attn.processor.img1_to_txt   # 1 24 2048 154
                            img2_to_txt = self.pipe.transformer.transformer_blocks[i].attn.processor.img2_to_txt   # 1 24 2048 154
                            
                            txt_to_img1 = txt_to_img1.mean(dim=1)     # 1 154 2048
                            txt_to_img2 = txt_to_img2.mean(dim=1)     # 1 154 2048
                            
                            img1_to_txt = img1_to_txt.mean(dim=1)     # 1 2048 154
                            img2_to_txt = img2_to_txt.mean(dim=1)     # 1 2048 154
                            
                            txt_to_imgs1.append(txt_to_img1)
                            txt_to_imgs2.append(txt_to_img2)
                            
                            img1_to_txts.append(img1_to_txt)
                            img2_to_txts.append(img2_to_txt)
                        
                        txt_to_imgs1 = torch.stack(txt_to_imgs1, dim=0)    # 24 1 154 2048
                        txt_to_imgs2 = torch.stack(txt_to_imgs2, dim=0)    # 24 1 154 2048
                        
                        img1_to_txts = torch.stack(img1_to_txts, dim=0)    # 24 1 2048 154
                        img2_to_txts = torch.stack(img2_to_txts, dim=0)    # 24 1 2048 154
                        
                        txt_to_imgs1 = txt_to_imgs1.mean(dim=0)    # 1 154 4096
                        txt_to_imgs2 = txt_to_imgs2.mean(dim=0)    # 1 154 4096
                        
                        img1_to_txts = img1_to_txts.mean(dim=0)    # 1 2048 154
                        img2_to_txts = img2_to_txts.mean(dim=0)    # 1 2048 154
                        
                        txt_to_imgs1 = (txt_to_imgs1 + img1_to_txts.transpose(-1, -2))/2.0    # 1 154 2048
                        txt_to_imgs2 = (txt_to_imgs2 + img2_to_txts.transpose(-1, -2))/2.0    # 1 154 2048
                        
                        id1 = self.pipe.tokenizer.encode(prompt)
                        id2 = self.pipe.tokenizer_2.encode(prompt)
                        
                        str1 = self.pipe.tokenizer.convert_ids_to_tokens(id1)
                        str2 = self.pipe.tokenizer_2.convert_ids_to_tokens(id2)
                        
                        img_h, img_w = img1_tensor.shape[2:]    # 512 1024
                        
                        feat_h, feat_w = img_h//16, img_w//16   # 32 64
                        
                        word1 = str1[5].split('<')[0]
                        word2 = str1[6].split('<')[0]
                        
                        word1_map1 = txt_to_imgs1[:,5].view(-1, feat_h, feat_w)      # 1 32 64
                        word2_map1 = txt_to_imgs1[:,6].view(-1, feat_h, feat_w)      # 1 32 64
                        
                        word1_map2 = txt_to_imgs2[:,5].view(-1, feat_h, feat_w)      # 1 32 64
                        word2_map2 = txt_to_imgs2[:,6].view(-1, feat_h, feat_w)      # 1 32 64
                        
                        word1_map1 = F.interpolate(word1_map1.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 1 1 512 1024
                        word2_map1 = F.interpolate(word2_map1.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 1 1 512 1024
                        
                        word1_map2 = F.interpolate(word1_map2.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 1 1 512 1024
                        word2_map2 = F.interpolate(word2_map2.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False)  # 1 1 512 1024
                        
                        tmp_save_path = f"./vis/spair_SORTED_VALSPLIT360/txt_to_img/{self.args.model}/try5/{self.args.save_dir.split('/')[-1]}/{img1_cat}"
                        if not os.path.exists(tmp_save_path):
                            os.makedirs(tmp_save_path)
                        
                        vis_img1 = (img1_tensor - img1_tensor.min()) / (img1_tensor.max() - img1_tensor.min())  # 1 3 512 1024
                        vis_img2 = (img2_tensor - img2_tensor.min()) / (img2_tensor.max() - img2_tensor.min())  # 1 3 512 1024
                        vis = torch.cat([vis_img1, vis_img2], dim=-2)  # 1 3 1024 1024
                        
                        # word1_map1 = (word1_map1 - word1_map1.min()) / (word1_map1.max() - word1_map1.min())  # 1 1 512 1024
                        # word2_map1 = (word2_map1 - word2_map1.min()) / (word2_map1.max() - word2_map1.min())
                        
                        # word1_map2 = (word1_map2 - word1_map2.min()) / (word1_map2.max() - word1_map2.min())
                        # word2_map2 = (word2_map2 - word2_map2.min()) / (word2_map2.max() - word2_map2.min())
                        
                        word1_vis = torch.cat([word1_map1, word1_map2], dim=-2).cuda()  # 1 1 1024 1024
                        word2_vis = torch.cat([word2_map1, word2_map2], dim=-2).cuda()  # 1 1 1024 1024
                        
                        # Save original attention maps
                        word1_attn = (word1_vis - word1_vis.min()) / (word1_vis.max() - word1_vis.min())  # 1 1 1024 1024
                        word2_attn = (word2_vis - word2_vis.min()) / (word2_vis.max() - word2_vis.min())  # 1 1 1024 1024
                        
                        # Save attention maps with original images
                        word1_attn_vis = torch.cat([vis, word1_attn.repeat(1, 3, 1, 1)], dim=2)
                        word2_attn_vis = torch.cat([vis, word2_attn.repeat(1, 3, 1, 1)], dim=2)
                        save_image(word1_attn_vis, f'{tmp_save_path}/{word1}_{img1_name}_{img2_name}.jpg')
                        save_image(word2_attn_vis, f'{tmp_save_path}/{word2}_{img1_name}_{img2_name}.jpg')
                        
                        # # Create binary masks using Otsu's thresholding method and fill holes
                        # def otsu_threshold(tensor):
                        #     # Flatten and convert to numpy for histogram
                        #     flat_tensor = tensor.view(-1).cpu().numpy()
                            
                        #     # Use scikit-image's efficient implementation of Otsu's method
                        #     from skimage.filters import threshold_otsu
                        #     from scipy import ndimage
                        #     threshold = threshold_otsu(flat_tensor)
                            
                        #     # Create initial binary mask
                        #     binary_mask = (tensor > threshold).float()
                            
                        #     # Convert to numpy for morphological operations
                        #     binary_np = binary_mask.squeeze().cpu().numpy()
                            
                        #     # Fill holes in the binary mask
                        #     filled = ndimage.binary_fill_holes(binary_np)
                            
                        #     # Convert back to tensor
                        #     filled_tensor = torch.from_numpy(filled.astype(float)).to(tensor.device)
                        #     filled_tensor = filled_tensor.view(tensor.shape)
                            
                        #     return filled_tensor

                        # # Apply Otsu's thresholding to get binary masks directly
                        # word1_vis_binary = otsu_threshold(word1_attn)  # 1 1 1024 1024
                        # word2_vis_binary = otsu_threshold(word2_attn)  # 1 1 1024 1024
                        
                        # # Save binary masks
                        # word1_binary_vis = torch.cat([vis, word1_vis_binary.repeat(1, 3, 1, 1)], dim=2)
                        # word2_binary_vis = torch.cat([vis, word2_vis_binary.repeat(1, 3, 1, 1)], dim=2)
                        # save_image(word1_binary_vis, f'{tmp_save_path}/{word1}_{img1_name}_{img2_name}_otsu_binary.jpg')
                        # save_image(word2_binary_vis, f'{tmp_save_path}/{word2}_{img1_name}_{img2_name}_otsu_binary.jpg')
                    
                noise_pred = trans_out[0]   # 1 16 128 128
                my_outputs = trans_out[1]
                
                # trans_out = self.pipe.transformer(
                #     hidden_states=latent_model_input,           # 2 16 h//8 w//8 
                #     timestep=timestep,                          # 2
                #     encoder_hidden_states=prompt_embeds,        # 1 333 4096
                #     pooled_projections=pooled_prompt_embeds,    # 2 2048
                #     joint_attention_kwargs=None,
                #     return_dict=False,
                #     args=self.args,
                # )

                if time_iter+1 == STOP_COUNT:
                    for key, value in my_outputs.items():
                        if not len(my_outputs[key]) == 0:
                            assert self.args.output_feat_type == key, f'args.output_feat_type is {self.args.output_feat_type} while EXTRACTED FEAT is {key}'
                            feat = torch.stack(my_outputs[key], dim=0)
                            print(f'ARGS.OUTPUT_FEAT_TYPE: {self.args.output_feat_type}')
                            print(f'EXTRACTED FEAT:        {key}')
                            print(f'FEAT SHAPE:            {feat.shape}')
                    
                    return feat
            
                # perform guidance
                if do_classifier_free_guidance:    # t
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)    # 1 16 128 128
                    should_skip_layers = (
                        True
                        if time_iter > num_inference_steps * skip_layer_guidance_start
                        and time_iter < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    print('should_skip_layers', should_skip_layers)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
