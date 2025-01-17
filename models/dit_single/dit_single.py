import os  
import cv2
import torch
import inspect
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F


class DITSingle:
    def __init__(self, args):
        
        self.args = args
        
        self.pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-512", torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self.pipe = self.pipe.to("cuda")

    @torch.no_grad()
    def forward(self, 
                img1_info=None,
                img2_info=None,
                prompt=None,
                negative_prompt=None,
                num_inference_steps=None,
                height=None,
                width=None,
                guidance_scale=None,
                do_classifier_free_guidance=True):
        
        # # pick words from Imagenet class labels
        # self.pipe.labels  # to print all available words
        # # pick words that exist in ImageNet
        # words = ["white shark", "umbrella"]
        # class_ids = self.pipe.get_label_ids(words)
        generator=torch.Generator(device=self.pipe._execution_device).manual_seed(self.args.seed)
        # output = self.pipe(class_labels=class_ids, num_inference_steps=self.args.inf_max_step, generator=generator)
        
        device = self.pipe._execution_device

        # JLP - prepare image tensors 
        img1 = img1_info['img1']
        img1_cat = img1_info['img1_cat']
        img1_name = img1_info['img1_name']
        img1_tensor = self.pipe.image_processor.preprocess(img1, height, width) # 1 3 512 512
        
        img2_tensor = None 
        if img2_info is not None:
            img2 = img2_info['img2']
            img2_cat = img2_info['img2_cat']
            img2_name = img2_info['img2_name']
            img2_tensor = self.pipe.image_processor.preprocess(img2, height, width) # 1 3 512 512
        
        # JLP - prepare image latents
        img1_tensor = img1_tensor.to(device=device, dtype=torch.float16)    # 1 3 512 512 
        img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample(generator=generator) # 1 4 64 64 
        img1_latents = img1_latents * self.pipe.vae.config.scaling_factor
        
        img2_latents = None
        if img2_tensor is not None:
            img2_tensor = img2_tensor.to(device=device, dtype=torch.float16)    # 1 3 512 512
            img2_latents = self.pipe.vae.encode(img2_tensor).latent_dist.sample(generator=generator) # 1 4 64 64
            img2_latents = img2_latents * self.pipe.vae.config.scaling_factor
        
        batch_size = 1
        latent_size = self.pipe.transformer.config.sample_size  # 64
        latent_channels = self.pipe.transformer.config.in_channels  # 4

        latent_model_input = torch.cat([img1_latents] * 2) if guidance_scale > 1 else img1_latents  # 2 4 64 64

        # class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self.pipe._execution_device)
        class_labels_input = torch.cat([class_null, class_null], 0) if guidance_scale > 1 else None

        # set step values
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        t = timesteps[self.args.inf_stop_step]
        
        # for i, t in enumerate(timesteps): 
        #     # encode 
        #     img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample()   # 1 4 64 64 
        #     img1_latents = img1_latents * self.pipe.vae.config.scaling_factor       # 1 4 64 64 
        #     save_image(img1_tensor, f'./vis/noisy_img/dit_single/unnorm/img_{img1_info["img1_name"]}.jpg', normalize=True)
            
        #     # add noise 
        #     noise = torch.randn_like(img1_latents)      # 1 4 64 64 
        #     timestep = t.expand(img1_latents.shape[0])  # 1 
        #     img1_latents = self.pipe.scheduler.add_noise(img1_latents, noise, timestep)  # 1 4 64 64 
            
        #     # decode
        #     img1_latents = img1_latents / self.pipe.vae.config.scaling_factor   # 1 4 64 64 
        #     samples = self.pipe.vae.decode(img1_latents).sample  # 1 3 512 512 
        #     image = (samples / 2 + 0.5).clamp(0, 1)
        #     # image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        #     save_image(image, f'./vis/noisy_img/dit_single/unnorm/img_{img1_info["img1_name"]}_step{i}_noise_t{round(t.item())}.jpg')
        
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]   # 1 4 64 64 
            latent_model_input = torch.cat([half, half], dim=0)         # 2 4 64 64 
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)  # 2 4 64 64 

        timesteps = t.cuda()
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        trans_out = self.pipe.transformer(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        )
        
        
        my_outputs = trans_out[1]
        dit_attns = my_outputs['dit_attns']
        
        if not len(dit_attns) == 0:
            feat = torch.stack(dit_attns, dim=0)  # 28 1024 1152 (num_blks, n, d)
        
        return_idx = self.args.output_layer 
        return feat[return_idx:return_idx+1]    # 1 1024 1152
        
        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if self.pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = self.pipe.scheduler.step(model_output, t, latent_model_input).prev_sample
