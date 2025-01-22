import os  
import cv2
import torch
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
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
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
                height=None,
                width=None,
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
        max_sequence_length=256
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
            prompt_embeds,                      # 1 333 4096
            negative_prompt_embeds,             # 1 333 4096    # none for cfg=False
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
            height = height // 2 
            width = width 
        else:
            height = height
            width = width
            
        # JLP - prepare image tensors 
        img1 = img1_info['img1']
        img1_cat = img1_info['img1_cat']
        img1_name = img1_info['img1_name']
        img1_tensor = self.pipe.image_processor.preprocess(img1, height, width).to(device=device, dtype=prompt_embeds.dtype)     # 1 3 h w 
        
        img2 = img2_info['img2']
        img2_cat = img2_info['img2_cat']
        img2_name = img2_info['img2_name']
        img2_tensor = self.pipe.image_processor.preprocess(img2, height, width).to(device=device, dtype=prompt_embeds.dtype)     # 1 3 h w 
        
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
            for i, t in enumerate(timesteps):
                
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
                )
                
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

                if i+1 == STOP_COUNT:
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
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
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
