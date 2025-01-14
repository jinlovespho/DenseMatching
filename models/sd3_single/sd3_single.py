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


class SD3Single:
    def __init__(self, args):
        
        self.args = args
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        self.pipe.to("cuda")


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
        generator=None
        scheduler_kwargs = {}
        sigmas=None
        lora_scale=None
        device = self.pipe._execution_device
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
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
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        # JLP - prepare image tensors 
        img1 = img1_info['img1']
        img1_cat = img1_info['img1_cat']
        img1_name = img1_info['img1_name']
        img1_tensor = self.pipe.image_processor.preprocess(img1, height, width) # 1 3 1024 1024
        
        img2_tensor = None 
        if img2_info is not None:
            img2 = img2_info['img2']
            img2_cat = img2_info['img2_cat']
            img2_name = img2_info['img2_name']
            img2_tensor = self.pipe.image_processor.preprocess(img2, height, width) # 1 3 1024 1024
        
        # JLP - prepare image latents
        img1_tensor = img1_tensor.to(device=device, dtype=prompt_embeds.dtype)    # 1 3 768 768
        img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample(generator=generator) # 1 16 96 96 
        img1_latents = img1_latents * self.pipe.vae.config.scaling_factor
        
        img2_latents = None
        if img2_tensor is not None:
            img2_tensor = img2_tensor.to(device=device, dtype=prompt_embeds.dtype)
            img2_latents = self.pipe.vae.encode(img2_tensor).latent_dist.sample(generator=generator)
            img2_latents = img2_latents * self.pipe.vae.config.scaling_factor
        
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
        #     # encode 
        #     img1_tensor = img1_tensor.to(device=device, dtype=torch.float16)    # 1 3 768 768
        #     img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample() # 1 16 96 96 
        #     img1_latents = img1_latents * self.pipe.vae.config.scaling_factor   # # 1 16 96 96 
        #     save_image(img1_tensor, f'./vis/noisy_img/unnorm/img_{img1_info["img1_name"]}.jpg')
            
        #     # add noise 
        #     noise = torch.randn_like(img1_latents)
        #     timestep = t.expand(img1_latents.shape[0])
        #     img1_latents = self.pipe.scheduler.scale_noise(img1_latents, timestep, noise)
            
        #     # decode 
        #     img1_latents = (img1_latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        #     image = self.pipe.vae.decode(img1_latents, return_dict=False)[0]
        #     # image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        #     save_image(image, f'./vis/noisy_img/unnorm/img_{img1_info["img1_name"]}_step{i}_noise_t{round(t.item())}.jpg')

        '''  timesteps: tensor([1000.0000,  987.3806,  974.1077,  960.1293,  945.3875,  929.8179,
                                913.3490,  895.9003,  877.3819,  857.6923,  836.7167,  814.3248,
                                790.3683,  764.6771,  737.0558,  707.2785,  675.0823,  640.1602,
                                602.1506,  560.6250,  515.0721,  464.8760,  409.2888,  347.3926,
                                278.0488,  199.8270,  110.9057,    8.9286], device='cuda:0')
        '''    
        
        # prepare noisy input
        t = timesteps[self.args.inf_stop_step]
        img_latent_model_input = torch.cat([img1_latents] * 2) if do_classifier_free_guidance else img1_latents  # 2 16 96 96            
        noise = torch.randn_like(img_latent_model_input)    # 2 16 128 128
        timestep = t.expand(img_latent_model_input.shape[0])    # 2
        latent_model_input = self.pipe.scheduler.scale_noise(img_latent_model_input, timestep, noise)    # 2 16 96 96 
            
        trans_out = self.pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
            args=self.args,
        )
        
        noise_pred = trans_out[0]   # 2 16 128 128
        my_outputs = trans_out[1]
        
        attn_maps = my_outputs['attn_maps']
        queries = my_outputs['queries']
        keys = my_outputs['keys']
        values = my_outputs['values']
        mmdit_attns = my_outputs['mmdit_attns']
        mmdit_ffs = my_outputs['mmdit_ffs']
        
        if not len(attn_maps) == 0 :
            # print('returning attn_maps')
            feat = torch.stack(attn_maps, dim=0)        # 24 2304 2304 (n_layer, n_img_tkn, n_img_tkn)
            
        elif not len(queries) == 0:
            # print('returning queries')
            feat = torch.stack(queries, dim=0)          # 24 2304 64 (n_layer, n_img_tkn, head_dim)
            
        elif not len(keys) == 0:
            # print('returning keys')
            feat = torch.stack(keys, dim=0)             # 24 2304 64 (n_layer, n_img_tkn, head_dim)
            
        elif not len(values) == 0:
            # print('returning values')
            feat = torch.stack(values, dim=0)           # 24 2304 64 (n_layer, n_img_tkn, head_dim)
            
        elif not len(mmdit_attns) == 0:
            # print('returning mmdit_blk_attn_outputs')
            feat = torch.stack(mmdit_attns, dim=0) # 24 2304 1536 
            
        elif not len(mmdit_ffs) == 0:
            # print('returning mmdit_blk_ff_outputs')
            feat = torch.stack(mmdit_ffs, dim=0)     # 24 2304 1536
        
        else:
            raise ValueError('No feature returned')

        return_idx = self.args.output_layer 
        
        return feat[return_idx:return_idx+1]    # 1 2304 1536
            
        breakpoint()
        VIS_ATTN_MAP = self.args.vis_attn_maps
        if VIS_ATTN_MAP:
            print(t)
            breakpoint()
            attn_maps = trans_out[1]
            attn_maps = torch.stack(attn_maps, dim=0) # 24 4429 4429 (4096+333)
            # save_image(img1_tensor, './img1_tensor.jpg', normalize=True)
            
            if self.args.is_joint:
                seq_img = 2304
                seq_text = 333 
                
                attn_maps_img12 = attn_maps[:, :seq_img, seq_img+seq_text:seq_img+seq_text+seq_img]   # src -> trg 
                attn_maps_img21 = attn_maps[:, seq_img+seq_text:seq_img+seq_text+seq_img, :seq_img]   # trg -> src 
                
                attn_maps_img = attn_maps_img12
                
            else:
                seq_img = 4096
                seq_text = 222

                attn_maps_img = attn_maps[:, :seq_img, :seq_img]    # 24 4096 4096
                attn_maps_text = attn_maps[:, seq_img:, seq_img:]
            
            
            if self.args.is_joint:
                ## VIS ATTN MAPS ## 
                # 1. convert tensor image to numpy 
                img1_tensor_re = (img1_tensor + 1) / 2  # [0,1]
                img1_np = (img1_tensor_re.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) # 1024 1024 3 
                img1_np = np.ascontiguousarray(img1_np) # 1024 1024 3 
                
                img2_tensor_re = (img2_tensor + 1) / 2  # [0,1]
                img2_np = (img2_tensor_re.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) # 1024 1024 3 
                img2_np = np.ascontiguousarray(img2_np) # 1024 1024 3 
                
                # 2. size infos 
                _, _, H, W = img1_tensor.shape 
                ps=16       # not real patch size, but patch size for visualization
                pH = int(seq_img ** 0.5)     # num patches in height 
                pW = int(seq_img ** 0.5)     # num patches in width 
                N = pH * pW  # total num patches 

                # 3. set vis points 
                num_vis=30
                vis_points = torch.rand(N).argsort()[:num_vis].tolist()
                
                vis_layers = [0, 1, 2, 11, 12, 13, 21, 22, 23]
                for l in vis_layers: 
                    for point in vis_points:
                        
                        # set save path 
                        vis_save_path = f'{self.args.save_dir}/attn_maps/{img1_cat}/src{img1_name}_trg{img2_name}/layer{l}'
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        
                        img1_np_vis = img1_np.copy()
                        img2_np_vis = img2_np.copy()
                        
                        # get l-th layer attention map with query_point
                        attn_mask = attn_maps_img[l][point].view(pH,pW)
                        attn_mask = F.interpolate(attn_mask[None, None], size=(H, W), mode='bilinear', align_corners=False).squeeze()
                        attn_mask = (attn_mask-attn_mask.min())/(attn_mask.max()-attn_mask.min())

                        idx_h = point // pW 
                        idx_w = point % pW 
                        
                        # Draw the query point as a circle
                        center = (idx_w*ps + ps//2, idx_h*ps + ps//2)
                        cv2.circle(img1_np_vis, center, ps//2, (0,0,255), -1)
                        cv2.circle(img1_np_vis, center, ps//2, (255,255,255), 2)
                        
                        attn_heatmap = cv2.applyColorMap(np.uint8(255*attn_mask), cv2.COLORMAP_JET)
                        
                        masked_img = img2_np_vis/255. + attn_heatmap/255.
                        masked_img = masked_img / masked_img.max()
                        
                        # Combine source and target images side by side
                        combined_img = np.concatenate([img1_np_vis, np.uint8(255*masked_img)], axis=1)
                        cv2.imwrite(f'{vis_save_path}/src{img1_name}_trg{img2_name}_point{point}.jpg', combined_img)
                
                
            else:
                
                ## VIS ATTN MAPS ## 
                # 1. convert tensor image to numpy 
                img1_tensor_re = (img1_tensor + 1) / 2  # [0,1]
                img1_np = (img1_tensor_re.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) # 1024 1024 3 
                img1_np = np.ascontiguousarray(img1_np) # 1024 1024 3 
                
                # 2. size infos 
                _, _, H, W = img1_tensor.shape 
                ps=16       # not real patch size, but patch size for visualization
                pH = int(seq_img ** 0.5)     # num patches in height 
                pW = int(seq_img ** 0.5)     # num patches in width 
                N = pH * pW  # total num patches 

                # 3. set vis points 
                num_vis=30
                vis_points = torch.rand(N).argsort()[:num_vis].tolist()
                
                vis_layers = [0, 1, 2, 11, 12, 13, 21, 22, 23]
                for l in vis_layers: 
                    for point in vis_points:
                        
                        # set save path 
                        vis_save_path = f'{self.args.save_dir}/attn_maps/{img1_cat}/src{img1_name}/layer{l}'
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        
                        img1_np_vis = img1_np.copy()

                        # get l-th layer attention map with query_point
                        attn_mask = attn_maps_img[l][point].view(pH,pW)
                        attn_mask = F.interpolate(attn_mask[None, None], size=(H, W), mode='bilinear', align_corners=False).squeeze()
                        attn_mask = (attn_mask-attn_mask.min())/(attn_mask.max()-attn_mask.min())

                        idx_h = point // pW 
                        idx_w = point % pW 
                        
                        # Draw the query point as a circle
                        center = (idx_w*ps + ps//2, idx_h*ps + ps//2)
                        cv2.circle(img1_np_vis, center, ps//2, (0,0,255), -1)
                        cv2.circle(img1_np_vis, center, ps//2, (255,255,255), 2)
                        # cv2.imwrite(f'{vis_save_path}/{img1_name}_query_point{point}.jpg', img1_np_vis[...,::-1])
                        
                        attn_heatmap = cv2.applyColorMap(np.uint8(255*attn_mask), cv2.COLORMAP_JET)
                        # cv2.imwrite(f'{vis_save_path}/{img1_name}_heatmap{point}.jpg', attn_heatmap[...,::-1])
                        
                        masked_img = img1_np_vis/255. + attn_heatmap/255. 
                        masked_img = masked_img / masked_img.max()
                        cv2.imwrite(f'{vis_save_path}/{img1_name}_point{point}.jpg', np.uint8(255*masked_img))
            
        # perform guidance
        if self.do_classifier_free_guidance:    # t
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)    # 1 16 128 128
            should_skip_layers = (
                True
                if i > num_inference_steps * skip_layer_guidance_start
                and i < num_inference_steps * skip_layer_guidance_stop
                else False
            )
            print('should_skip_layers', should_skip_layers)
            if skip_guidance_layers is not None and should_skip_layers: # f
                print('skip_guidance_layers')
                timestep = t.expand(latents.shape[0])
                latent_model_input = latents
                noise_pred_skip_layers = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=original_prompt_embeds,
                    pooled_projections=original_pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    skip_layers=skip_guidance_layers,
                )[0]
                noise_pred = (
                    noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                )

        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for k in callback_on_step_end_tensor_inputs:
                callback_kwargs[k] = locals()[k]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
            negative_pooled_prompt_embeds = callback_outputs.pop(
                "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
            )

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            progress_bar.update()

        if XLA_AVAILABLE:
            xm.mark_step()
