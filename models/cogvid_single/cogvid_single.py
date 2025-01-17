import os  
import cv2
import torch
import inspect
from diffusers import CogVideoXPipeline
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
from diffusers.utils import export_to_video
from einops import rearrange

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


class CogVidSingle:
    def __init__(self, args):
        
        self.args = args
        
        self.pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
        self.pipe.to("cuda")

    @torch.no_grad()
    def forward(self, 
                img1_info=None,
                img2_info=None,
                num_frames=None,
                prompt=None,
                negative_prompt="",
                num_inference_steps=None,
                height=None,
                width=None,
                guidance_scale=None,
                do_classifier_free_guidance=True):
        
        
    #     prompt = (
    #     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    #     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    #     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    #     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    #     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    #     "atmosphere of this unique musical performance."
    # )
    #     video = self.pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
    #     export_to_video(video, "output.mp4", fps=8)
        
    #     breakpoint()
        
        height = height or self.pipe.transformer.config.sample_height * self.pipe.vae_scale_factor_spatial    # 480
        width = width or self.pipe.transformer.config.sample_width * self.pipe.vae_scale_factor_spatial      # 720
        num_frames = num_frames     # 2
        num_videos_per_prompt = 1
        batch_size=1
        device = self.pipe._execution_device
        do_classifier_free_guidance = do_classifier_free_guidance
        max_sequence_length=226
        generator=torch.Generator(device=self.pipe._execution_device).manual_seed(self.args.seed)
        
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance: # t
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # JLP - prepare image tensors 
        img1 = img1_info['img1']
        img1_cat = img1_info['img1_cat']
        img1_name = img1_info['img1_name']
        img1_tensor = self.pipe.video_processor.preprocess_video(img1, height=height, width=width)      # 1 3 1 480 720 (b c f h w)
        
        img2_tensor = None 
        if img2_info is not None:
            img2 = img2_info['img2']
            img2_cat = img2_info['img2_cat']
            img2_name = img2_info['img2_name']
            img2_tensor = self.pipe.video_processor.preprocess_video(img2, height=height, width=width)   # 1 3 1 480 720 (b c f h w)
        
        # JLP - prepare image latents
        img1_tensor = img1_tensor.to(device=device, dtype=prompt_embeds.dtype)                      # 1 3 1 480 720
        img1_latents = self.pipe.vae.encode(img1_tensor).latent_dist.sample(generator=generator)    # 1 16 1 60 90 (b c f h w)
        img1_latents = img1_latents * self.pipe.vae.config.scaling_factor
        img1_latents = rearrange(img1_latents, 'b c f h w -> b f c h w')                            # 1 1 16 60 90 (b f c h w)

        img2_latents = None
        if img2_tensor is not None:
            img2_tensor = img2_tensor.to(device=device, dtype=prompt_embeds.dtype)
            img2_latents = self.pipe.vae.encode(img2_tensor).latent_dist.sample(generator=generator)
            img2_latents = img2_latents * self.pipe.vae.config.scaling_factor
            img2_latents = rearrange(img2_latents, 'b c f h w -> b f c h w')


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipe.scheduler, num_inference_steps, device)
        self._num_timesteps = len(timesteps)    # 50
        
        t = timesteps[self.args.inf_stop_step]
        
        # prepare frame input 
        # latent_model_input = torch.cat([img1_latents, img2_latents], dim=1) # frame concatenation   # 1 2 16 60 90
        latent_model_input = img1_latents               # 1 1 16 60 90 (b f c h w)
        noise = torch.randn_like(latent_model_input)    # 1 1 16 60 90
        latent_model_input = self.pipe.scheduler.add_noise(latent_model_input, noise, t) # 1 1 16 60 90
        
        latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input     # 2 1 16 60 90
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)                                   # 2 1 16 60 90
            
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])

        # predict noise model_output
        trans_out = self.pipe.transformer(
            hidden_states=latent_model_input,       # 2 1 16 60 90
            encoder_hidden_states=prompt_embeds,    # 2 226 4096
            timestep=timestep,
            image_rotary_emb=None,
            attention_kwargs=None,
            return_dict=False,
        )

        my_outputs = trans_out[1]
        cogvid_attns = my_outputs['cogvid_attns']
        
        if not len(cogvid_attns) == 0:
            feat = torch.stack(cogvid_attns, dim=0)  # 30 1350 1920 (num_blks, n, d)
        
        return_idx = self.args.output_layer 
        return feat[return_idx:return_idx+1]    # 1 1024 1152
        