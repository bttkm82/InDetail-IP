import os
import numpy as np
from PIL import Image

import torch
from diffusers.utils.torch_utils import randn_tensor
from src.utils import logging_info, import_module

def adain_feat(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.

    size = feat.size()
    B, C = size[:2]
    feat_3dim = feat.reshape(B, C, -1)

    if cond_feat.shape[0] == 1:
        cond_feat = torch.cat([cond_feat] * B, 0)

    feat_var = feat_3dim.view(B, C, -1).var(dim=-1) + eps
    feat_std = feat_var.sqrt().view(B, C, 1)
    feat_mean = feat_3dim.view(B, C, -1).mean(dim=-1).view(B, C, 1)

    cond_feat_var = cond_feat.view(B, C, -1).var(dim=-1) + eps
    cond_feat_std = cond_feat_var.sqrt().view(B, C, 1)
    cond_feat_mean = cond_feat.view(B, C, -1).mean(dim=-1).view(B, C, 1)

    feat_3dim = (feat_3dim - feat_mean.expand(feat_3dim.size())) / feat_std.expand(feat_3dim.size())
    out = feat_3dim * cond_feat_std.expand(feat_3dim.size()) + cond_feat_mean.expand(feat_3dim.size())
    return out.reshape(size)

class InDetailIP:
    def __init__(self, sd_pipe, args, device, dtype=torch.float16):
        self.dtype = dtype
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.pipe.scheduler.set_timesteps(args.ddim_step)
        self.timesteps = self.pipe.scheduler.timesteps
        self.set_InDetailIP(args)
        self.set_unet(args)
        self.use_adain = args.use_adain
        self.generator = torch.Generator(self.device).manual_seed(args.seed) if args.seed is not None else None
        Inversion = import_module(args.inversion)
        self.inversion = Inversion(self.pipe, args.ddim_step, dtype=dtype)

    def set_InDetailIP(self, args):
        attn_type = 'attn1'
        block_list = [k for k in self.pipe.unet.attn_processors.keys() if attn_type in k and args.ap_block_dicts in k]

        logging_info("Set attention processor")
        attn_procs = self.pipe.unet.attn_processors
        attn_proccesor = import_module(args.ap_name)
        for name in self.pipe.unet.attn_processors.keys():
            if name in block_list:
                attn_procs[name] = attn_proccesor(
                    **args.ap_kwargs
                )
                logging_info(f"{name}: {attn_procs[name].sain}")
        self.pipe.unet.set_attn_processor(attn_procs)

    ## For controlnet
    def set_unet(self, args):
        if hasattr(args, 'unet'):
            if hasattr(args.unet, 'unet_forward_fn'):
                unet_forward_fn = import_module(args.unet.unet_forward_fn)
                self.pipe.unet.forward = unet_forward_fn(self.pipe.unet)
                self.pipe.unet.all_timesteps = self.timesteps
            for attr_name, attr_value in args.unet.items():
                if attr_name in ['latents_per_ts', 'noisepred_per_ts']:
                    setattr(self.pipe.unet, attr_name, {})
                elif attr_name not in ['unet_forward_fn']:
                    setattr(self.pipe.unet, attr_name, attr_value)

    def generate(
            self,
            pil_image=None,
            prompt=None,
            negative_prompt="",
            num_samples=4,
            inv_prompt=None,
            inv_offset=0,
            **kwargs,
    ):
        neg_inv_prompt = ''
        logging_info(f"inv offset: {inv_offset}")
        logging_info(f"inv: {inv_prompt}")
        logging_info(f"inf: {prompt}")

        x_ts = self.inversion.ddim_inversion(pil_image, prompt=inv_prompt)

        x_t, inversion_callback = self.inversion.make_inversion_callback(x_ts, offset=inv_offset)
        x_t = x_t.unsqueeze(0)

        C = self.pipe.unet.config.in_channels
        X, Y = pil_image.size[1] // 8, pil_image.size[0] // 8
        latents = randn_tensor([num_samples, C, X, Y], device=torch.device(self.device), dtype=self.dtype)
        if self.use_adain:
            latents = adain_feat(latents, x_t)
        latents = torch.cat([x_t, latents], 0)

        prompt = [inv_prompt] + [prompt] * num_samples
        negative_prompt = [neg_inv_prompt] + [negative_prompt] * num_samples

        outs = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            latents=latents,
            generator=self.generator,
            callback_on_step_end=inversion_callback,
            **kwargs,
        )

        gt_rec = Image.fromarray(np.concatenate((pil_image, outs.images[0]), 1))

        return outs.images[1:], gt_rec


