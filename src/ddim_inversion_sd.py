# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from typing import Callable
from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import pdb

T = torch.Tensor
# TN = T | None
InversionCallback = Callable[[StableDiffusionPipeline, int, T, dict[str, T]], dict[str, T]]

class Inversion:
    def __init__(self, model:StableDiffusionPipeline, NUM_DDIM_STEPS, dtype):
        self.device = model._execution_device
        self.model = model
        self.num_inference_steps = NUM_DDIM_STEPS
        # self.guidance_scale = GUIDANCE_SCALE
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS, device=self.device)
        self.dtype = dtype

    @torch.no_grad()
    def _get_text_embeddings(self, prompt: str):
        # Tokenize text and get embeddings
        text_inputs = self.model.tokenizer(prompt, padding='max_length',
                                           max_length=self.model.tokenizer.model_max_length,
                                           truncation=True if str != "" else False,
                                           return_tensors='pt')
        text_input_ids = text_inputs.input_ids

        with torch.no_grad():
            prompt_embeds = self.model.text_encoder(
                text_input_ids.to(self.device),
                # output_hidden_states=True,
            )[0]
        return prompt_embeds
    
    
    def _encode_text(self, prompt: str) -> T:
        prompt_embeds = self._get_text_embeddings(prompt)
        return prompt_embeds
    
    
    def _encode_text_with_negative(self, prompt: str) -> T:
        prompt_embeds = self._encode_text(prompt)
        prompt_embeds_uncond = self._encode_text("")
        prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds))
        return prompt_embeds
    
    @torch.no_grad()
    def _encode_image(self, image: np.ndarray) -> T:
        if type(image) is Image:
            image = np.array(image)
        # self.model.vae.to(dtype=torch.float32)
        image = torch.from_numpy(image).float() / 255.
        image = (image * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        latent = self.model.vae.encode(image)['latent_dist'].mean * self.model.vae.config.scaling_factor
        # self.model.vae.to(dtype=torch.float16)
        return latent

    # @torch.no_grad()
    # def _decode_image(self, latents, return_type='np'):
    #     latents = latents.detach() / self.model.vae.config.scaling_factor
    #     image = self.model.vae.decode(latents)['sample']
    #     if return_type == 'np':
    #         image = (image / 2 + 0.5).clamp(0, 1)
    #         image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    #         image = (image * 255).astype(np.uint8)
    #     return image
    
    def _next_step(self, model_output: T, timestep: int, sample: T) -> T:
        timestep, next_timestep = min(timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[int(next_timestep)]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def _get_noise_pred(self, latent: T, t: T, context: T, mask=None, masked_image_latents=None):
        # latents_input = torch.cat([latent] * 2)
        if mask is not None and masked_image_latents is not None:
            latent_model_input = torch.cat([latent, mask, masked_image_latents], dim=1)
        else:
            latent_model_input = latent
        noise_pred = self.model.unet(latent_model_input, t, encoder_hidden_states=context)["sample"]
        # noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
        return noise_pred

    @torch.no_grad()
    def _ddim_loop(self, z0, prompt, mask=None, masked_image_latents=None) -> T:
        text_embedding = self._encode_text(prompt)
        all_latent = [z0]
        latent = z0.clone().detach().to(self.dtype)
        for i in tqdm(range(self.num_inference_steps)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self._get_noise_pred(latent, t, text_embedding, mask, masked_image_latents)
            latent = self._next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return torch.cat(all_latent).flip(0)
    
    @torch.no_grad()
    def ddim_inversion(self, x0: np.ndarray, prompt: str, mask=None, masked_image_latents=None) -> T:
        if isinstance(x0, Image.Image):
            x0 = np.array(x0)
        z0 = self._encode_image(x0)
        # rec = self._decode_image(z0)  # image: (512, 512, 3); latent: torch.Size([1, 4, 64, 64])
        zs = self._ddim_loop(z0, prompt, mask, masked_image_latents)
        return zs

    @staticmethod
    def make_inversion_callback(zts, offset: int = 0) -> [T, InversionCallback]:

        def callback_on_step_end(pipeline: StableDiffusionPipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
            latents = callback_kwargs['latents']
            latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
            return {'latents': latents}
        return zts[offset], callback_on_step_end
