import pdb
from typing import Callable, Optional, Union
import torch
import math
from einops import rearrange

import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging


## from styleaigned
def expand_first(feat: torch.Tensor, scale=1., div=2) -> torch.Tensor:
    b = feat.shape[0]
    stacks = [feat[b // div * i] for i in range(div)]
    feat_style = torch.stack(stacks).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(div, b // div, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // div, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: torch.Tensor, div=2) -> torch.Tensor:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean, div=div)
    feat_style_std = expand_first(feat_std, div=div)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def scaled_dot_product_attention_for_together(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> [torch.Tensor, torch.Tensor]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.get_device())
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    prob_unc, prob_c = attn_weight.chunk(2)
    prob_unc_src, prob_unc_self = prob_unc[1:].chunk(2, dim=-1)
    bb, nhead, npxl = prob_unc_src.shape[0:3]

    prob_unc_src = prob_unc_src.transpose(1, 2).reshape(bb, npxl, -1) / nhead ## batch, npixels, ch
    prob_unc_src_std, prob_unc_src_mean = torch.std_mean(torch.sum(prob_unc_src, -1))
    prob_unc_self = prob_unc_self.transpose(1, 2).reshape(bb, npxl, -1) / nhead  ## batch, npixels, ch
    prob_unc_self_std, prob_unc_self_mean = torch.std_mean(torch.sum(prob_unc_self, -1))

    prob_c_src, prob_c_self = prob_c[1:].chunk(2, dim=-1)
    bb, nhead, npxl = prob_c_src.shape[0:3]

    prob_c_src = prob_c_src.transpose(1, 2).reshape(bb, npxl, -1) / nhead  ## batch, npixels, ch
    prob_c_src_std, prob_c_src_mean = torch.std_mean(torch.sum(prob_c_src, -1))
    prob_c_self = prob_c_self.transpose(1, 2).reshape(bb, npxl, -1) / nhead  ## batch, npixels, ch
    prob_c_self_std, prob_c_self_mean = torch.std_mean(torch.sum(prob_c_self, -1))

    return attn_weight @ value, [prob_unc_src_mean.item(), prob_unc_self_mean.item(), prob_c_src_mean.item(),
                                 prob_c_self_mean.item()], [prob_unc_src_std.item(), prob_unc_self_std.item(),
                                                           prob_c_src_std.item(), prob_c_self_std.item()]

def scaled_dot_product_attention_for_apart(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> [torch.Tensor, torch.Tensor]:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.get_device())
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    prob_unc, prob_c = attn_weight.chunk(2)
    prob_unc_src = prob_unc[1:]
    bb, nhead, npxl = prob_unc_src.shape[0:3]

    prob_unc_src = prob_unc_src.transpose(1, 2).reshape(bb, npxl, -1) / nhead ## batch, npixels, ch
    prob_unc_src_std, prob_unc_src_mean = torch.std_mean(torch.sum(prob_unc_src, -1))

    prob_c_src = prob_c[1:]
    bb, nhead, npxl = prob_c_src.shape[0:3]

    prob_c_src = prob_c_src.transpose(1, 2).reshape(bb, npxl, -1) / nhead  ## batch, npixels, ch
    prob_c_src_std, prob_c_src_mean = torch.std_mean(torch.sum(prob_c_src, -1))

    return attn_weight @ value, [prob_unc_src_mean.item(), prob_c_src_mean.item()], \
        [prob_unc_src_std.item(), prob_c_src_std.item()],

class AttnDevilProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, sain, share_start_tsidx, inference_step, sain_start_tsidx,
                 sain_end_tsidx, guidance_scale, init_tsidx=0, verbose=False,
                 lambda_img=1, lambda_self=1,
                 end_stratified_attention=50
                 ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnSaveProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.sain = sain
        self.cur_tsidx = init_tsidx
        self.inference_step = inference_step
        self.share_start_tsidx = share_start_tsidx
        self.sain_start_tsidx = sain_start_tsidx
        self.sain_end_tsidx = sain_end_tsidx
        self.do_classifier_free_guidance = True if guidance_scale > 1 else False
        self.verbose=verbose
        self.lambda_hs = [lambda_img, lambda_self]    #src, own
        self.end_stratified_attention = end_stratified_attention
        self.sain_mask = None

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        if (hidden_states.shape[0] < 2):    ## inversion
            return self.org_call(attn=attn, hidden_states=hidden_states,
                                 encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask,
                                 temb=temb, scale=scale)
        else:
            if self.sain not in ['none'] and self.sain_start_tsidx <= self.cur_tsidx <= self.sain_end_tsidx:
                if self.cur_tsidx >= self.end_stratified_attention:
                    hidden_states = self.mod_selfattention(attn=attn, hidden_states=hidden_states,
                                                  encoder_hidden_states=encoder_hidden_states,
                                                  attention_mask=attention_mask,
                                                  temb=temb, scale=scale)
                else:   # self.cur_tsidx < self.end_stratified_attention
                    hidden_states = self.stratified_attention(attn=attn, hidden_states=hidden_states,
                                                   encoder_hidden_states=encoder_hidden_states,
                                                   attention_mask=attention_mask,
                                                   temb=temb, scale=scale)
            else:
                hidden_states = self.org_call(attn=attn, hidden_states=hidden_states,
                                               encoder_hidden_states=encoder_hidden_states,
                                               attention_mask=attention_mask,
                                               temb=temb, scale=scale)
            self.cur_tsidx += 1
            if self.cur_tsidx == self.inference_step:
                self.cur_tsidx = 0
            return hidden_states


    def org_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def mod_selfattention(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        n_frame = batch_size // 2 if self.do_classifier_free_guidance else batch_size

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        key = rearrange(key, "(b f) d c -> b f d c", f=n_frame)
        value = rearrange(value, "(b f) d c -> b f d c", f=n_frame)

        sain_mode = "kv_replacement" if self.cur_tsidx < self.share_start_tsidx else "kv_concatenation"

        if self.verbose:
            print(f"{self.cur_tsidx}: {sain_mode}")

        if sain_mode == 'kv_replacement':
            # Conflict-free guidance
            if self.sain == 'cond' and self.do_classifier_free_guidance:
                key[-1] = torch.cat([key[-1, [0] * int(n_frame)]], dim=-2)
                value[-1] = torch.cat([value[-1, [0] * int(n_frame)]], dim=-2)
            # Conflicting guidance
            else:  # self.sain == 'all' or not self.do_classifier_free_guidance:
                key = torch.cat([key[:, [0] * int(n_frame)]], dim=2)
                value = torch.cat([value[:, [0] * int(n_frame)]], dim=2)

        elif sain_mode == 'kv_concatenation':
            # Conflict-free guidance
            if self.sain == 'cond' and self.do_classifier_free_guidance:  ## do not sa injection in uncond
                uc_key = torch.cat([key[:1], key[:1]], dim=-2)
                c_key = torch.cat([key[1:, [0] * int(n_frame)], key[1:]], dim=-2)
                key = torch.cat([uc_key, c_key], 0)

                uc_value = torch.cat([value[:1], value[:1]], dim=-2)
                c_value = torch.cat([value[1:, [0] * int(n_frame)], value[1:]], dim=-2)
                value = torch.cat([uc_value, c_value], 0)
            # Conflicting guidance
            else:  # self.sain == 'all' or not self.do_classifier_free_guidance:
                key = torch.cat([key[:, [0] * int(n_frame)], key], dim=2)
                value = torch.cat([value[:, [0] * int(n_frame)], value], dim=2)

        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def stratified_attention(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        n_frame = batch_size // 2 if self.do_classifier_free_guidance else batch_size

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        sain_mode = "kv_replacement" if self.cur_tsidx < self.share_start_tsidx else "kv_concatenation"
        iters = 1 if sain_mode == "kv_replacement" else 2

        hidden_states = None

        for iii in range(iters):
            if self.do_classifier_free_guidance:
                if iii == 0:    ## from src
                    if self.sain == 'all':  # Conflicting guidance
                        cur_key = torch.cat([key[:1].repeat(n_frame, 1, 1, 1), key[n_frame:n_frame + 1].repeat(n_frame, 1, 1, 1)], dim=0)
                        cur_value = torch.cat([value[:1].repeat(n_frame, 1, 1, 1), value[n_frame:n_frame + 1].repeat(n_frame, 1, 1, 1)], dim=0)
                    else: #self.sain == "cond", Conflict-free guidance
                        cur_key = torch.cat([key[:n_frame], key[n_frame:n_frame+1].repeat(n_frame, 1, 1, 1)], dim=0)
                        cur_value = torch.cat([value[:n_frame], value[n_frame:n_frame+1].repeat(n_frame, 1, 1, 1)], dim=0)
                else:   ## from own
                    cur_key = key
                    cur_value = value
            else:
                if iii == 0:    ## from src
                    cur_key = key[:1].repeat(n_frame, 1, 1, 1)
                    cur_value = value[:1].repeat(n_frame, 1, 1, 1)
                else:   ## from own
                    cur_key = key
                    cur_value = value

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hs = F.scaled_dot_product_attention(
                query, cur_key, cur_value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hs = hs.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            if self.sain_mask is None:
                hs = hs * self.lambda_hs[iii] / sum(self.lambda_hs[:iters])
            else:
                if iii == 0:    ## image prompt
                    hs = hs * self.sain_mask * self.lambda_hs[iii] / sum(self.lambda_hs[:iters])
                else:
                    hs = hs * self.sain_mask * self.lambda_hs[iii] / sum(self.lambda_hs[:iters]) + hs * (1-self.sain_mask)

            hidden_states = hs if hidden_states is None else hidden_states + hs

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states