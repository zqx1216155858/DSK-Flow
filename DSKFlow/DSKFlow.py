import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
import torch.nn.functional as F

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()

    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


import torch
import torch.nn as nn
import torch.nn.functional as F

class DegradationEncoder(nn.Module):
    def __init__(self, in_channels=3, num_tokens=4, embed_dim=1024, latent_res=(64, 64)):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.SiLU())

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, num_tokens * embed_dim)

        self.spatial_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.target_res = latent_res

    def forward(self, x):
        B = x.shape[0]
        # 逐层提取
        feat = self.enc1(x)
        feat = self.enc2(feat)
        feat = self.enc3(feat)
        feat = self.enc4(feat) # [B, 512, H/16, W/16]

        g_feat = self.global_pool(feat).view(B, -1)
        tokens = self.proj(g_feat).view(B, self.num_tokens, self.embed_dim)

        mask = self.spatial_head(feat)
        mask = F.interpolate(mask, size=self.target_res, mode='bilinear', align_corners=False)

        return tokens, mask

crit_fm, crit_ke ,crit_div =  torch.nn.L1Loss(), torch.nn.MSELoss(), torch.nn.MSELoss()

class DSKFlow(torch.nn.Module):
    def __init__(self,  pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae

        sd = torch.load(pretrained_path)
        self.load_ckpt_from_state_dict(sd)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.caption = None
        self.direction = None
        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_ckpt_from_state_dict(self, sd):
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)


    @staticmethod
    def forward_with_networks(x, y=None, direction='a2b', vae_enc=None, unet=None, vae_dec=None,
                              text_emb=None, deg_encoder=None, deg_tokens=None, deg_mask=None,  # 外部传入
                              lambda_fm=1.0, lambda_ke=1.0, lambda_div=1.0, lambda_mi=1.0):

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        x_latent = vae_enc(x, direction=direction).to(dtype)
        latent_size = x_latent.shape[-2:]  # 比如 (64, 64)

        spatial_mask = F.interpolate(deg_mask, size=latent_size, mode='bilinear', align_corners=False)

        if text_emb.shape[0] != B:
            text_emb = text_emb.expand(B, -1, -1)
        combined_emb = torch.cat([text_emb, deg_tokens], dim=1)

        t = torch.rand(B, device=device, dtype=dtype)
        t_broadcast = t[:, None, None, None]

        if y is not None:
            y_latent = vae_enc(y, direction=direction).to(dtype)
            target = y_latent - x_latent
            zt = (1 - t_broadcast) * x_latent + t_broadcast * y_latent

            v_pred = unet(zt, t, encoder_hidden_states=combined_emb).sample

            loss_fm = F.l1_loss(v_pred.float(), target.float()) * lambda_fm

            loss_ke = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_ke > 0:

                ke_weight = 2 - spatial_mask

                loss_ke = torch.mean(ke_weight * (v_pred.float() ** 2))

            loss_div = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_div > 0:
                eta = 1e-2
                epsilon = torch.randn_like(zt)
                v_pred_perturbed = unet(zt + eta * epsilon, t, encoder_hidden_states=combined_emb).sample
                div = torch.sum(epsilon * (v_pred_perturbed.float() - v_pred.float()) / eta, dim=[1, 2, 3])
                loss_div = torch.abs(div).mean()

            loss_mi = torch.tensor(0.0, device=device)
            v_curr_snap, deg_curr_snap = None, None

            if lambda_mi > 0:

                v_curr = F.normalize(F.adaptive_avg_pool2d(v_pred.float(), (4, 4)).view(B, -1), dim=1)
                deg_curr = F.normalize(deg_tokens.float().mean(dim=1), dim=1)

                sim_v = torch.mm(v_curr, deg_encoder.v_queue.t())
                sim_d = torch.mm(deg_curr, deg_encoder.deg_queue.t())
                loss_mi = F.mse_loss(sim_v, sim_d)

                v_curr_snap = v_curr.detach()
                deg_curr_snap = deg_curr.detach()

            return loss_fm, loss_ke, loss_div, loss_mi, v_curr_snap, deg_curr_snap

        else:
            zt = x_latent
            t_zero = torch.zeros(B, device=device, dtype=dtype)
            v_pred = unet(zt, t_zero, encoder_hidden_states=combined_emb).sample
            x_out_latent = x_latent + v_pred
            return vae_dec(x_out_latent, direction=direction)



    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):
        # add all unet parameters
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        for n,p in unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                params_gen.append(p)
        
        # add all vae_a2b parameters
        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x_t, direction=None, caption=None, caption_emb=None):
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc)

def divergence_hutchinson_fd(forward_func, x, eps=1e-3):
    e = torch.randn_like(x)
    e = e / (e.norm() + 1e-8)  # normalize

    v = forward_func(x)
    v_eps = forward_func(x + eps * e)

    div = ((v_eps - v) * e).sum() / eps  # now stable!

    return div