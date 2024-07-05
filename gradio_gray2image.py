import os
from PIL import Image
import json
import random

import cv2
import einops
import gradio as gr
import numpy as np
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import torch.nn as nn
from torch.nn.functional import threshold, normalize,interpolate
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import SamModel,SamProcessor,SamVisionConfig,SamConfig



# XFORMERS_AVAILABLE = True
# if XFORMERS_AVAILABLE:
#     from modules.sd_hijack_optimizations import SdOptimizationXformers
#     opt = SdOptimizationXformers()
#     opt.apply()

model = create_model('./models/control_sd15_colorize.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_colorize_epoch=156.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

class CustomSamModel(SamModel):
    def __init__(self, config):
        super(CustomSamModel, self).__init__(config)

        # vision_encoder.patch_embed.projection 레이어를 수정합니다.
        original_projection = self.vision_encoder.patch_embed.projection
        bias = original_projection.bias is not None
        self.vision_encoder.patch_embed.projection = nn.Conv2d(
            in_channels=1,  # 흑백 이미지의 채널 수
            out_channels=original_projection.out_channels,
            kernel_size=original_projection.kernel_size,
            stride=original_projection.stride,
            padding=original_projection.padding,
            bias=bias
        )

config = SamVisionConfig(num_channels=1, image_size=1024)
config = SamConfig(vision_config = config)

sam_model = CustomSamModel(config)
sam_model.load_state_dict(torch.load('./models/sam_state_dict.pth'))
sam_model.cuda()
sam_model.eval()


def LGB_TO_RGB(gray_image, rgb_image):
    # gray_image [H, W, 1]
    # rgb_image [H, W, 3]

    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = gray_image[:, :, 0]

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, save_memory=False):
    # center crop image to square
    # H, W, _ = input_image.shape
    # if H > W:
    #     input_image = input_image[(H - W) // 2:(H + W) // 2, :, :]
    # elif W > H:
    #     input_image = input_image[:, (W - H) // 2:(H + W) // 2, :]

    with torch.no_grad():
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if C == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_map = img[:, :, None]
        control = torch.from_numpy(detected_map.copy()).float().cuda()
        control = einops.rearrange(control, 'h w c -> 1 c h w')

        with torch.no_grad():
            pred = sam_model(control)
            masks = pred.pred_masks # [1, 1, c, h, w]
        masks = einops.rearrange(masks, '1 1 c h w -> h w c')

        control = control / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        results = [LGB_TO_RGB(detected_map, result) for result in results]
    return [detected_map.squeeze(-1)] + results + [masks]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(share=True)
