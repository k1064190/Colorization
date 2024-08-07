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
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parseargs = argparse.ArgumentParser()
parseargs.add_argument('--model', type=str, default='control_sd15_colorize_epoch=156.ckpt')
args = parseargs.parse_args()
model_path = args.model

feature_extractor = SegformerFeatureExtractor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
segmodel = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")

model = create_model('./models/control_sd15_colorize.yaml').cpu()
model.load_state_dict(load_state_dict(f"./models/{model_path}", location=device))
model = model.to(device)
ddim_sampler = DDIMSampler(model)

def LGB_TO_RGB(gray_image, rgb_image):
    # gray_image [H, W, 1]
    # rgb_image [H, W, 3]

    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = gray_image[:, :, 0]

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, threshold, save_memory=False):
    # center crop image to square
    # H, W, _ = input_image.shape
    # if H > W:
    #     input_image = input_image[(H - W) // 2:(H + W) // 2, :, :]
    # elif W > H:
    #     input_image = input_image[:, (W - H) // 2:(H + W) // 2, :]

    with torch.no_grad():
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        print("img shape: ", img.shape)
        if C == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_map = img[:, :, None]
            print("Gray image shape: ", detected_map.shape)
        control = torch.from_numpy(detected_map.copy()).float().to(device)
        # control = einops.rearrange(control, 'h w c -> 1 c h w')
        print("Control shape: ", control.shape)

        control = control / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        print("Stacked control shape: ", control.shape)
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

        # results의 각 이미지를 mask로 변환
        masks = []
        for result in results:
            inputs = feature_extractor(images=result, return_tensors="pt")
            outputs = segmodel(**inputs)
            logits = outputs.logits
            logits = logits.squeeze(0)
            thresholded = torch.zeros_like(logits)
            thresholded[logits > threshold] = 1
            mask = thresholded[1: ,:, :].sum(dim=0)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = interpolate(mask, size=(H, W), mode='bilinear')
            mask = mask.detach().numpy()
            mask = np.squeeze(mask)
            mask = np.where(mask > threshold, 1, 0)
            masks.append(mask)

        # results의 각 이미지를 mask를 이용해 mask가 0인 부분은 img 즉 흑백 이미지로 변환.
        # img를 channel이 3인 rgb 이미지로 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    # [H, W, 3]
        final = [gray_img * (1 - mask[:, :, None]) + result * mask[:, :, None] for result, mask in zip(results, masks)]

    # mask to 255 img

    mask_img = [mask * 255 for mask in masks]
    return [detected_map.squeeze(-1)] + results + mask_img + final


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Gray Image")
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
                threshold = gr.Slider(label="segmentation threshold", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(share=True)
