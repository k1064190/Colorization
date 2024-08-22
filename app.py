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
from torch.nn.functional import threshold, normalize, interpolate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from einops import rearrange, repeat

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parseargs = argparse.ArgumentParser()
parseargs.add_argument('--pretrained_model', type=str, default='runwayml/stable-diffusion-v1-5')
parseargs.add_argument('--controlnet', type=str, default='controlnet')
parseargs.add_argument('--precision', type=str, default='fp32')
args = parseargs.parse_args()
pretrained_model = args.pretrained_model

# Check for different hardware architectures
if torch.cuda.is_available():
    device = "cuda"
    # Check for xformers
    try:
        import xformers

        enable_xformers = True
    except ImportError:
        enable_xformers = False
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load models
if args.precision == 'fp32':
    torch_dtype = torch.float32
elif args.precision == 'fp16':
    torch_dtype = torch.float16
elif args.precision == 'bf16':
    torch_dtype = torch.bfloat16
else:
    raise ValueError(f"Invalid precision: {args.precision}")

controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch_dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    args.pretrained_model, controlnet=controlnet, torch_dtype=torch_dtype
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Apply optimizations based on hardware
if device == "cuda":
    pipe = pipe.to(device)
    if enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers optimization enabled")
elif device == "mps":
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print("Attention slicing enabled for Apple Silicon")
else:
    # CPU-specific optimizations
    pipe = pipe.to(device)
    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_attention_slicing()

feature_extractor = SegformerFeatureExtractor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
segmodel = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")


def LGB_TO_RGB(gray_image, rgb_image):
    # gray_image [H, W, 3]
    # rgb_image [H, W, 3]

    print("gray_image shape: ", gray_image.shape)
    print("rgb_image shape: ", rgb_image.shape)

    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = gray_image[:, :]

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


@torch.inference_mode()
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength,
            guidance_scale, seed, eta, threshold, save_memory=False):
    with torch.no_grad():
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        print("img shape: ", img.shape)
        if C == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        control = torch.from_numpy(img).to(device).float()
        control = control / 255.0
        control = rearrange(control, 'h w c -> 1 c h w')
        # control = repeat(control, 'b c h w -> b c h w', b=num_samples)
        # control = rearrange(control, 'b h w c -> b c h w')

        if a_prompt:
            prompt = prompt + ', ' + a_prompt

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        generator = torch.Generator(device=device).manual_seed(seed)
        # Generate images
        output = pipe(
            num_images_per_prompt=num_samples,
            prompt=prompt,
            image=control.to(device),
            negative_prompt=n_prompt,
            num_inference_steps=ddim_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            eta=eta,
            strength=strength,
            output_type='np',

        ).images

        # output = einops.rearrange(output, 'b c h w -> b h w c')
        output = (output * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

        results = [output[i] for i in range(num_samples)]
        results = [LGB_TO_RGB(img, result) for result in results]

        # results의 각 이미지를 mask로 변환
        masks = []
        for result in results:
            inputs = feature_extractor(images=result, return_tensors="pt")
            outputs = segmodel(**inputs)
            logits = outputs.logits
            logits = logits.squeeze(0)
            thresholded = torch.zeros_like(logits)
            thresholded[logits > threshold] = 1
            mask = thresholded[1:, :, :].sum(dim=0)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = interpolate(mask, size=(H, W), mode='bilinear')
            mask = mask.detach().numpy()
            mask = np.squeeze(mask)
            mask = np.where(mask > threshold, 1, 0)
            masks.append(mask)

        # results의 각 이미지를 mask를 이용해 mask가 0인 부분은 img 즉 흑백 이미지로 변환.
        # img를 channel이 3인 rgb 이미지로 변환
        final = [img * (1 - mask[:, :, None]) + result * mask[:, :, None] for result, mask in zip(results, masks)]

    # mask to 255 img

    mask_img = [mask * 255 for mask in masks]
    return [img] + results + mask_img + final


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
                num_samples = gr.Slider(label="Images", minimum=1, maximum=1, value=1, step=1, visible=False)
                # num_samples = 1
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                # guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=20, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                threshold = gr.Slider(label="Segmentation Threshold", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed,
           eta, threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

block.queue(concurrency_count=4, max_size=100)
block.launch(share=True)
