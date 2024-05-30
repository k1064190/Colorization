# from share import *
# import config

import cv2
# import einops
# import gradio as gr
import numpy as np
import torch
import random
from PIL import Image

# from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, \
    StableDiffusionControlNetImg2ImgPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, StableDiffusionPipeline

# model = create_model('./models/control_sd15_colorize.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_colorize_epoch=156.ckpt', location='cuda'))
# model = model.cuda()
# ddim_sampler = DDIMSampler(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
precision = torch.bfloat16 if torch.cuda.is_available() else torch.float16
print("Device:", device)

gray_image = cv2.imread('samples/cielo.snap_29.jpg', cv2.IMREAD_GRAYSCALE)
gray_image = cv2.resize(gray_image, (512, 512))
H, W = gray_image.shape
gray_image = torch.from_numpy(gray_image).float().to(device=device) / 255.0
gray_image = gray_image.unsqueeze(0).unsqueeze(0)   # [batch, channel, height, width]

controlnet = ControlNetModel.from_pretrained(pretrained_model_name_or_path="cuda_model", torch_dtype=torch.bfloat16).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.bfloat16).to(device)
pipe.safety_checker = lambda images, clip_input: (images, False)
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.bfloat16).to(device)
scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(20)
pipe.scheduler = scheduler

# pipe.enable_model_cpu_offload()

output = pipe(prompt="girl standing, portrait", image=gray_image, num_inference_steps=20)
image = output.images[0]
print(output)

# output is Image object
# Image to numpy array
image = np.asarray(image)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold,):
#     with torch.no_grad():
#         img = resize_image(input_image, image_resolution)
#         H, W, C = img.shape
#         if C == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             detected_map = img[:, :, None]
#
#         control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
#         control = torch.stack([control for _ in range(num_samples)], dim=0)
#         control = einops.rearrange(control, 'b h w c -> b c h w').clone()
#
#         if seed == -1:
#             seed = random.randint(0, 65535)
#         seed_everything(seed)
#
#         if config.save_memory:
#             model.low_vram_shift(is_diffusing=False)
#
#         cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
#         un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
#         shape = (4, H // 8, W // 8)
#
#         if config.save_memory:
#             model.low_vram_shift(is_diffusing=True)
#
#         model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
#         samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
#                                                      shape, cond, verbose=False, eta=eta,
#                                                      unconditional_guidance_scale=scale,
#                                                      unconditional_conditioning=un_cond)
#
#         if config.save_memory:
#             model.low_vram_shift(is_diffusing=False)
#
#         x_samples = model.decode_first_stage(samples)
#         x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
#
#         results = [x_samples[i] for i in range(num_samples)]
#     return [detected_map.squeeze(-1)] + results
#
#
# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(sources=['upload'], type="numpy")
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button(value="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                 image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
#                 strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
#                 guess_mode = gr.Checkbox(label='Guess Mode', value=False)
#                 low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
#                 high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
#                 seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
#                 eta = gr.Number(label="eta (DDIM)", value=0.0)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
#         with gr.Column():
#             # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
#             result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
#     ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
#     run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
#
#
# block.launch(server_name='localhost', server_port=7860, share=True)
