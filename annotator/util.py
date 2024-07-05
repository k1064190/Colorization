import numpy as np
import cv2
import os


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    # H, W 중 짧은 쪽을 resolution과 맞춤

    if H > W:
        k = resolution / W
        W = resolution
        H = int(H * k)
    else:
        k = resolution / H
        H = resolution
        W = int(W * k)

    input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_CUBIC)

    # resolution으로 center crop
    start_w = (W - resolution) // 2
    start_h = (H - resolution) // 2
    img = input_image[start_h:start_h + resolution, start_w:start_w + resolution]

    return img
