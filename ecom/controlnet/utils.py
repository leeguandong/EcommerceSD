import torch
import numpy as np
from PIL import Image


def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image


def add_fg(full_img, fg_img, mask_img):
    full_img = np.array(full_img).astype(np.float32)
    fg_img = np.array(fg_img).astype(np.float32)
    mask_img = np.array(mask_img).astype(np.float32) / 255.
    full_img = full_img * mask_img + fg_img * (1 - mask_img)
    return Image.fromarray(np.clip(full_img, 0, 255).astype(np.uint8))
