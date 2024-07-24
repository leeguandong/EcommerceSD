# coding:utf-8
import os
import cv2
import torch
import numpy as np
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation
from ui.utils.list_models import annotators_aux_model_dir


def to_canny(image):
    low_threshold = 100
    high_threshold = 200

    if type(image) is Image.Image:
        image = np.array(image)
    if type(image) is np.ndarray:
        print('image type after: {}'.format(type(image)))
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
    else:
        return ''


def to_depth(image):
    processor = DPTImageProcessor.from_pretrained(os.path.join(annotators_aux_model_dir, 'dpt-large'))
    model = DPTForDepthEstimation.from_pretrained(os.path.join(annotators_aux_model_dir, 'dpt-large'))
    image = Image.fromarray(image)

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    image = Image.fromarray(formatted)

    return image
