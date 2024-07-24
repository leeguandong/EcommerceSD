from __future__ import annotations

import gc
import PIL.Image
import torch
import numpy as np
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    HunyuanDiT2DControlNetModel,
    HunyuanDiTControlNetPipeline
)
from PIL import Image
from typing import List, Union, Optional
from ecom.controlnet.preprocessor import Preprocessor
from ecom.configs.config import CONTROLNET_MODEL_IDS, VAE_MODEL_IDS
from ecom.controlnet.utils import make_inpaint_condition, add_fg


class ControlnetDiffusers:
    def __init__(self, base_model_id: str = "runwayml/stable-diffusion-v1-5", task_name: str = "Canny"):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.base_model_id = ""
        self.task_name = ""
        self.pipe = self.load_pipe(base_model_id, task_name)
        self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str, task_name) -> DiffusionPipeline:
        if (
                base_model_id == self.base_model_id
                and task_name == self.task_name
                and hasattr(self, "pipe")
                and self.pipe is not None
        ):
            return self.pipe
        model_id = CONTROLNET_MODEL_IDS[task_name]

        if "ecomxl_softedge" in task_name:
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            vae = AutoencoderKL.from_pretrained(VAE_MODEL_IDS[task_name], torch_dtype=torch.float16)
            # base_model_id:"stabilityai/stable-diffusion-xl-base-1.0"
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                       controlnet=controlnet,
                                                                       vae=vae,
                                                                       torch_dtype=torch.float32)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_vae_slicing()
        elif "ecomxl_inpaint" in task_name:
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            # base_model:"stabilityai/stable-diffusion-xl-base-1.0"
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                       controlnet=controlnet)
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif "hunyuandit_canny" in task_name:
            controlnet = HunyuanDiT2DControlNetModel(model_id, torch_dtype=torch.float16)
            pipe = HunyuanDiTControlNetPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
                                                                controlnet=controlnet,
                                                                torch_dtype=torch.float16)
        else:
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                safety_checker=None,
                controlnet=controlnet,
                torch_dtype=torch.float16
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        try:
            if self.device.type == "cuda":
                pipe.enable_xformers_memory_efficient_attention()
        except ImportError as ee:
            print("xformers not found. Please install xformers to enable memory-efficient attention on CUDA.")

        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_lora(self, lora_path: str, lora_scale=0.7):
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=lora_scale)

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, "controlnet"):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        if "hunyuandit" in task_name:
            controlnet = HunyuanDiTControlNetPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    @torch.autocast("cuda")
    def run_pipe(
            self,
            prompt: str,
            negative_prompt: str,
            control_image: PIL.Image.Image,
            num_images: int,
            num_steps: int,
            guidance_scale: float,
            seed: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            controlnet_conditioning_scale: float = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
    ) -> list[PIL.Image.Image]:
        if seed is None:
            seed = torch.randint(0, 1000000, (1,))
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator,
            image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        ).images

    @torch.inference_mode()
    def process_ecomxl_softedge(self,
                                image,
                                prompt,
                                negative_prompt,
                                num_images,
                                num_steps,
                                guidance_scale,
                                controlnet_conditioning_scale,
                                width,
                                height):
        self.preprocessor.load("PidiNet")
        control_image = self.preprocessor(image, safe=False)
        self.load_controlnet_weight("ecomxl_softedge")
        results = self.run_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width,
            height=height)
        return [control_image] + results

    @torch.inference_mode()
    def process_ecomxl_inpaint(self,
                               image,
                               mask,
                               prompt,
                               negative_prompt,
                               num_images,
                               num_steps,
                               guidance_scale,
                               controlnet_conditioning_scale,
                               width,
                               height):
        mask = Image.fromarray(255 - np.array(mask))
        control_image = make_inpaint_condition(image, mask)
        results = self.run_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width,
            height=height)
        result = add_fg(results[0], image, mask)
        return result

    @torch.inference_mode()
    def process_hunyuandit_canny(self,
                                 image,
                                 prompt,
                                 negative_prompt,
                                 num_images,
                                 num_steps,
                                 guidance_scale,
                                 width,
                                 height,
                                 low_threshold,
                                 high_threshold):
        self.preprocessor.load("Canny")
        control_image = self.preprocessor(image=image, low_threshold=low_threshold, high_threshold=high_threshold)
        control_image = PIL.Image.fromarray(control_image)

        self.load_controlnet_weight("hunyuandit_canny")
        results = self.run_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_steps=num_steps,
            control_image=control_image,
            guidance_scale=guidance_scale,
            height=height,
            width=width)
        return [control_image] + results

    @torch.inference_mode()
    def process_hunyuandit_depth(self,
                                 image,
                                 prompt,
                                 negative_prompt,
                                 num_images,
                                 num_steps,
                                 guidance_scale,
                                 width,
                                 height):
        self.preprocessor.load("depth")
        control_image = self.preprocessor(image=image)
        self.load_controlnet_weight("hunyuandit_depth")

        results = self.run_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_steps=num_steps,
            control_image=control_image,
            guidance_scale=guidance_scale,
            height=height,
            width=width)
        return [control_image] + results

    @torch.inference_mode()
    def process_hunyuandit_pose(self,
                                image,
                                prompt,
                                negative_prompt,
                                num_images,
                                num_steps,
                                guidance_scale,
                                width,
                                height):
        self.preprocessor.load("Openpose")
        control_image = self.preprocessor(image=image, hand_and_face=True, )
        self.load_controlnet_weight("hunyuandit_pose")

        results = self.run_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            num_steps=num_steps,
            control_image=control_image,
            guidance_scale=guidance_scale,
            height=height,
            width=width)
        return [control_image] + results
