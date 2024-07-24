import os
import torch

from diffusers import \
    (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel,
     StableDiffusionInpaintPipelineLegacy,
     EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler,
     DPMSolverMultistepScheduler, PNDMScheduler)

base_dir = "E:/comprehensive_library/EcommerceSD/weights/"
stable_diffusion_dir = os.path.join(base_dir, "stable_diffusion")
lora_model_dir = os.path.join(base_dir, "lora")
controlnet_model_dir = os.path.join(base_dir, "controlnet")
annotators_aux_model_dir = os.path.join(base_dir, "annotators_aux")

available_models = []
available_controlnet = []
available_lora = []


# ---------------------------------------------------- list models ----------------------------------------------------
def list_available_models():
    global available_models
    available_models.clear()
    available_models.append('None')

    if not os.path.exists(stable_diffusion_dir):
        os.makedirs(stable_diffusion_dir, exist_ok=True)
    for dirname in os.listdir(stable_diffusion_dir):
        if os.path.isdir(os.path.join(stable_diffusion_dir, dirname)):
            available_models.append(dirname)


def list_available_controlnet():
    global available_controlnet
    available_controlnet.clear()
    available_controlnet.append('None')

    if not os.path.exists(controlnet_model_dir):
        os.makedirs(controlnet_model_dir, exist_ok=True)
    for dirname in os.listdir(controlnet_model_dir):
        if dirname == 'dpt-large':
            continue
        if os.path.isdir(os.path.join(controlnet_model_dir, dirname)):
            available_controlnet.append(dirname)


def list_lora_models():
    global available_lora
    available_lora.clear()
    available_lora.append('None')

    if not os.path.exists(lora_model_dir):
        os.makedirs(lora_model_dir, exist_ok=True)
    for dirname in os.listdir(lora_model_dir):
        if os.path.isdir(os.path.join(lora_model_dir, dirname)):
            available_lora.append(dirname)


# ------------------------------------------------------ select --------------------------------------------------------
def select_pipe(model_name):
    model_dir = os.path.join(stable_diffusion_dir, model_name)
    if os.path.exists(model_dir):
        pipeline = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh',
                                                           torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh',
                                                           torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    else:
        return None, 'Can not finding the model: ' + model_name + ' in path of: ' + model_dir + '. Please check it and download your model.'


def select_img2img_pipe(model_name):
    model_dir = os.path.join(stable_diffusion_dir, model_name)
    if os.path.exists(model_dir):
        print('load model now')
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh',
                                                                  torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh')
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    else:
        return None, 'Can not finding the model: ' + model_name + ' in path of: ' + model_dir + '. Please check it and download your model.'


def select_inpainting_pipe(model_name):
    model_dir = os.path.join(stable_diffusion_dir, model_name)
    if os.path.exists(model_dir):
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh',
                                                                        torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name == 'pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh')
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    else:
        return None, 'Can not finding the model: ' + model_name + ' in path of: ' + model_dir + '. Please check it and download your model.'


def select_controlnet_pipe(model_name, control_model):
    model_dir = os.path.join(stable_diffusion_dir, model_name)
    controlnet, warning_inf = select_controlnet(control_model)

    if os.path.exists(model_dir):
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_dir, controlnet=controlnet,
                                                                     torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. ' + str(warning_inf) + '. Enjoy it!'

    else:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh',
                                                                     controlnet=controlnet, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline, 'Loading model : ' + model_name + ' done. ' + str(warning_inf) + '. Enjoy it!'


def select_controlnet(control_model):
    control_model_dir = os.path.join(controlnet_model_dir, control_model)
    if os.path.exists(control_model_dir):
        controlnet = ControlNetModel.from_pretrained(control_model_dir, torch_dtype=torch.float16)
        return controlnet, 'The current control model: ' + control_model
    else:
        return None, 'controlnet model: ' + control_model + ' is not exist now, please download it first!'
