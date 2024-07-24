import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, DDIMScheduler, DDPMScheduler, \
    EulerDiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler
from ui.utils.list_models import select_pipe, select_img2img_pipe, select_inpainting_pipe, select_controlnet_pipe, \
    lora_model_dir, stable_diffusion_dir
from ui.utils.utils import to_canny, to_depth


# ----------------------------------------------------- refresh -----------------------------------------------------
def refresh_model(model_name, tag, control_model, lora, whether_lora):
    if not model_name:
        return 'Select the model you need.', gr.Slider.update()
    if not (control_model == 'txt2img' or control_model == 'img2img' or control_model == 'inpainting'):
        if not control_model:
            return 'The current selected base model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update()
        if control_model == 'None':
            return 'The current selected base model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update()

    if model_name == 'pai-diffusion-anime-large-zh':
        return refresh_checkpoints(model_name, tag, control_model, lora, whether_lora), gr.Slider.update(value=768)
    else:
        return refresh_checkpoints(model_name, tag, control_model, lora, whether_lora), gr.Slider.update(value=512)


def refresh_checkpoints(model_name, tag, control_model, lora, whether_lora):
    global pipe

    if tag == 'txt2img':
        pipe, warning_inf = select_pipe(model_name)

        if not lora == 'None' and lora:
            lora_path = os.path.join(lora_model_dir, lora)

            pipe.unet.load_attn_procs(lora_path)
            if not model_name == 'pai-diffusion-artist-large-zh':
                warning_inf_lora = 'The current lora model: ' + str(
                    lora) + '. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d ' \
                            ' are based on the pai-diffusion-artist-large-zh, choose other models may cause unexpected outputs.'
            else:
                warning_inf_lora = 'The current lora model: ' + str(lora) + '.'
            warning_inf = warning_inf + ' ' + warning_inf_lora
        return warning_inf

    if tag == 'img2img':
        pipe, warning_inf = select_img2img_pipe(model_name)
        if not lora == 'None' and lora:
            lora_path = os.path.join(lora_model_dir, lora)

            pipe.unet.load_attn_procs(lora_path)
            if not model_name == 'pai-diffusion-artist-large-zh':
                warning_inf_lora = 'The current lora model: ' + str(
                    lora) + '. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d ' \
                            ' are based on the pai-diffusion-artist-large-zh, choose other models may cause unexpected outputs.'
            else:
                warning_inf_lora = 'The current lora model: ' + str(lora) + '.'
            warning_inf = warning_inf + ' ' + warning_inf_lora
        return warning_inf

    if tag == 'inpainting':
        pipe, warning_inf = select_inpainting_pipe(model_name)
        return warning_inf

    if tag == 'controlnet':
        pipe, warning_inf = select_controlnet_pipe(model_name, control_model)
        return warning_inf


def refresh_model_controlnet(model_name, tag, control_model, lora, whether_lora):
    if not model_name:
        return 'Select the model you need.', gr.Slider.update(), gr.CheckboxGroup.update()
    if model_name == 'None':
        return 'The current control_model is' + str(
            control_model) + '. Please select the model you need.', gr.Slider.update(), gr.CheckboxGroup.update()

    if model_name == 'pai-diffusion-anime-large-zh':
        if not control_model:
            return 'The current selected base chinses model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update(), gr.CheckboxGroup.update()
        elif control_model == 'None':
            return 'The current selected base chinses model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update(), gr.CheckboxGroup.update()
        elif control_model == 'pai-diffusion-artist-large-zh-controlnet-canny':
            return refresh_checkpoints(model_name, tag, control_model, lora, whether_lora), gr.Slider.update(
                value=768), gr.CheckboxGroup.update(value='canny')
        elif control_model == 'pai-diffusion-artist-large-zh-controlnet-depth':
            return refresh_checkpoints(model_name, tag, control_model, lora, whether_lora), gr.Slider.update(
                value=768), gr.CheckboxGroup.update(value='depth')
        else:
            return refresh_checkpoints(model_name, tag, control_model, lora,
                                       whether_lora), gr.Slider.update(), gr.CheckboxGroup.update()
    else:
        if not control_model:
            return 'The current selected base chinses model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update(), gr.CheckboxGroup.update()
        elif control_model == 'None':
            return 'The current selected base chinses model is ' + str(
                model_name) + '. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.', gr.Slider.update(), gr.CheckboxGroup.update()
        elif control_model == 'pai-diffusion-artist-large-zh-controlnet-canny':
            return refresh_checkpoints(model_name, tag, control_model, lora,
                                       whether_lora), gr.Slider.update(), gr.CheckboxGroup.update(value='canny')
        elif control_model == 'pai-diffusion-artist-large-zh-controlnet-depth':
            return refresh_checkpoints(model_name, tag, control_model, lora,
                                       whether_lora), gr.Slider.update(), gr.CheckboxGroup.update(value='depth')
        else:
            return refresh_checkpoints(model_name, tag, control_model, lora,
                                       whether_lora), gr.Slider.update(), gr.CheckboxGroup.update()


def refresh_lora_model(model_name, tag, control_model, lora, whether_lora):
    return refresh_checkpoints_lora(model_name, tag, control_model, lora, whether_lora)


def refresh_checkpoints_lora(model_name, tag, control_model, lora, whether_lora):
    global pipe

    if not model_name:  # in case of only select lora, no based model is selected.
        warning_inf = 'The current lora model: ' + str(
            lora) + '. Now, select the chinsese diffusion model you need. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                    ' choose other models may cause an unexpected error.'
    elif model_name == 'None':
        return 'The current lora model: ' + str(
            lora) + '. Now, select the chinsede diffusion model you need. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                    ' choose other models may cause an unexpected error.'
    elif lora == 'None' and model_name:  # in case of from using lora to free lora
        warning_inf = refresh_checkpoints(model_name, tag, control_model, lora, whether_lora)
    else:  # in case of select model and further to use lora
        lora_path = os.path.join(lora_model_dir, lora)

        if tag == 'txt2img':
            model_dir = os.path.join(stable_diffusion_dir, model_name)
            if os.path.exists(model_dir):
                pipe.unet.load_attn_procs(lora_path)
                if model_name == 'pai-diffusion-artist-large-zh':
                    warning_inf = 'The current lora model: ' + str(lora) + '. Enjoy it!'
                else:
                    warning_inf = 'The current lora model: ' + str(
                        lora) + '. Enjoy it! Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                                ' choose other models may cause an unexpected error.'
            else:
                warning_inf = 'Can not finding the model: ' + model_name + ' in path of: ' + model_dir + '. Please check it and download your model.'

        if tag == 'img2img':
            model_dir = os.path.join(stable_diffusion_dir, model_name)
            if os.path.exists(model_dir):
                pipe.unet.load_attn_procs(lora_path)
                warning_inf = 'The current lora model: ' + str(lora) + '. Enjoy it!'
            else:
                warning_inf = 'Can not finding the model: ' + model_name + ' in path of: ' + model_dir + '. Please check it and download your model.'
    return warning_inf


# ----------------------------------------------------------- infer ----------------------------------------------------
def infer_text2img(model_name, prompt, negative_prompt, height, width, guide, steps, num_images, seed, scheduler,
                   use_Lora):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)
    scheduler = scheduler
    if scheduler == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    torch.manual_seed(seed)
    sample_images = num_images

    # Forward embeddings and negative embeddings through text encoder
    if len(prompt) >= 33:
        max_length = pipe.tokenizer.model_max_length

        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        # print('input_ids: {}'.format(input_ids))
        if len(prompt) < len(negative_prompt):
            negative_prompt = negative_prompt[0:len(prompt)]
        negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                      max_length=input_ids.shape[-1], return_tensors="pt").input_ids
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        if use_Lora:
            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, height=height,
                         width=width, guidance_scale=guide, num_images_per_prompt=sample_images,
                         num_inference_steps=steps, cross_attention_kwargs={"scale": 0.4}).images
        else:
            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, height=height,
                         width=width, guidance_scale=guide, num_images_per_prompt=sample_images,
                         num_inference_steps=steps).images
    else:
        if use_Lora:
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                         guidance_scale=guide, num_images_per_prompt=sample_images, num_inference_steps=steps,
                         cross_attention_kwargs={"scale": 0.4}).images
        else:
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                         guidance_scale=guide, num_images_per_prompt=sample_images, num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])

    return image_out


def infer_img2img(model_name, prompt, image_in, height, width, num_images, guide, steps, strength, seed, use_Lora):
    image = Image.fromarray(image_in)
    image_in = image.convert("RGB")
    w, h = map(lambda x: x - x % 32, (height, width))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image_in = torch.from_numpy(image)
    image_in = 2. * image_in - 1.

    torch.manual_seed(seed)
    sample_images = num_images

    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained("model_name").to("cuda")
    if use_Lora:
        image = pipe(prompt=prompt, image=image_in, strength=strength, guidance_scale=guide,
                     num_images_per_prompt=sample_images, num_inference_steps=steps,
                     cross_attention_kwargs={"scale": 0.4}).images
    else:
        image = pipe(prompt=prompt, image=image_in, strength=strength, guidance_scale=guide,
                     num_images_per_prompt=sample_images, num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])
    return image_out


def infer_inpainting(model_name, prompt, negative_prompt, image_in, mask_in, height, width, strength, num_images, guide,
                     steps, scheduler, seed):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)

    scheduler = scheduler
    if scheduler == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    image = Image.fromarray(image_in)
    mask = Image.fromarray(mask_in)

    image_in = image.convert("RGB").resize((width, height))
    mask_in = mask.convert('L').resize((width, height), resample=Image.Resampling.NEAREST)
    torch.manual_seed(seed)
    sample_images = num_images

    image = pipe(prompt=prompt, image=image_in, mask_image=mask_in, strength=strength, guidance_scale=guide,
                 num_images_per_prompt=sample_images, num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])
    return image_out


def infer_controlnet(control_mode, prompt, negative_prompt, image_in, height, width, guide, steps, num_images, seed,
                     scheduler):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)
    scheduler = scheduler
    if scheduler == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print(f'control mode: {control_mode[0]}')
    if control_mode[0] == 'canny':
        image_in = to_canny(image_in)
    elif control_mode[0] == 'depth':
        image_in = to_depth(image_in)
    else:
        image_in = to_canny(image_in)
    print('control mode trans Done')

    torch.manual_seed(seed)
    sample_images = num_images

    # Forward embeddings and negative embeddings through text encoder
    if len(prompt) >= 33:
        max_length = pipe.tokenizer.model_max_length

        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        # print('input_ids: {}'.format(input_ids))
        if len(prompt) < len(negative_prompt):
            negative_prompt = negative_prompt[0:len(prompt)]
        negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                      max_length=input_ids.shape[-1], return_tensors="pt").input_ids
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        image = pipe(prompt_embeds=prompt_embeds, image=image_in, negative_prompt_embeds=negative_prompt_embeds,
                     height=height, width=width, guidance_scale=guide, num_images_per_prompt=sample_images,
                     num_inference_steps=steps).images
    else:
        print('before inference')
        image = pipe(prompt=prompt, image=image_in, negative_prompt=negative_prompt, height=height, width=width,
                     guidance_scale=guide, num_images_per_prompt=sample_images, num_inference_steps=steps).images
    print('after inference')

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])

    return image_out
