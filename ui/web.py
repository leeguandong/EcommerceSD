import gradio as gr
from ui.utils.list_models import list_available_models, list_lora_models, list_available_controlnet, \
    available_models, available_lora, available_controlnet
from ui.utils.refresh import refresh_model, refresh_checkpoints, refresh_lora_model, refresh_model_controlnet, \
    infer_text2img, infer_img2img, infer_controlnet, infer_inpainting


def get_sample_method():
    return ['Euler a', 'Euler', 'Heun', 'DPM', 'DDIM', 'DDPM', 'PNDM']


def use_lora(whether_lora, model_name, tag, contorl_mode, lora_model):
    if whether_lora:
        warning_inf = ''
        return gr.Dropdown.update(visible=True), warning_inf
    else:
        lora_model = 'None'
        warning_inf = refresh_model(model_name, tag, contorl_mode, lora_model, whether_lora)
        return gr.Dropdown.update(value='None', visible=False), warning_inf


def txt2img_ui():
    list_available_models()
    list_lora_models()
    with gr.Row():
        with gr.Column(scale=1, ):
            prompt = gr.Textbox(label='提示词(prompt)')
            negative_prompt = gr.Textbox(label='负面词(negative_prompt)')
            tag = gr.State('txt2img')
            with gr.Column(scale=0.5):
                scheduler = gr.Dropdown(get_sample_method(),
                                        value='DPM',
                                        label='scheduler')
                height = gr.Slider(256, 768, value=512, step=64, label='高度(height)')
                width = gr.Slider(256, 768, value=512, step=64, label='宽度(width)')
                guide = gr.Slider(2, 15, value=5, step=0.1, label='文本引导强度(guidance scale)')
                steps = gr.Slider(10, 50, value=20, step=1, label='迭代次数(inference steps)')
                seed = gr.Slider(0, 2147483647, default=-1, step=1, label='Seed')
                num_images = gr.Slider(0, 8, value=2, step=1, label='图像数量(images)')

        with gr.Column(scale=1, ):
            submit_btn = gr.Button("生成图像(Generate)").style(full_width=False)
            model_name = gr.Dropdown(label="模型 (Model)", elem_id="Chinese_model", choices=available_models)
            whether_lora = gr.Checkbox(label='Lora, 点击选择lora模型', info="Do you want to use lora")
            lora_model = gr.Dropdown(label="lora模型 (lora Model)", choices=available_lora, visible=False)
            warning_box = gr.HTML("<p>&nbsp")
            # select_model = gr.Button("加载模型").style(full_width=False)
            image_out = gr.Gallery(label="输出(output)",
                                   show_label=False,
                                   elem_id="gallery").style(grid=[2], height="auto")

        model_name.change(fn=refresh_model,
                          inputs=[model_name, tag, tag, lora_model, whether_lora],
                          outputs=[warning_box, height])
        whether_lora.change(fn=use_lora,
                            inputs=[whether_lora, model_name, tag, tag, lora_model],
                            outputs=[lora_model, warning_box])
        lora_model.change(fn=refresh_lora_model,
                          inputs=[model_name, tag, tag, lora_model, whether_lora],
                          outputs=warning_box)
        submit_btn.click(fn=infer_text2img,
                         inputs=[model_name, prompt, negative_prompt, height, width, guide, steps, num_images, seed,
                                 scheduler, whether_lora], outputs=image_out)


def img2img_ui():
    list_available_models()
    list_lora_models()

    with gr.Row():
        with gr.Column(scale=1, ):
            prompt = gr.Textbox(label='提示词(prompt)')
            image_in = gr.Image(label='输入图像(image)')
            tag = gr.State('img2img')
            with gr.Column(scale=0.5):
                height = gr.Slider(256, 768, value=512, step=64, label='高度(height)')
                width = gr.Slider(256, 768, value=512, step=64, label='宽度(width)')
                num_images = gr.Slider(0, 8, value=2, step=1, label='图像数量(images)')
                guide = gr.Slider(2, 15, value=5, step=0.1, label='文本引导强度(guidance scale)')
                steps = gr.Slider(10, 50, value=20, step=1, label='迭代次数(inference steps)')
                strength = gr.Slider(0.05, 1.0, value=0.8, label='强度(strength)')
                seed = gr.Slider(0, 2147483647, default=-1, step=1, label='Seed')

        with gr.Column(scale=1, ):
            submit_btn = gr.Button("生成图像(Generate)").style(full_width=False)
            model_name = gr.Dropdown(label="模型 (Model)", elem_id="Chinese_model", choices=available_models)
            whether_lora = gr.Checkbox(label='使用Lora', info="Do you want to use lora")
            lora_model = gr.Dropdown(label="lora模型 (lora Model)", choices=available_lora, visible=False)
            warning_box = gr.HTML("<p>&nbsp")
            image_out = gr.Gallery(label="输出(output)", show_label=False, elem_id="gallery"). \
                style(grid=[2], height="auto")
        model_name.change(fn=refresh_model, inputs=[model_name, tag, tag, lora_model, whether_lora],
                          outputs=[warning_box, height])
        whether_lora.change(fn=use_lora, inputs=[whether_lora, model_name, tag, tag, lora_model],
                            outputs=[lora_model, warning_box])
        lora_model.change(fn=refresh_lora_model, inputs=[model_name, tag, tag, lora_model, whether_lora],
                          outputs=warning_box)
        submit_btn.click(fn=infer_img2img,
                         inputs=[model_name, prompt, image_in, height, width, num_images, guide, steps, strength, seed,
                                 whether_lora], outputs=image_out)


def controlnet_ui():
    list_available_models()
    list_available_controlnet()

    with gr.Row():
        with gr.Column(scale=1, ):
            tag = gr.State('controlnet')
            prompt = gr.Textbox(label='提示词(prompt)')
            negative_prompt = gr.Textbox(label='负面词(negative_prompt)')
            image_in = gr.Image(label='输入图像(image)')
            with gr.Column(scale=0.5):
                scheduler = gr.Dropdown(get_sample_method(), value='Euler a', label='scheduler')
                height = gr.Slider(256, 768, value=512, step=64, label='高度(height)')
                width = gr.Slider(256, 768, value=512, step=64, label='宽度(width)')
                guide = gr.Slider(2, 15, value=5, step=0.1, label='文本引导强度(guidance scale)')
                steps = gr.Slider(10, 50, value=20, step=1, label='迭代次数(inference steps)')
                seed = gr.Slider(0, 2147483647, default=-1, step=1, label='Seed')
                num_images = gr.Slider(0, 8, value=2, step=1, label='图像数量(images)')

        with gr.Column(scale=1, ):
            submit_btn = gr.Button("生成图像(Generate)").style(full_width=False)
            model_name = gr.Dropdown(label="模型 (Model)", elem_id="Chinese_model", choices=available_models)
            control_model = gr.Dropdown(choices=available_controlnet, label='Controlnet model',
                                        info='Select the control model you need.')
            control_mode = gr.CheckboxGroup(value='canny', choices=['canny', 'depth'], label='Control mode',
                                            info='Select the control mode you need. Only one value to be choose.')
            warning_box = gr.HTML("<p>&nbsp")
            image_out = gr.Gallery(label="输出(output)", show_label=False, elem_id="gallery").style(grid=[2],
                                                                                                    height="auto")
        model_name.change(fn=refresh_model, inputs=[model_name, tag, control_model, tag, tag],
                          outputs=[warning_box, height])
        control_model.change(fn=refresh_model_controlnet, inputs=[model_name, tag, control_model, tag, tag],
                             outputs=[warning_box, height, control_mode])
        # control_mode.change(fn= refresh_model, inputs = [control_model,tag,control_mode,tag,tag], outputs = warning_box)
        submit_btn.click(fn=infer_controlnet,
                         inputs=[control_mode, prompt, negative_prompt, image_in, height, width, guide, steps,
                                 num_images, seed, scheduler], outputs=image_out)


def inpainting_ui():
    list_available_models()
    # list_lora_models()

    with gr.Row():
        with gr.Column(scale=1, ):
            prompt = gr.Textbox(label='提示词(prompt)')
            negative_prompt = gr.Textbox(label='负面词(negative_prompt)')
            image_in = gr.Image(label='输入图像(image)')
            mask_in = gr.Image(label='输入掩膜(mask)')

            tag = gr.State('inpainting')
            with gr.Column(scale=0.5):
                scheduler = gr.Dropdown(get_sample_method(), value='DPM', label='scheduler')
                height = gr.Slider(256, 768, value=512, step=64, label='高度(height)')
                width = gr.Slider(256, 768, value=512, step=64, label='宽度(width)')
                strength = gr.Slider(0.05, 1.0, value=0.8, label='强度(strength)')
                num_images = gr.Slider(0, 8, value=2, step=1, label='图像数量(images)')
                guide = gr.Slider(2, 15, value=5, step=0.1, label='文本引导强度(guidance scale)')
                steps = gr.Slider(10, 50, value=20, step=1, label='迭代次数(inference steps)')
                seed = gr.Slider(0, 2147483647, default=-1, step=1, label='Seed')

        with gr.Column(scale=1, ):
            submit_btn = gr.Button("生成图像(Generate)").style(full_width=False)
            model_name = gr.Dropdown(label="模型 (Model)", elem_id="Chinese_model", choices=available_models)
            warning_box = gr.HTML("<p>&nbsp")
            image_out = gr.Gallery(label="输出(output)", show_label=False, elem_id="gallery").style(grid=[2],
                                                                                                    height="auto")

        model_name.change(fn=refresh_model, inputs=[model_name, tag, tag, tag, tag], outputs=[warning_box, height])
        submit_btn.click(fn=infer_inpainting,
                         inputs=[model_name, prompt, negative_prompt, image_in, mask_in, height, width, strength,
                                 num_images, guide, steps, scheduler, seed], outputs=image_out)


def ecommercesd_ui():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Txt2img"):
            txt2img_ui()
        with gr.Tab("Img2img"):
            img2img_ui()
        with gr.Tab("Inpainting"):
            inpainting_ui()
        with gr.Tab("Controlnet"):
            controlnet_ui()
    return ui



