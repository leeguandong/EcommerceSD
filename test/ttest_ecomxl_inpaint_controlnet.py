from diffusers.utils import load_image
from ecom.controlnet.controlnet_hf import ControlnetDiffusers

image = load_image(
    "https://huggingface.co/alimama-creative/EcomXL_controlnet_inpaint/resolve/main/images/inp_0.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/EcomXL_controlnet_inpaint/resolve/main/images/inp_1.png"
)

prompt = "a product on the table"
negative_prompt = "low quality, bad quality, sketches"

output = ControlnetDiffusers(base_model_id="", task_name="ecomxl_softedge").process_ecomxl_inpaint(
    image,
    prompt,
    negative_prompt,
    num_images=1,
    num_steps=25,
    guidance_scale=7,
    controlnet_conditioning_scale=0.5,
    width=1024,
    height=1024
)

output.save("test_inpaint.png")
