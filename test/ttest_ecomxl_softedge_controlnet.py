from diffusers.utils import load_image
from ecom.controlnet.controlnet_hf import ControlnetDiffusers

image = load_image(
    "https://huggingface.co/alimama-creative/EcomXL_controlnet_softedge/resolve/main/images/1_1.png"
)
prompt = "a bottle on the Twilight Grassland, Sitting on the ground, a couple of tall grass sitting in a field of tall grass, sunset,"
negative_prompt = "low quality, bad quality, sketches"

output = ControlnetDiffusers(base_model_id="", task_name="ecomxl_softedge").process_ecomxl_softedge(
    image,
    prompt,
    negative_prompt,
    num_images=1,
    num_steps=25,
    guidance_scale=7,
    controlnet_conditioning_scale=0.6,
    width=1024,
    height=1024
)

output.save("test_softedge.png")
