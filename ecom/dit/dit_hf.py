import torch
from safetensors import safe_open
from diffusers import HunyuanDiTPipeline
from ecom.dit.utils import load_hunyuan_dit_lora


class DiTDiffusers:
    def __init__(self, base_model_id: str = ""):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.pipe = self.load_pipe(base_model_id)

    def load_pipe(self, model_id):
        if "hunyundit" in model_id.lower():
            pipe = HunyuanDiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            try:
                pipe.transformer.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)

                pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune",
                                                 fullgraph=True)
                pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
            except:
                print("no torch compile faster!!!")
        else:
            raise ValueError("just support hunyuandit")

        pipe.to(self.device)
        return pipe

    def load_hunyuandit_lora(self, lora_model_id):
        lora_state_dict = {}
        with safe_open("./ckpts/t2i/lora/jade/adapter_model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                lora_state_dict[k[17:]] = f.get_tensor(k)  # remove 'basemodel.model'

        transformer_state_dict = self.pipe.transformer.state_dict()
        transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=1.0)
        self.pipe.transformer.load_state_dict(transformer_state_dict)

    def run_pipe(self,
                     prompt,
                     negative_prompt,
                     num_images,
                     num_steps,
                     guidance_scale,
                     seed):
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator).images
