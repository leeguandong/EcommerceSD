from ecom.dit.dit_hf import DiTDiffusers

pipeline = DiTDiffusers(base_model_id="").load_hunyuandit_lora(lora_model_id="")

prompt = "玉石绘画风格，一只猫在追蝴蝶"
negative_prompt = ""

images = pipeline.run_pipe(prompt,
                           negative_prompt,
                           num_images=4,
                           num_steps=100,
                           guidance_scale=6.0,
                           seed=42)[0]
images.save("hunyuandit_lora.png")


