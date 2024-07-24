from pathlib import Path
from loguru import logger
from hydit.config import get_args
from hydit.inference import End2End


def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    return args, gen


if __name__ == "__main__":
    args, gen = inferencer()

    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size
    results = gen.predict(args.prompt,
                          height=height,
                          width=width,
                          seed=args.seed,
                          enhanced_prompt=None,
                          negative_prompt=args.negative,
                          infer_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          batch_size=args.batch_size,
                          src_size_cond=args.size_cond,
                          )
    images = results['images']

    # Save images
    save_dir = Path("/home/image_team/image_team_docker_home/lgd/EcommerceSD/results/hunyuandit")
    save_dir.mkdir(exist_ok=True)
    # Find the first available index
    all_files = list(save_dir.glob('*.png'))
    if all_files:
        start = max([int(f.stem) for f in all_files]) + 1
    else:
        start = 0

    for idx, pil_img in enumerate(images):
        save_path = save_dir / f"{idx + start}.png"
        pil_img.save(save_path)
        logger.info(f"Save to {save_path}")

# ---------------------------------------------------T2I------------------------------------------------
# Prompt Enhancement + Text-to-Image. Torch mode
# python hunyuandit_t2i.py --prompt "渔舟唱晚"

# Only Text-to-Image. Torch mode
# python hunyuandit_t2i.py --prompt "渔舟唱晚"

# Only Text-to-Image. Flash Attention mode
# python hunyuandit_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# Generate an image with other image sizes.
# python hunyuandit_t2i.py --prompt "渔舟唱晚" --image-size 1280 768

# Prompt Enhancement + Text-to-Image. DialogGen loads with 4-bit quantization, but it may loss performance.
# python hunyuandit_t2i.py --prompt "渔舟唱晚"  --load-4bit

# --------------------------------------------------- LORA ---------------------
# python hunyuandit_t2i.py --prompt "青花瓷风格，一只猫在追蝴蝶"  --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
# python hunyuandit_t2i.py --prompt "青花瓷风格，一只小狗"   --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt
