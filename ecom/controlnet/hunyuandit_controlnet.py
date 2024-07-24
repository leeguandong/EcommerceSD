from pathlib import Path
from loguru import logger
from ecom.dit.hydit.config import get_args
from ecom.dit.hydit.inference_controlnet import End2End

from torchvision import transforms as T

norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

from PIL import Image


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

    condition = Image.open(args.condition_image_path).convert('RGB').resize((height, width))
    image = norm_transform(condition)
    image = image.unsqueeze(0).cuda()

    results = gen.predict(args.prompt,
                          height=height,
                          width=width,
                          image=image,
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
    save_dir = Path('results')
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

# python3 hunyuandit_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
