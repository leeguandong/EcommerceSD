from diffusers.utils import load_image
from ecom.controlnet.controlnet_hf import ControlnetDiffusers

pipeline = ControlnetDiffusers(base_model_id="", task_name="canny")

cond_image = load_image(
    'https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt = "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"

pipeline.process_hunyuandit_canny(cond_image,
                                  prompt,
                                  negative_prompt="",
                                  num_images=4,
                                  num_steps=50,
                                  guidance_scale=1,
                                  width=1024,
                                  height=1024,
                                  low_threshold=100,
                                  high_threshold=200)

cond_image = load_image(
    'https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth/resolve/main/depth.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt = "在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足"

pipeline.process_hunyuandit_depth(cond_image,
                                  prompt,
                                  negative_prompt="",
                                  num_images=4,
                                  num_steps=50,
                                  guidance_scale=1,
                                  width=1024,
                                  height=1024, )

cond_image = load_image(
    'https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose/resolve/main/pose.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt = "一位亚洲女性，身穿绿色上衣，戴着紫色头巾和紫色围巾，站在黑板前。背景是黑板。照片采用近景、平视和居中构图的方式呈现真实摄影风格"

pipeline.process_hunyuandit_pose(cond_image,
                                 prompt,
                                 num_images=4,
                                 num_steps=50,
                                 guidance_scale=1,
                                 width=1024,
                                 height=1024, )
