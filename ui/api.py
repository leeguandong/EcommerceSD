import time
import asyncio
from ecom import engineer_log as eng
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# from aidraw.ardraw import run_aidraw_api

# 起 fastapi
app = FastAPI(description="FastAPI for txt2img_adetailer_controlnet Application", version="1.1.0")
# 创建一个异步锁,阻塞处理并发的异步请求。
lock = asyncio.Lock()


class Item(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    atmosphere: Optional[str] = ""
    location: Optional[str] = ""

    image_nums: Optional[int] = 1
    size: Optional[int] = 1
    upload_image: Optional[dict] = {}
    sketch_image: Optional[dict] = {}
    reference_image: Optional[dict] = {}
    style: Optional[str] = ""


@app.post("/ecommsd")
async def apply(input_json: Item):
    async with lock:
        try:
            eng.log.info(meg="=" * 20 + "Service started running" + "=" * 20)
            input_json = dict(input_json)
            results = {
                "ImgUrls": [],
                "code": "10100",
                "message": "算法返回成功",
                "task_id": input_json.get('task_id', "")
            }
            time_total = time.time()
            # img_urls = run_aidraw_api(input_json)
            img_urls = None
            results['ImgUrls'] = img_urls
            eng.log.info(f"##-------------Total elapsed time:{time.time() - time_total}s-------------##")
        except Exception as ee:
            eng.log.error(f"error:{ee}")
            try:
                msg = eval(str(ee))
                results['code'] = msg['code']
                results['message'] = msg['message']
            except:
                results["code"] = 10301
                results['message'] = "算法内部错误"
            eng.log.error(str(results))
        return results
