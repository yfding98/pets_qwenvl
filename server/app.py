import time

from fastapi import FastAPI
from modelscope import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import requests
import torch
import torch_musa
import uvicorn
from qwen_vl_utils import process_vision_info

app = FastAPI()

# 加载模型
model_name = "/models/Qwen2.5-VL-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 加载模型并设定FP16精度
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="musa", trust_remote_code=True, fp16=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)


@app.post("/emotion_analysis")
async def emotion_analysis(url: str):
    response = requests.get(url)
    # 判断是否是图片
    if response.headers.get("Content-Type").startswith("image/"):
        torch.musa.empty_cache()
        start_time = time.time()
        query = tokenizer.from_list_format([
            {'image': url},
            {'text': "请你看看这只宠物的情绪怎么样，主要从眼睛、嘴巴、耳朵、胡须的状态，瞳孔的大小以及宠物的姿势和毛发质量来判断，"
                             "给出喜怒哀乐四个维度的得分（百分制度），以json的格式返回眼睛、嘴巴、耳朵、胡须的状态，"
                             "瞳孔的大小以及宠物的姿势和毛发质量这些属性的字符串，以及喜怒哀乐四个维度的得分"},
        ])

        response, history = model.chat(tokenizer, query=query, history=None)
        total_tokens = len(tokenizer.encode(response))
        inference_time = time.time() - start_time

        print(f"模型响应: {response}")
        print(f"推理时间: {inference_time:.2f} 秒")
        print(f"总生成 token 数: {total_tokens}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)