from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import requests
from io import BytesIO
import torch
import torch_musa
from PIL import Image
import uvicorn
from qwen_vl_utils import process_vision_info

app = FastAPI()

# 加载模型
model_name = "/models/Qwen2-VL-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="musa"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

@app.post("/emotion_analysis")
async def emotion_analysis(url: str):
    response = requests.get(url)
    # 判断是否是图片
    if response.headers.get("Content-Type").startswith("image/"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": url,
                    },
                    {"type": "text",
                     "text": "请你看看这只宠物的情绪怎么样，主要从眼睛、嘴巴、耳朵、胡须的状态，瞳孔的大小以及宠物的姿势和毛发质量来判断，"
                             "给出喜怒哀乐四个维度的得分（百分制度），以json的格式返回眼睛、嘴巴、耳朵、胡须的状态，"
                             "瞳孔的大小以及宠物的姿势和毛发质量这些属性的字符串，以及喜怒哀乐四个维度的得分"},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("musa")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)