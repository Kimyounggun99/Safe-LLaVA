import json
import os
import uuid
from PIL import Image
from vllm import LLM
import argparse
from transformers import (
    
    AutoTokenizer, AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    Gemma3ForConditionalGeneration, 
    LlavaOnevisionForConditionalGeneration,   
    Qwen2_5_VLForConditionalGeneration,
    AutoModel
)
import gc
import torch
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from llava.model.builder import load_pretrained_model
from llava.mm_utils import  process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import copy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, help="refusal_hard/refusal_soft/implicit_leakage")
    return parser.parse_args()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

  
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)


    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]


    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



args = get_args()

# 모델 ID 매핑
if args.model == 'LLaVA':
    model = "llava-hf/llava-1.5-7b-hf"
elif args.model == "LLaVA-Next":
    model = "llava-hf/llava-v1.6-vicuna-7b-hf"
elif args.model == "LLaVA-Onevision":
    model = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
elif args.model == "LLaVA-Onevision-0_5B":
    model = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
elif args.model == 'Gemma':
    model = "google/gemma-3-4b-it"
elif args.model == "Qwen25":
    model = "Qwen/Qwen2.5-VL-7B-Instruct"
elif args.model == "InternVLC2_5":
    model = 'OpenGVLab/InternVL2_5-8B'
elif args.model == "InternVLC3":
    model = 'OpenGVLab/InternVL3-8B'
elif args.model =="Safe-LLaVA-0_5B":
    model = "./checkpoints/Safe-LLaVA-0_5B"

else:
    raise ValueError("Unsupported model")

# 모델 로딩
if args.model =="LLaVA":
    llm = LlavaForConditionalGeneration.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model)
elif args.model == "LLaVA-Next":
    processor = LlavaNextProcessor.from_pretrained(model)
    llm = LlavaNextForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    llm.to("cuda")
elif args.model == "LLaVA-Onevision" or args.model == "LLaVA-Onevision-0_5B":
    llm = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model)

elif args.model == "Gemma":
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model)
    llm = Gemma3ForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch.bfloat16
    ).to("cuda").eval()
elif args.model=="Qwen25":
    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, torch_dtype=torch.float16, low_cpu_mem_usage=True,  
    ).cuda()
    processor = AutoProcessor.from_pretrained(model)
elif args.model=="InternVLC3" or args.model=="InternVLC2_5":
   
    llm = AutoModel.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=128, do_sample=True)
    
elif args.model =="Safe-LLaVA-0_5B":
    model_name = "llava_qwen"
    conv_template = "qwen_1_5"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, llm, processor, context_len = load_pretrained_model(
        model_path=model,
        model_base=None,
        model_name=model_name,
        attn_implementation="eager",
    )


else:
    llm = LLM(model=model, trust_remote_code=True)

# 경로 설정
input_path = f"PRISM_{args.task}.jsonl"
image_base_dir = "biometric_images"
output_path = f"./answers/{args.model}/{args.task}_Answer.jsonl"
model_id = args.model

# 결과 리스트
outputs = []

with open(input_path, "r") as f:
    lines= f.readlines()
    for line in tqdm(lines, desc=f"Evaluating {args.model} on {args.task}"):
        item = json.loads(line)
        image_path = os.path.join(image_base_dir, item["image"])
        prompt = item["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error to load images: {image_path}, error: {e}")
            continue

        # 모델 예측
        if args.model=="LLaVA" or args.model=="LLaVA-Next" or args.model =="LLaVA-Onevision" or args.model == "LLaVA-Onevision-0_5B":
            conversation = [
                {
                
                  "role": "user",
                  "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image"},
                    ],
                },
            ]
            chat = processor.apply_chat_template(conversation, add_generation_prompt=True) 
            inputs = processor(images=image, text=chat, return_tensors='pt').to("cuda", torch.float16)
            output = llm.generate(**inputs, max_new_tokens=200, do_sample=False)
            answer= processor.decode(output[0][2:], skip_special_tokens=True)

        elif args.model == "Gemma":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
            ).to("cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                output = llm.generate(**inputs, max_new_tokens=100)
                answer = processor.decode(output[0], skip_special_tokens=True)
        elif args.model == "Qwen25":

            #image = image.resize((420, 280), resample=Image.BICUBIC)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
 
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            image_inputs= image_inputs[0].resize((420, 280), resample=Image.BICUBIC)
            inputs = processor(
                text=[text],
                images=image_inputs,     
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to("cuda")
        
            generated_ids = llm.generate(**inputs, max_new_tokens=200)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        elif args.model=="InternVLC3" or args.model=="InternVLC2_5":
            question = f'<image>\n{prompt}'
       
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            answer = llm.chat(tokenizer, pixel_values, question, generation_config)
       
        elif args.model=='Safe-LLaVA-0_5B':
            
            image_tensor = process_images([image], processor, llm.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            question = f'<image>\n{prompt}'
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size]


            cont = llm.generate(input_ids, images=image_tensor,image_sizes=image_sizes,max_new_tokens=100,)
            
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

            answer= text_outputs[0]

        print(f"Raw Answer: {answer}")
        
        if prompt in answer:
            answer = answer.split(prompt, 1)[-1].strip()
            if args.model == "Gemma":
                answer = answer.split("model\n", 1)[-1].strip()
            elif args.model =="LLaVA" or args.model =="LLaVA-Next" :
                answer = answer.split("ASSISTANT:", 1)[-1].strip()
            elif args.model =="LLaVA-Onevision" or args.model == "LLaVA-Onevision-0_5B":
                answer = answer.split("assistant\n", 1)[-1].strip()
  
        else:
            if args.model =="Qwen25":
                answer = answer[0]
            elif args.model == "Safe-LLaVA-0_5B":
                answer= answer.split("Assistant:", 1)[-1].strip().split("\n",1)[0].strip()

        outputs.append({
            "question_id": item["question_id"],
            "prompt": prompt,
            "text": answer,
            "answer_id": uuid.uuid4().hex[:22],
            "model_id": model_id,
            "metadata": {}
        })


        

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for out in outputs:
        f.write(json.dumps(out) + "\n")