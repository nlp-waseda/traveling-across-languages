import os

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    print("Qwen2.5 is not installed.")

try:
    from transformers import MllamaForConditionalGeneration
except Exception:
    print("LLama is not installed.")

try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    print("Gemini is not installed.")

try:
    from cambrian.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from cambrian.mm_utils import get_model_name_from_path as cambrian_get_model_name_from_path
    from cambrian.model.builder import load_pretrained_model as cambrian_load_pretrained_model
except Exception:
    print("Cambrian is not installed.")

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.model.builder import get_model_name_from_path, load_pretrained_model
except Exception:
    print("LLava is not installed.")


def init_llava_ov():
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    tokenizer, model, _, _ = load_pretrained_model(pretrained, None, model_name)

    return tokenizer, model

def init_llava_1_5():
    model_path = "liuhaotian/llava-v1.5-7b"
    tokenizer, model, _, _ = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

    return tokenizer, model

def init_pangea():
    args = {"multimodal": True}
    model_path = "neulab/Pangea-7B"
    model_name = "Pangea-7B-qwen"
    tokenizer, model, _, _ = load_pretrained_model(model_path, None, model_name, **args)

    return tokenizer, model

def init_llama_3_2():
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return processor, model

def init_internvl():
    path = 'OpenGVLab/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return tokenizer, model

def init_qwen2_5vl():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return processor, model

def init_cambrian(model_name_or_path: str):
    model_name = cambrian_get_model_name_from_path(model_name_or_path)
    tokenizer, model, image_processor, max_length = cambrian_load_pretrained_model(model_name_or_path, None, model_name)
    return tokenizer, model

def init_qwen2():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    return tokenizer, model

def init_qwen2_5():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    return tokenizer, model

def init_llama_3_0():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model

def init_internlm():
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", torch_dtype=torch.float16, trust_remote_code=True).cuda()
    return tokenizer, model

def init_gemma_2():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model

def init_gemini(args):
    model = genai.GenerativeModel(model_name=args.model_name_or_path)
    return None, model

def init_gpt4o():
    return None, "gpt-4o-2024-11-20"
