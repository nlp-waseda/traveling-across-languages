import asyncio
import copy
import os
import re

import torch
import transformers
from openai import AsyncOpenAI

try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception:
    print("OpenAI is not installed.")

try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    print("Gemini is not installed.")

try:
    from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from cambrian.conversation import conv_templates
    from cambrian.mm_utils import (
        get_model_name_from_path as cambrian_get_model_name_from_path,
    )
    from cambrian.mm_utils import (
        tokenizer_image_token as cambrian_tokenizer_image_token,
    )
except Exception:
    print("Cambrian is not installed.")

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
except Exception:
    print("LLava is not installed.")

semaphore = asyncio.Semaphore(25)


def generate_llava_ov(args, tokenizer, model, prompt) -> str:
    device = "cuda"
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=512,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    return text_outputs[0]

def generate_llava_1_5(args, tokenizer, model, prompt) -> str:
    device = "cuda"
    conv_template = "llava_v1"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=512,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    return text_outputs[0]

def generate_pangea(args, tokenizer, model, prompt) -> str:
    def _preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> dict:
        roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        im_start, im_end = tokenizer.additional_special_tokens_ids
        nl_tokens = tokenizer("\n").input_ids
        _system = tokenizer("system").input_ids + nl_tokens
        input_ids = []
        source = sources
        if roles[source[0]["from"]] != roles["human"]: source = source[1:]
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
                num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                texts = sentence["value"].split("<image>")
                _input_id = tokenizer(role).input_ids + nl_tokens 
                for i,text in enumerate(texts):
                    _input_id += tokenizer(text).input_ids 
                    if i<len(texts)-1: _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
                _input_id += [im_end] + nl_tokens
                assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
            else:
                if sentence["value"] is None: _input_id = tokenizer(role).input_ids + nl_tokens
                else: _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
        input_ids.append(input_id)
        return torch.tensor(input_ids, dtype=torch.long)

    input_ids = _preprocess_qwen([{"from": "human", "value": prompt},{"from": "gpt","value": None}], tokenizer, has_image=False).cuda()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=512,
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def generate_llama_3_2(args, tokenizer, model, prompt) -> str:
    prompt_ = "<|begin_of_text|>" + prompt
    inputs = tokenizer(None, prompt_, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, max_new_tokens=512)
    return(tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, ""))

def generate_internvl(args, tokenizer, model, prompt) -> str:
    generation_config = dict(max_new_tokens=512, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p)
    response, history = model.chat(tokenizer, None, prompt, generation_config, history=None, return_history=True)
    return response

def generate_qwen2_5vl(args, tokenizer, model, prompt) -> str:
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ],
    }
]
    # Preparation for inference
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return output_text[0]

def generate_cambrian(args, tokenizer, model, prompt) -> str:
    conv_modes = {
        "cambrian-phi3-3b":  "phi3",
        "cambrian-8b":  "llama_3",
        "cambrian-13b":  "vicuna_v1",
        "cambrian-34b":  "chatml_direct",
        "amasia-8b": "qwen_2",
        "amasia-8b-without-geo": "qwen_2",
    }

    conv_mode = conv_modes[cambrian_get_model_name_from_path(args.model_name_or_path)]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = cambrian_tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    output_ids = model.generate(
        input_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=512,
        use_cache=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def generate_qwen2(args, tokenizer, model, prompt) -> str:
    device = "cuda"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def generate_qwen2_5(args, tokenizer, model, prompt) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def generate_llama_3_0(args, tokenizer, model, prompt) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def generate_internlm(args, tokenizer, model, prompt) -> str:
    response, history = model.chat(tokenizer, prompt, max_new_tokens=512, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, history=[])
    return response

def generate_gemma_2(args, tokenizer, model, prompt) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

async def generate_gemini(args, tokenizer, model, prompt) -> str:
    async with semaphore:
        try_num = 3
        while try_num > 0:
            try_num -= 1
            try:
                response = await model.generate_content_async(
                    prompt,
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=512,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    ),
                )
                text = response.text.strip()
                return text
            except Exception as e:
                await asyncio.sleep(1)
        return "Error"

async def generate_gpt4o(args, tokenizer, model, prompt) -> str:
    async with semaphore:
        try_num = 3
        while try_num > 0:
            try_num -= 1
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_completion_tokens=512,
                )
                text = response.choices[0].message.content.strip()
                return text
            except Exception as e:
                await asyncio.sleep(1)
        return "Error"
