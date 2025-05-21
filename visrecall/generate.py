import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

import transformers
from generate_functions import (
    generate_cambrian,
    generate_gemini,
    generate_gemma_2,
    generate_gpt4o,
    generate_internlm,
    generate_internvl,
    generate_llama_3_0,
    generate_llama_3_2,
    generate_llava_1_5,
    generate_llava_ov,
    generate_pangea,
    generate_qwen2,
    generate_qwen2_5,
    generate_qwen2_5vl,
)
from init_models import (
    init_cambrian,
    init_gemini,
    init_gemma_2,
    init_gpt4o,
    init_internlm,
    init_internvl,
    init_llama_3_0,
    init_llama_3_2,
    init_llava_1_5,
    init_llava_ov,
    init_pangea,
    init_qwen2,
    init_qwen2_5,
    init_qwen2_5vl,
)
from jinja2 import Template
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

sys.path.append(str(Path("..").resolve()))
from utils.country_language_code import language_codes_visrecall as language_codes


def init_model(model_name_or_path: str):
    # multimodal models
    if model_name_or_path == "lmms-lab/llava-onevision-qwen2-7b-ov":
        return init_llava_ov()
    elif model_name_or_path == "liuhaotian/llava-v1.5-7b":
        return init_llava_1_5()
    elif model_name_or_path == "neulab/Pangea-7B":
        return init_pangea()
    elif model_name_or_path == "meta-llama/Llama-3.2-11B-Vision":
        return init_llama_3_2()
    elif model_name_or_path == "OpenGVLab/InternVL2_5-8B":
        return init_internvl()
    elif model_name_or_path == "Qwen/Qwen2.5-VL-7B-Instruct":
        return init_qwen2_5vl()
    elif "cambrian" in model_name_or_path:
        return init_cambrian(model_name_or_path)

    # text-only models
    if model_name_or_path == "Qwen/Qwen2-7B-Instruct":
        return init_qwen2()
    if model_name_or_path == "Qwen/Qwen2.5-7B-Instruct":
        return init_qwen2_5()
    elif model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        return init_llama_3_0()
    elif model_name_or_path == "internlm/internlm2_5-7b-chat":
        return init_internlm()
    elif model_name_or_path == "google/gemma-2-9b-it":
        return init_gemma_2()

    # proprietary models
    if "gemini" in model_name_or_path:
        return init_gemini(args)
    elif model_name_or_path == "gpt-4o-2024-11-20":
        return init_gpt4o()

    return None, None


def generate(
    args: argparse.Namespace,
    model_name_or_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    prompt: str,
) -> str:
    # multimodal models
    if model_name_or_path == "lmms-lab/llava-onevision-qwen2-7b-ov":
        return generate_llava_ov(args, tokenizer, model, prompt)
    elif model_name_or_path == "liuhaotian/llava-v1.5-7b":
        return generate_llava_1_5(args, tokenizer, model, prompt)
    elif model_name_or_path == "neulab/Pangea-7B":
        return generate_pangea(args, tokenizer, model, prompt)
    elif model_name_or_path == "meta-llama/Llama-3.2-11B-Vision":
        return generate_llama_3_2(args, tokenizer, model, prompt)
    elif model_name_or_path == "OpenGVLab/InternVL2_5-8B":
        return generate_internvl(args, tokenizer, model, prompt)
    elif model_name_or_path == "Qwen/Qwen2.5-VL-7B-Instruct":
        return generate_qwen2_5vl(args, tokenizer, model, prompt)
    elif "cambrian" in model_name_or_path or "amasia" in model_name_or_path:
        return generate_cambrian(args, tokenizer, model, prompt)

    # text-only models
    if model_name_or_path == "Qwen/Qwen2-7B-Instruct":
        return generate_qwen2(args, tokenizer, model, prompt)
    elif model_name_or_path == "Qwen/Qwen2.5-7B-Instruct":
        return generate_qwen2_5(args, tokenizer, model, prompt)
    elif model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        return generate_llama_3_0(args, tokenizer, model, prompt)
    elif model_name_or_path == "internlm/internlm2_5-7b-chat":
        return generate_internlm(args, tokenizer, model, prompt)
    elif model_name_or_path == "google/gemma-2-9b-it":
        return generate_gemma_2(args, tokenizer, model, prompt)

    return "Model not found"


async def generate_async(
    args: argparse.Namespace,
    model_name_or_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    prompt: str,
) -> str:
    # proprietary models
    if "gemini" in model_name_or_path:
        return await generate_gemini(args, tokenizer, model, prompt)
    if model_name_or_path == "gpt-4o-2024-11-20":
        return await generate_gpt4o(args, tokenizer, model, prompt)

    return "Model not found"


async def main():
    current_path = Path(__file__).parent.resolve()

    # read multilingual prompts
    prompt_templates = {}
    for lan in language_codes:
        prompt_templates[lan] = []
        for i in range(2):
            prompt_path = current_path / Path(f"prompts/{lan}/{i}.j2")
            prompt_templates[lan].append(Template(prompt_path.read_text(encoding="utf-8")))

    # read landmark data
    with open(current_path / Path("landmark_list.json")) as f:
        landmarks = json.load(f)

    # init model
    tokenizer, model = init_model(args.model_name_or_path)
    with contextlib.suppress(Exception):
        model = model.eval()

    # generate descriptions
    results = {}
    jobs = []
    for item in tqdm(landmarks):
        results[item["landmark_id"]] = {}
        for lan in language_codes:
            if args.debug and lan != "zh":
                continue

            results[item["landmark_id"]][lan] = []
            for i in range(2):
                prompt_template = prompt_templates[lan][i]
                prompt = prompt_template.render(landmark_name=item[f"wikipedia_{lan}"])

                if "gemini" in args.model_name_or_path or "gpt-4o" in args.model_name_or_path:
                    jobs.append(
                        generate_async(args, args.model_name_or_path, None, model, prompt),
                    )
                else:
                    model_output = generate(args, args.model_name_or_path, tokenizer, model, prompt)
                    results[item["landmark_id"]][lan].append(model_output.strip())

            if args.debug:
                break
        if args.debug:
            break

    if "gemini" in args.model_name_or_path or "gpt-4o" in args.model_name_or_path:
        model_outputs = await tqdm_asyncio.gather(*jobs)

        for landmark_id in results:
            for lan in results[landmark_id]:
                for _ in range(2):
                    results[landmark_id][lan].append(model_outputs.pop(0))

    # save results
    predictions_path = current_path / "predictions"
    predictions_path.mkdir(exist_ok=True)
    if args.do_sample:
        output_file = predictions_path / (
            f"{os.path.basename(args.model_name_or_path)}_"
            f"{args.temperature}_{args.top_p}_{args.num_beams}_{args.length_penalty}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    else:
        output_file = predictions_path / f"{os.path.basename(args.model_name_or_path)}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main())
