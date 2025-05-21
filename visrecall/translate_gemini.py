import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import google.generativeai as genai
from jinja2 import Template
from tqdm.asyncio import tqdm_asyncio

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")
semaphore = asyncio.Semaphore(10)


async def translate(prompt: str):
    async with semaphore:
        try_num = 10
        while try_num > 0:
            try_num -= 1
            try:
                response = await model.generate_content_async(prompt)
                json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
                matches = json_pattern.findall(response.text)
                if matches:
                    response = json.loads(matches[0].replace("\n", " "))
                return response["translation"]
            except Exception as e:
                print(e)
                await asyncio.sleep(1)
        return "Error"


async def main():
    current_path = Path(__file__).parent.resolve()

    with open(args.prediction_file) as f:
        predictions = json.load(f)
        predictions = dict(sorted(predictions.items(), key=lambda x: x[0]))

    prompt_template = Template(
        (current_path / Path("prompts/translate.j2")).read_text(encoding="utf-8"),
    )

    translated_predictions = {}
    jobs = []
    for landmark_id in predictions:
        translated_predictions[landmark_id] = {}
        prediction = predictions[landmark_id]

        for lan in prediction:
            if lan == "en":
                translated_predictions[landmark_id][lan] = prediction[lan]
                continue

            translated_predictions[landmark_id][lan] = []
            for pred in prediction[lan]:
                prompt = prompt_template.render(
                    description=pred.replace('"', "'"),
                )
                jobs.append(
                    translate(prompt),
                )

    responses = await tqdm_asyncio.gather(*jobs)
    for landmark_id in translated_predictions:
        for lan in translated_predictions[landmark_id]:
            if lan == "en":
                continue
            for _ in range(2):
                translated_predictions[landmark_id][lan].append(responses.pop(0))

    with open(args.prediction_file.replace(".json", "_translated_gemini.json"), "w") as f:
        json.dump(translated_predictions, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main())
