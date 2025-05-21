# Standard library imports
import asyncio
import base64
import os
from io import BytesIO

import numpy as np

# Related third-party imports
from accelerate import Accelerator
from loguru import logger as eval_logger
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# Local application/library specific imports
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Conditional imports
try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("Decord is not installed. Video input will not be supported.")

# Constants and global configurations
API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 1
semaphore = asyncio.Semaphore(25)

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY",  "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
else:
    API_URL = os.getenv("GEMINI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")


@register_model("asyncio_gpt4")
class AsyncioGPT4(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        api_key: str = API_KEY,
        api_url: str = API_URL,
        modality: str = "image",
        max_frames_num: int = 10,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.image_token = "<image>"
        self.timeout = timeout

        self.api_key = api_key
        self.api_url = api_url
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_url)

        accelerator = Accelerator()
        self.accelerator = accelerator
        assert accelerator.state.local_process_index == 0, "BatchGPT4 does not support distributed inference."
        assert accelerator.state.num_processes == 1, "BatchGPT4 does not support distributed inference."

    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        return asyncio.run(self.launch_batch(requests))

    async def launch_batch(self, requests):
        bar = async_tqdm(total=len(requests), desc="Batch Generating")
        jobs = []
        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in tqdm(enumerate([reg.args for reg in requests]), total=len(requests), desc="Preparing Batch"):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            imgs = []
            if visuals[0] is not None:
                visuals = self.flatten(visuals)
                for visual in visuals:
                    if self.modality == "image":
                        img = self.encode_image(visual)
                        imgs.append(img)
                    elif self.modality == "video":
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)

            messages = []
            if self.image_token not in contexts:
                messages.append({"role": "user", "content": []})
                messages[0]["content"].append(
                    {"type": "text", "text": contexts}
                )
                for img in imgs:
                    messages[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    messages.append({"role": "user", "content": []})
                    messages[idx]["content"].append(
                        {"type": "text", "text": contexts[idx]},
                    )
                    messages[idx]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )

            jobs.append(
                self.call_api
                (
                    {"model": self.model_version, "messages": messages, "max_tokens": gen_kwargs.get("max_new_tokens", 1024)},
                    bar,
                ),
            )

        res = await asyncio.gather(*jobs)
        return res

    async def call_api(self, item, bar):
        async with semaphore:
            try_num = 3
            text = "Error"
            while try_num > 0:
                try_num -= 1
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_version,
                        messages=item["messages"],
                        max_completion_tokens=item["max_tokens"],
                    )
                    text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(e)
                    await asyncio.sleep(NUM_SECONDS_TO_SLEEP)
            bar.update(1)
            return text

    def loglikelihood(self, requests):
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for BatchGPT4")
