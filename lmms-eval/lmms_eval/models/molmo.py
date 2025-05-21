import torch

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, GenerationConfig
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger
from datetime import timedelta

@register_model("molmo")
class Molmo(lmms):
    """Molmo model wrapper for the LMMS evaluation framework.
    """

    def __init__(
        self,
        pretrained: str = "allenai/Molmo-7B-O-0924",
        device: str = "cuda:0",
        device_map="auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        # Load model.
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, device_map="auto", trust_remote_code=trust_remote_code, torch_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(pretrained, device_map="auto", trust_remote_code=trust_remote_code, torch_dtype=dtype)
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "batch_size_per_gpu > 1 is not supported for now."
        self.use_cache = use_cache
        if accelerator.num_processes > 1:
            distributed_type_list = [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            assert accelerator.distributed_type in distributed_type_list, "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            # self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            contexts = contexts + "Please answer the question as precisely as possible."
            
            model_inputs = self.processor.process(images=visuals,text=contexts)
            model_inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in model_inputs.items()}

            if model_inputs['input_ids'].size(1) > 4096:
                print(f"Input too long. This sample {doc_id} has size of {model_inputs['input_ids'].size(1)} and the model \
                    only supports up to 4096 tokens. Skipping this sample.")
                print("This is not perfect solution and might make minor differences in evaluations. Alternative approach is to truncate the input to 4096 tokens.")
                res.append("")
                pbar.update(1)
                continue

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "max_length" in gen_kwargs and "max_new_tokens" in gen_kwargs:
                gen_kwargs.pop("max_length")
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            
            generation_config = GenerationConfig.from_dict(gen_kwargs)

            generation_output = self.model.generate_from_batch(model_inputs, generation_config, stop_strings="<|endoftext|>", tokenizer=self.processor.tokenizer)
            generated_tokens = generation_output[0, model_inputs['input_ids'].size(1):]
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            assert type(response) == str, f"Expected response to be a string, but got {type(response)}"
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Molmo.")
